package agent

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/bpowers/go-agent/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockSystemReminderClient tracks system reminder calls
type mockSystemReminderClient struct {
	mu                     sync.Mutex
	contextsSeen           []context.Context // Track all contexts passed to NewChat.Message
	systemRemindersPresent []bool            // Track whether system reminder was present in each context
}

func (m *mockSystemReminderClient) NewChat(systemPrompt string, initialMsgs ...chat.Message) chat.Chat {
	return &mockSystemReminderChat{
		client:       m,
		systemPrompt: systemPrompt,
		messages:     append([]chat.Message{}, initialMsgs...),
		tools:        make(map[string]func(context.Context, string) string),
	}
}

// mockSystemReminderChat implements chat.Chat and tracks context with system reminders
type mockSystemReminderChat struct {
	mu           sync.Mutex
	client       *mockSystemReminderClient
	systemPrompt string
	messages     []chat.Message
	tools        map[string]func(context.Context, string) string
	tokenUsage   chat.TokenUsage
}

func (m *mockSystemReminderChat) Message(ctx context.Context, msg chat.Message, opts ...chat.Option) (chat.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Track the context to see if it has a system reminder
	m.client.mu.Lock()
	m.client.contextsSeen = append(m.client.contextsSeen, ctx)
	reminderFunc := chat.GetSystemReminder(ctx)
	m.client.systemRemindersPresent = append(m.client.systemRemindersPresent, reminderFunc != nil)
	m.client.mu.Unlock()

	// Create simple response
	response := chat.AssistantMessage("Response to: " + msg.GetText())

	m.messages = append(m.messages, msg)
	m.messages = append(m.messages, response)

	// Update token usage
	inputTokens := len(msg.GetText()) / 4
	outputTokens := len(response.GetText()) / 4

	m.tokenUsage.LastMessage = chat.TokenUsageDetails{
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		TotalTokens:  inputTokens + outputTokens,
	}

	m.tokenUsage.Cumulative.InputTokens += inputTokens
	m.tokenUsage.Cumulative.OutputTokens += outputTokens
	m.tokenUsage.Cumulative.TotalTokens = m.tokenUsage.Cumulative.InputTokens + m.tokenUsage.Cumulative.OutputTokens

	return response, nil
}

func (m *mockSystemReminderChat) History() (string, []chat.Message) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.systemPrompt, m.messages
}

func (m *mockSystemReminderChat) TokenUsage() (chat.TokenUsage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.tokenUsage, nil
}

func (m *mockSystemReminderChat) MaxTokens() int {
	return 4096
}

func (m *mockSystemReminderChat) RegisterTool(tool chat.Tool) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tools[tool.Name()] = tool.Call
	return nil
}

func (m *mockSystemReminderChat) DeregisterTool(name string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.tools, name)
}

func (m *mockSystemReminderChat) ListTools() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	tools := make([]string, 0, len(m.tools))
	for name := range m.tools {
		tools = append(tools, name)
	}
	return tools
}

// TestSessionPreservesSystemReminder tests that system reminders in the context
// are properly passed through to the underlying chat client's Message method
func TestSessionPreservesSystemReminder(t *testing.T) {
	t.Parallel()

	client := &mockSystemReminderClient{}
	session, err := NewSession(client, "Test system prompt")
	require.NoError(t, err)

	// Create context with a system reminder
	reminderText := "<system-reminder>User is viewing: models/test.sd.json</system-reminder>"
	ctx := chat.WithSystemReminder(context.Background(), func() string {
		return reminderText
	})

	// Send a message with the system reminder context
	_, err = session.Message(ctx, chat.UserMessage("Tell me about this model"))
	require.NoError(t, err)

	// Verify that the context with system reminder was seen by the underlying chat
	client.mu.Lock()
	defer client.mu.Unlock()

	require.Len(t, client.contextsSeen, 1, "Should have seen one context passed to Message")
	require.Len(t, client.systemRemindersPresent, 1, "Should have tracked one system reminder presence")

	// The critical assertion: the system reminder should be present in the context
	// that was passed to the underlying chat's Message method
	assert.True(t, client.systemRemindersPresent[0],
		"System reminder should be present in context passed to underlying chat.Message()")

	// Also verify we can actually retrieve the reminder
	reminderFunc := chat.GetSystemReminder(client.contextsSeen[0])
	require.NotNil(t, reminderFunc, "Should be able to retrieve system reminder from context")
	assert.Equal(t, reminderText, reminderFunc(), "System reminder text should match")
}

// TestSessionPreservesSystemReminderAcrossMultipleMessages tests that system reminders
// work correctly across multiple message exchanges
func TestSessionPreservesSystemReminderAcrossMultipleMessages(t *testing.T) {
	t.Parallel()

	client := &mockSystemReminderClient{}
	session, err := NewSession(client, "Test system prompt")
	require.NoError(t, err)

	// First message with system reminder
	ctx1 := chat.WithSystemReminder(context.Background(), func() string {
		return "<system-reminder>Viewing: model1.sd.json</system-reminder>"
	})

	_, err = session.Message(ctx1, chat.UserMessage("First question"))
	require.NoError(t, err)

	// Second message with different system reminder
	ctx2 := chat.WithSystemReminder(context.Background(), func() string {
		return "<system-reminder>Viewing: model2.sd.json</system-reminder>"
	})

	_, err = session.Message(ctx2, chat.UserMessage("Second question"))
	require.NoError(t, err)

	// Third message without system reminder
	ctx3 := context.Background()

	_, err = session.Message(ctx3, chat.UserMessage("Third question"))
	require.NoError(t, err)

	// Verify all contexts were tracked
	client.mu.Lock()
	defer client.mu.Unlock()

	require.Len(t, client.contextsSeen, 3, "Should have seen three contexts")
	require.Len(t, client.systemRemindersPresent, 3, "Should have tracked three system reminder presences")

	// First message should have system reminder
	assert.True(t, client.systemRemindersPresent[0],
		"First message should have system reminder in context")

	// Second message should have system reminder
	assert.True(t, client.systemRemindersPresent[1],
		"Second message should have system reminder in context")

	// Third message should NOT have system reminder
	assert.False(t, client.systemRemindersPresent[2],
		"Third message should not have system reminder in context")

	// Verify the actual reminder text for first two messages
	reminder1 := chat.GetSystemReminder(client.contextsSeen[0])
	require.NotNil(t, reminder1)
	assert.True(t, strings.Contains(reminder1(), "model1.sd.json"),
		"First reminder should reference model1")

	reminder2 := chat.GetSystemReminder(client.contextsSeen[1])
	require.NotNil(t, reminder2)
	assert.True(t, strings.Contains(reminder2(), "model2.sd.json"),
		"Second reminder should reference model2")
}
