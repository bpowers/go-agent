package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
)

// estimateTokens provides a simple token count estimate for testing
func estimateTokens(content string) int {
	// Simple estimate: ~4 characters per token
	return len(content) / 4
}

// mockChat implements chat.Chat for testing
type mockChat struct {
	systemPrompt string
	messages     []chat.Message
	tools        map[string]func(context.Context, string) string
	maxTokens    int
	tokenUsage   chat.TokenUsage

	// Track calls for assertions
	messageCalls       int
	messageStreamCalls int
}

func (m *mockChat) Message(ctx context.Context, msg chat.Message, opts ...chat.Option) (chat.Message, error) {
	m.messageCalls++
	m.messages = append(m.messages, msg)

	// Simple mock response
	response := chat.Message{
		Role:    chat.AssistantRole,
		Content: fmt.Sprintf("Response to: %s", msg.Content),
	}
	m.messages = append(m.messages, response)

	// Update token usage - new format with LastMessage and Cumulative
	inputTokens := estimateTokens(msg.Content)
	outputTokens := estimateTokens(response.Content)

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

func (m *mockChat) MessageStream(ctx context.Context, msg chat.Message, callback chat.StreamCallback, opts ...chat.Option) (chat.Message, error) {
	m.messageStreamCalls++

	// Simulate streaming
	response := fmt.Sprintf("Response to: %s", msg.Content)
	for _, word := range strings.Fields(response) {
		if err := callback(chat.StreamEvent{
			Type:    chat.StreamEventTypeContent,
			Content: word + " ",
		}); err != nil {
			return chat.Message{}, err
		}
	}

	// Send done event
	callback(chat.StreamEvent{
		Type: chat.StreamEventTypeDone,
	})

	fullResponse := chat.Message{
		Role:    chat.AssistantRole,
		Content: response,
	}

	m.messages = append(m.messages, msg)
	m.messages = append(m.messages, fullResponse)

	// Update token usage - new format with LastMessage and Cumulative
	inputTokens := estimateTokens(msg.Content)
	outputTokens := estimateTokens(response)

	m.tokenUsage.LastMessage = chat.TokenUsageDetails{
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		TotalTokens:  inputTokens + outputTokens,
	}

	m.tokenUsage.Cumulative.InputTokens += inputTokens
	m.tokenUsage.Cumulative.OutputTokens += outputTokens
	m.tokenUsage.Cumulative.TotalTokens = m.tokenUsage.Cumulative.InputTokens + m.tokenUsage.Cumulative.OutputTokens

	return fullResponse, nil
}

func (m *mockChat) History() (systemPrompt string, msgs []chat.Message) {
	return m.systemPrompt, m.messages
}

func (m *mockChat) TokenUsage() (chat.TokenUsage, error) {
	return m.tokenUsage, nil
}

func (m *mockChat) MaxTokens() int {
	if m.maxTokens == 0 {
		return 4096 // Default for testing
	}
	return m.maxTokens
}

func (m *mockChat) RegisterTool(def chat.ToolDef, fn func(context.Context, string) string) error {
	if m.tools == nil {
		m.tools = make(map[string]func(context.Context, string) string)
	}
	m.tools[def.Name()] = fn
	return nil
}

func (m *mockChat) DeregisterTool(name string) {
	delete(m.tools, name)
}

func (m *mockChat) ListTools() []string {
	var names []string
	for name := range m.tools {
		names = append(names, name)
	}
	return names
}

// mockClient implements chat.Client for testing
type mockClient struct {
	chats []*mockChat
}

func (c *mockClient) NewChat(systemPrompt string, initialMsgs ...chat.Message) chat.Chat {
	chat := &mockChat{
		systemPrompt: systemPrompt,
		messages:     append([]chat.Message{}, initialMsgs...),
		maxTokens:    4096,
		tools:        make(map[string]func(context.Context, string) string),
	}
	c.chats = append(c.chats, chat)
	return chat
}

// mockToolDef implements chat.ToolDef for testing
type mockToolDef struct {
	name        string
	description string
	schema      string
}

func (t *mockToolDef) Name() string          { return t.name }
func (t *mockToolDef) Description() string   { return t.description }
func (t *mockToolDef) MCPJsonSchema() string { return t.schema }

// Tests

func TestSessionBasics(t *testing.T) {
	client := &mockClient{}
	session := NewSession(client, "You are a helpful assistant")

	// Test that Session implements chat.Chat
	var _ chat.Chat = session

	// Test initial state
	assert.Equal(t, 4096, session.MaxTokens())

	metrics := session.Metrics()
	assert.Equal(t, 0, metrics.CumulativeTokens)
	assert.Equal(t, 1, metrics.RecordsLive) // System prompt
	assert.Equal(t, 1, metrics.RecordsTotal)

	// Test sending a message
	ctx := context.Background()
	response, err := session.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: "Hello",
	})
	require.NoError(t, err)
	assert.Equal(t, chat.AssistantRole, response.Role)
	assert.Contains(t, response.Content, "Hello")

	// Check records
	records := session.LiveRecords()
	assert.Len(t, records, 3) // System, user, assistant
	assert.Equal(t, "system", string(records[0].Role))
	assert.Equal(t, chat.UserRole, records[1].Role)
	assert.Equal(t, chat.AssistantRole, records[2].Role)
}

func TestSessionStreaming(t *testing.T) {
	client := &mockClient{}
	session := NewSession(client, "You are a helpful assistant")

	ctx := context.Background()
	var streamedContent strings.Builder

	response, err := session.MessageStream(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: "Stream test",
	}, func(event chat.StreamEvent) error {
		if event.Type == chat.StreamEventTypeContent {
			streamedContent.WriteString(event.Content)
		}
		return nil
	})

	require.NoError(t, err)
	assert.Contains(t, response.Content, "Stream test")
	assert.Contains(t, streamedContent.String(), "Stream test")

	// Check that it was recorded
	records := session.LiveRecords()
	assert.Len(t, records, 3)
	assert.Equal(t, "Stream test", records[1].Content)
}

func TestSessionHistory(t *testing.T) {
	client := &mockClient{}
	initialMsgs := []chat.Message{
		{Role: chat.UserRole, Content: "Initial message"},
		{Role: chat.AssistantRole, Content: "Initial response"},
	}

	session := NewSession(client, "System prompt",
		WithInitialMessages(initialMsgs...))

	systemPrompt, msgs := session.History()
	assert.Equal(t, "System prompt", systemPrompt)
	assert.Len(t, msgs, 2)
	assert.Equal(t, "Initial message", msgs[0].Content)
	assert.Equal(t, "Initial response", msgs[1].Content)

	// Add more messages
	ctx := context.Background()
	session.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: "Another message",
	})

	systemPrompt, msgs = session.History()
	assert.Equal(t, "System prompt", systemPrompt)
	assert.Len(t, msgs, 4) // Initial 2 + new user + assistant
}

func TestSessionTools(t *testing.T) {
	client := &mockClient{}
	session := NewSession(client, "You are a helpful assistant")

	// Register a tool
	toolDef := &mockToolDef{
		name:        "test_tool",
		description: "A test tool",
		schema:      `{"type": "object"}`,
	}

	err := session.RegisterTool(toolDef, func(ctx context.Context, args string) string {
		return "Tool result"
	})
	require.NoError(t, err)

	// Check tool is registered
	tools := session.ListTools()
	assert.Contains(t, tools, "test_tool")

	// Deregister tool
	session.DeregisterTool("test_tool")
	tools = session.ListTools()
	assert.NotContains(t, tools, "test_tool")
}

func TestSessionMetrics(t *testing.T) {
	client := &mockClient{}
	session := NewSession(client, "System")

	ctx := context.Background()

	// Send several messages
	for i := 0; i < 3; i++ {
		_, err := session.Message(ctx, chat.Message{
			Role:    chat.UserRole,
			Content: fmt.Sprintf("Message %d", i),
		})
		require.NoError(t, err)
	}

	metrics := session.Metrics()
	assert.Equal(t, 7, metrics.RecordsLive) // System + 3*(user+assistant)
	assert.Equal(t, 7, metrics.RecordsTotal)
	assert.Greater(t, metrics.LiveTokens, 0)
	assert.Less(t, metrics.PercentFull, 1.0)
}

func TestSessionCompaction(t *testing.T) {
	client := &mockClient{}
	session := NewSession(client, "System prompt")

	// Lower the threshold for testing
	session.SetCompactionThreshold(0.1) // Compact at 10% full

	ctx := context.Background()

	// Add enough messages to trigger compaction
	for i := 0; i < 10; i++ {
		_, err := session.Message(ctx, chat.Message{
			Role:    chat.UserRole,
			Content: strings.Repeat("Long message ", 100), // Make it long to use more tokens
		})
		require.NoError(t, err)
	}

	// Check that some records are marked as dead after compaction
	allRecords := session.TotalRecords()
	liveRecords := session.LiveRecords()

	assert.Greater(t, len(allRecords), len(liveRecords))

	// Check metrics show compaction occurred
	metrics := session.Metrics()
	assert.Greater(t, metrics.CompactionCount, 0)
	assert.False(t, metrics.LastCompaction.IsZero())
}

func TestManualCompaction(t *testing.T) {
	client := &mockClient{}
	session := NewSession(client, "System")

	ctx := context.Background()

	// Add several messages
	for i := 0; i < 5; i++ {
		_, err := session.Message(ctx, chat.Message{
			Role:    chat.UserRole,
			Content: fmt.Sprintf("Message %d with some content", i),
		})
		require.NoError(t, err)
	}

	// Manually trigger compaction
	err := session.CompactNow()
	require.NoError(t, err)

	// Check that compaction occurred
	metrics := session.Metrics()
	assert.Equal(t, 1, metrics.CompactionCount)

	// Verify some records are dead
	allRecords := session.TotalRecords()
	liveRecords := session.LiveRecords()
	assert.Greater(t, len(allRecords), len(liveRecords))

	// Verify there's a summary in the live records
	foundSummary := false
	for _, r := range liveRecords {
		if strings.Contains(r.Content, "[Previous conversation summary]") {
			foundSummary = true
			break
		}
	}
	assert.True(t, foundSummary, "Should have a summary record")
}

func TestSessionTokenTracking(t *testing.T) {
	client := &mockClient{}
	session := NewSession(client, "System")

	ctx := context.Background()

	// Send a message
	_, err := session.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: "Test message",
	})
	require.NoError(t, err)

	// Check token usage
	usage, err := session.TokenUsage()
	require.NoError(t, err)
	assert.Greater(t, usage.Cumulative.TotalTokens, 0)

	// Send another message
	_, err = session.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: "Another test",
	})
	require.NoError(t, err)

	// Token usage should increase
	newUsage, err := session.TokenUsage()
	require.NoError(t, err)
	assert.Greater(t, newUsage.Cumulative.TotalTokens, usage.Cumulative.TotalTokens)
}

func TestSessionRecordTimestamps(t *testing.T) {
	client := &mockClient{}
	session := NewSession(client, "System")

	ctx := context.Background()

	// Send a message
	_, err := session.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: "Test",
	})
	require.NoError(t, err)

	// Check that records have timestamps
	records := session.LiveRecords()
	for _, r := range records {
		assert.False(t, r.Timestamp.IsZero(), "Record should have timestamp")
	}

	// Timestamps should be in order
	for i := 1; i < len(records); i++ {
		assert.True(t, records[i].Timestamp.After(records[i-1].Timestamp) ||
			records[i].Timestamp.Equal(records[i-1].Timestamp),
			"Timestamps should be in chronological order")
	}
}

func TestSessionWithInitialMessages(t *testing.T) {
	client := &mockClient{}

	initialMsgs := []chat.Message{
		{Role: chat.UserRole, Content: "First message"},
		{Role: chat.AssistantRole, Content: "First response"},
		{Role: chat.UserRole, Content: "Second message"},
		{Role: chat.AssistantRole, Content: "Second response"},
	}

	session := NewSession(client, "System",
		WithInitialMessages(initialMsgs...))

	// Check initial records
	records := session.LiveRecords()
	assert.Len(t, records, 5) // System + 4 initial messages

	assert.Equal(t, "system", string(records[0].Role))
	assert.Equal(t, "First message", records[1].Content)
	assert.Equal(t, "First response", records[2].Content)
	assert.Equal(t, "Second message", records[3].Content)
	assert.Equal(t, "Second response", records[4].Content)

	// Verify all are marked as live
	for _, r := range records {
		assert.True(t, r.Live)
	}
}

func TestCompactionThreshold(t *testing.T) {
	client := &mockClient{}
	session := NewSession(client, "System")

	// Test setting valid thresholds
	session.SetCompactionThreshold(0.5)
	// Verify it doesn't panic

	// Test boundary values
	session.SetCompactionThreshold(0.0)
	session.SetCompactionThreshold(1.0)

	// Test out of range values (should be clamped)
	session.SetCompactionThreshold(-0.5)
	session.SetCompactionThreshold(1.5)
}
