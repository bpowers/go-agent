package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/persistence"
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
	appliedOpts := chat.ApplyOptions(opts...)
	callback := appliedOpts.StreamingCb

	// Simple mock response
	response := chat.Message{
		Role:    chat.AssistantRole,
		Content: fmt.Sprintf("Response to: %s", msg.Content),
	}

	// If callback is provided, simulate streaming
	if callback != nil {
		m.messageStreamCalls++
		for _, word := range strings.Fields(response.Content) {
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
	}

	m.messages = append(m.messages, msg)
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

type toolMockChat struct {
	mockChat
}

func (m *toolMockChat) Message(ctx context.Context, msg chat.Message, opts ...chat.Option) (chat.Message, error) {
	m.messageCalls++
	m.messages = append(m.messages, msg)

	toolCall := chat.Message{
		Role: chat.AssistantRole,
		ToolCalls: []chat.ToolCall{
			{
				ID:        "tool-call-1",
				Name:      "echo",
				Arguments: json.RawMessage(`{"message":"hi"}`),
			},
		},
	}

	toolResult := chat.Message{
		Role: chat.ToolRole,
		ToolResults: []chat.ToolResult{
			{
				ToolCallID: "tool-call-1",
				Name:       "echo",
				Content:    `{"result":"Echo: hi"}`,
			},
		},
		Content: `{"result":"Echo: hi"}`,
	}

	final := chat.Message{
		Role:    chat.AssistantRole,
		Content: "Echo complete",
	}

	m.messages = append(m.messages, toolCall, toolResult, final)

	usage := chat.TokenUsageDetails{InputTokens: 3, OutputTokens: 4, TotalTokens: 7}
	m.tokenUsage.LastMessage = usage
	m.tokenUsage.Cumulative.InputTokens += usage.InputTokens
	m.tokenUsage.Cumulative.OutputTokens += usage.OutputTokens
	m.tokenUsage.Cumulative.TotalTokens += usage.TotalTokens

	return final, nil
}

type toolClient struct {
	chat *toolMockChat
}

func (c *toolClient) NewChat(systemPrompt string, initialMsgs ...chat.Message) chat.Chat {
	newChat := &toolMockChat{}
	newChat.systemPrompt = systemPrompt
	newChat.messages = append([]chat.Message{}, initialMsgs...)
	newChat.maxTokens = 4096
	newChat.tools = make(map[string]func(context.Context, string) string)
	c.chat = newChat
	return newChat
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

	response, err := session.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: "Stream test",
	}, chat.WithStreamingCb(func(event chat.StreamEvent) error {
		if event.Type == chat.StreamEventTypeContent {
			streamedContent.WriteString(event.Content)
		}
		return nil
	}))

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

	// Verify there's a summary in the live records with assistant role
	foundSummary := false
	for _, r := range liveRecords {
		if strings.Contains(r.Content, "[Previous conversation summary]") {
			foundSummary = true
			// Verify summary is assistant role, not system
			assert.Equal(t, chat.AssistantRole, r.Role, "Summary should be assistant role")
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
	// LastMessage should have values from the mock
	assert.Greater(t, usage.LastMessage.InputTokens, 0)
	assert.Greater(t, usage.LastMessage.OutputTokens, 0)

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

	// Verify input tokens are stored in records
	records := session.LiveRecords()
	// Find the user message we just sent
	foundUserMsg := false
	for _, r := range records {
		if r.Role == chat.UserRole && r.Content == "Another test" {
			foundUserMsg = true
			assert.Greater(t, r.InputTokens, 0, "User message should have input tokens recorded")
			break
		}
	}
	assert.True(t, foundUserMsg, "Should find the user message in records")
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
	// The threshold is not exposed in metrics, but we can verify it doesn't panic

	// Test boundary values
	session.SetCompactionThreshold(0.0) // Should allow "never compact"
	session.SetCompactionThreshold(1.0)

	// Test out of range values (should be clamped)
	session.SetCompactionThreshold(-0.5)
	session.SetCompactionThreshold(1.5)
}

func TestSummarizerContextCancellation(t *testing.T) {
	// Test that context cancellation propagates to summarizer
	client := &mockClient{}

	// Create a summarizer that checks for context cancellation
	cancelCheckerSummarizer := &contextCheckingSummarizer{
		t: t,
	}

	session := NewSession(client, "System", WithSummarizer(cancelCheckerSummarizer))

	// Create a context that we'll cancel
	ctx, cancel := context.WithCancel(context.Background())

	// Add enough messages to trigger compaction
	session.SetCompactionThreshold(0.1)
	for i := 0; i < 5; i++ {
		_, err := session.Message(ctx, chat.Message{
			Role:    chat.UserRole,
			Content: strings.Repeat("Message ", 100),
		})
		require.NoError(t, err)
	}

	// Cancel the context before the next message (which would trigger compaction)
	cancel()

	// This should fail because context is cancelled
	_, err := session.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: strings.Repeat("Trigger compaction ", 200),
	})

	// Should get a context cancellation error
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "context canceled")
}

// contextCheckingSummarizer is a test summarizer that verifies context propagation
type contextCheckingSummarizer struct {
	t *testing.T
}

func (s *contextCheckingSummarizer) Summarize(ctx context.Context, records []Record) (string, error) {
	// Check if context is cancelled
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	default:
		// Context is not cancelled, return a simple summary
		return "Test summary", nil
	}
}

func (s *contextCheckingSummarizer) SetPrompt(prompt string) {
	// No-op for test
}

func TestCompactionThresholdZeroPersistence(t *testing.T) {
	// Test that a threshold of 0.0 can be persisted and isn't overwritten
	client := &mockClient{}
	store := persistence.NewMemoryStore()

	// Use the same session ID for both sessions to test persistence
	sessionID := "test-zero-threshold"

	// Create first session and set threshold to 0
	session1 := NewSession(client, "System", WithStore(store), WithRestoreSession(sessionID))
	session1.SetCompactionThreshold(0.0)

	// Create second session with same store and session ID
	session2 := NewSession(client, "System", WithStore(store), WithRestoreSession(sessionID))

	// Send messages to test that compaction doesn't occur
	ctx := context.Background()
	for i := 0; i < 20; i++ {
		_, err := session2.Message(ctx, chat.Message{
			Role:    chat.UserRole,
			Content: strings.Repeat("Long message ", 100),
		})
		require.NoError(t, err)
	}

	// With threshold 0.0, no compaction should occur
	metrics := session2.Metrics()
	assert.Equal(t, 0, metrics.CompactionCount, "No compaction should occur with threshold 0.0")
	assert.Equal(t, 41, metrics.RecordsLive) // System + 20*(user+assistant)
}

func TestRecordStatus(t *testing.T) {
	// Test that record status is properly tracked
	client := &mockClient{}
	store := persistence.NewMemoryStore()
	sessionID := "test-record-status"

	session := NewSession(client, "System", WithStore(store), WithRestoreSession(sessionID))

	// Send a successful message
	ctx := context.Background()
	_, err := session.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: "Test message",
	})
	require.NoError(t, err)

	// Check that records have success status
	records := session.LiveRecords()
	assert.Len(t, records, 3)                               // System, user, assistant
	assert.Equal(t, RecordStatusSuccess, records[0].Status) // System
	assert.Equal(t, RecordStatusSuccess, records[1].Status) // User
	assert.Equal(t, RecordStatusSuccess, records[2].Status) // Assistant

	// Test failed message by simulating error
	// We'd need to mock an error response from the client to fully test this
	// For now just verify the constants exist
	assert.Equal(t, RecordStatus("pending"), RecordStatusPending)
	assert.Equal(t, RecordStatus("success"), RecordStatusSuccess)
	assert.Equal(t, RecordStatus("failed"), RecordStatusFailed)
}

func TestGetRecordEfficiency(t *testing.T) {
	// Test that GetRecord works efficiently without O(N) scanning
	store := persistence.NewMemoryStore()
	sessionID := "test-get-record"

	// Add multiple records
	var ids []int64
	for i := 0; i < 100; i++ {
		id, err := store.AddRecord(sessionID, persistence.Record{
			Role:    "user",
			Content: fmt.Sprintf("Message %d", i),
			Live:    true,
		})
		require.NoError(t, err)
		ids = append(ids, id)
	}

	// Test retrieving various records by ID
	// Should work efficiently even with many records
	for i := 0; i < 10; i++ {
		idx := i * 10 // Test every 10th record
		record, err := store.GetRecord(sessionID, ids[idx])
		require.NoError(t, err)
		assert.Equal(t, ids[idx], record.ID)
		assert.Equal(t, fmt.Sprintf("Message %d", idx), record.Content)
	}

	// Test retrieving the last record (best case for backwards iteration)
	lastRecord, err := store.GetRecord(sessionID, ids[99])
	require.NoError(t, err)
	assert.Equal(t, ids[99], lastRecord.ID)
	assert.Equal(t, "Message 99", lastRecord.Content)

	// Test non-existent record
	_, err = store.GetRecord(sessionID, 99999)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "record not found")
}

func TestSessionPersistsToolInteractions(t *testing.T) {
	store := persistence.NewMemoryStore()
	client := &toolClient{}
	session := NewSession(client, "You are a tool tester", WithStore(store))

	_, err := session.Message(context.Background(), chat.Message{
		Role:    chat.UserRole,
		Content: "Trigger a tool call",
	})
	require.NoError(t, err)

	records := session.LiveRecords()
	require.Len(t, records, 5)
	assert.Equal(t, chat.UserRole, records[1].Role)
	assert.Equal(t, chat.AssistantRole, records[2].Role)
	require.Len(t, records[2].ToolCalls, 1)
	assert.Equal(t, "tool-call-1", records[2].ToolCalls[0].ID)
	assert.Equal(t, chat.ToolRole, records[3].Role)
	require.Len(t, records[3].ToolResults, 1)
	assert.Equal(t, "tool-call-1", records[3].ToolResults[0].ToolCallID)
	assert.Equal(t, chat.AssistantRole, records[4].Role)

	systemPrompt, history := session.History()
	assert.Equal(t, "You are a tool tester", systemPrompt)
	require.Len(t, history, 4)
	assert.Equal(t, chat.UserRole, history[0].Role)
	assert.Equal(t, chat.AssistantRole, history[1].Role)
	require.Len(t, history[1].ToolCalls, 1)
	assert.Equal(t, chat.ToolRole, history[2].Role)
	require.Len(t, history[2].ToolResults, 1)
	assert.Equal(t, chat.AssistantRole, history[3].Role)
}
