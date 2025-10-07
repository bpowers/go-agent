package agent

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/persistence"
	"github.com/bpowers/go-agent/persistence/sqlitestore"
)

// TestSessionToolCallPersistence verifies that tool calls round-trip correctly through SQLite persistence
func TestSessionToolCallPersistence(t *testing.T) {
	// Create a temporary database
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	// Create store
	store, err := sqlitestore.New(dbPath)
	require.NoError(t, err)
	defer store.Close()

	// Create a mock session with tool calls
	sessionID := "test-session"

	// Simulate what happens during a tool call round:
	// 1. User message
	userMsg := chat.UserMessage("test")
	userRecord := persistence.Record{
		Role:     chat.UserRole,
		Contents: userMsg.Contents,
		Live:     true,
		Status:   persistence.RecordStatusSuccess,
	}
	_, err = store.AddRecord(sessionID, userRecord)
	require.NoError(t, err)

	// 2. Assistant message with tool calls (built correctly without empty text block)
	assistantMsg := chat.Message{Role: chat.AssistantRole}
	assistantMsg.AddToolCall(chat.ToolCall{
		ID:        "toolu_test123",
		Name:      "test_tool",
		Arguments: json.RawMessage(`{"arg":"value"}`),
	})
	assistantRecord := persistence.Record{
		Role:     chat.AssistantRole,
		Contents: assistantMsg.Contents,
		Live:     true,
		Status:   persistence.RecordStatusSuccess,
	}
	_, err = store.AddRecord(sessionID, assistantRecord)
	require.NoError(t, err)

	// 3. Tool result message
	toolMsg := chat.Message{Role: chat.ToolRole}
	toolMsg.AddToolResult(chat.ToolResult{
		ToolCallID: "toolu_test123",
		Name:       "test_tool",
		Content:    "result",
	})
	toolRecord := persistence.Record{
		Role:     chat.ToolRole,
		Contents: toolMsg.Contents,
		Live:     true,
		Status:   persistence.RecordStatusSuccess,
	}
	_, err = store.AddRecord(sessionID, toolRecord)
	require.NoError(t, err)

	// Now load the records back
	records, err := store.GetLiveRecords(sessionID)
	require.NoError(t, err)
	require.Len(t, records, 3)

	// Verify the assistant message still has tool calls after persistence round-trip
	assistantLoaded := records[1]
	assert.Equal(t, chat.AssistantRole, assistantLoaded.Role)
	assert.NotEmpty(t, assistantLoaded.Contents, "Assistant message should have contents")

	// Check that we have at least one tool call
	hasToolCall := false
	hasEmptyContent := false
	for _, content := range assistantLoaded.Contents {
		if content.ToolCall != nil {
			hasToolCall = true
		}
		// Check for "empty" content blocks (all fields zero)
		if content.Text == "" && content.ToolCall == nil && content.ToolResult == nil && content.Thinking == nil {
			hasEmptyContent = true
		}
	}

	assert.True(t, hasToolCall, "Assistant message should have at least one tool call")

	// After the fix: there should be NO empty content blocks
	assert.False(t, hasEmptyContent, "Assistant message should not have empty content blocks")
	if hasEmptyContent {
		t.Errorf("Found empty content block in assistant message: %+v", assistantLoaded.Contents)
	}

	// Verify tool result message
	toolLoaded := records[2]
	assert.Equal(t, chat.ToolRole, toolLoaded.Role)
	assert.True(t, toolLoaded.HasToolResults())
}

// TestEmptyContentBlockSerialization tests that we properly avoid creating empty content blocks
func TestEmptyContentBlockSerialization(t *testing.T) {
	// Create a message like claude.go NOW does (after fix)
	msg := chat.Message{Role: chat.AssistantRole}
	// Don't add empty text
	msg.AddToolCall(chat.ToolCall{
		ID:        "test123",
		Name:      "test",
		Arguments: json.RawMessage(`{}`),
	})

	// Serialize to JSON (like SQLite store does)
	data, err := json.Marshal(msg.Contents)
	require.NoError(t, err)
	t.Logf("Serialized contents: %s", string(data))

	// Deserialize
	var contents []chat.Content
	err = json.Unmarshal(data, &contents)
	require.NoError(t, err)

	// Check what we got back
	t.Logf("Deserialized %d content blocks", len(contents))
	for i, c := range contents {
		isEmpty := c.Text == "" && c.ToolCall == nil && c.ToolResult == nil && c.Thinking == nil
		t.Logf("  Content[%d]: Text=%q, ToolCall=%v, isEmpty=%v", i, c.Text, c.ToolCall != nil, isEmpty)
	}

	// After the fix: should have only 1 content block (the tool call), no empty blocks
	require.Len(t, contents, 1, "Should have only 1 content block")
	assert.NotNil(t, contents[0].ToolCall, "First block has tool call")

	// Verify no empty blocks
	for i, c := range contents {
		isEmpty := c.Text == "" && c.ToolCall == nil && c.ToolResult == nil && c.Thinking == nil
		assert.False(t, isEmpty, "Content block %d should not be empty", i)
	}
}

// TestSystemReminderPersistence verifies that system reminders are persisted to DB but filtered when history is rebuilt
func TestSystemReminderPersistence(t *testing.T) {
	// Create a temporary database
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	// Create store
	store, err := sqlitestore.New(dbPath)
	require.NoError(t, err)
	defer store.Close()

	sessionID := "test-session-reminder"

	// Create a user message with system reminder (simulating what provider would persist)
	userMsg := chat.Message{Role: chat.UserRole}
	userMsg.Contents = []chat.Content{
		{SystemReminder: "<system-reminder>You modified 3 files</system-reminder>"},
		{Text: "What did I just do?"},
	}
	userRecord := persistence.Record{
		Role:     chat.UserRole,
		Contents: userMsg.Contents,
		Live:     true,
		Status:   persistence.RecordStatusSuccess,
	}
	_, err = store.AddRecord(sessionID, userRecord)
	require.NoError(t, err)

	// Create assistant response
	assistantMsg := chat.AssistantMessage("You modified 3 files.")
	assistantRecord := persistence.Record{
		Role:     chat.AssistantRole,
		Contents: assistantMsg.Contents,
		Live:     true,
		Status:   persistence.RecordStatusSuccess,
	}
	_, err = store.AddRecord(sessionID, assistantRecord)
	require.NoError(t, err)

	// Load records back from DB
	records, err := store.GetLiveRecords(sessionID)
	require.NoError(t, err)
	require.Len(t, records, 2)

	// Verify system reminder was persisted
	userLoaded := records[0]
	assert.Equal(t, chat.UserRole, userLoaded.Role)
	require.Len(t, userLoaded.Contents, 2, "User message should have 2 content blocks")

	// First content block should be the system reminder
	assert.Equal(t, "<system-reminder>You modified 3 files</system-reminder>", userLoaded.Contents[0].SystemReminder)
	assert.Empty(t, userLoaded.Contents[0].Text, "System reminder block should not have text")

	// Second content block should be the user text
	assert.Equal(t, "What did I just do?", userLoaded.Contents[1].Text)
	assert.Empty(t, userLoaded.Contents[1].SystemReminder, "Text block should not have system reminder")

	t.Log("✓ System reminder was persisted to DB correctly")

	// Now test that Session filters out SystemReminder content when rebuilding history
	// Create a mock client and session
	mockClient := &mockClient{}
	session := NewSession(mockClient, "test system", WithStore(store), WithRestoreSession(sessionID))

	// Get history - should filter out SystemReminder content
	_, history := session.History()
	require.Len(t, history, 2, "Should have 2 messages in history")

	// Verify user message has NO system reminder content
	userMsg = history[0]
	assert.Equal(t, chat.UserRole, userMsg.Role)
	require.Len(t, userMsg.Contents, 1, "User message should have only 1 content block (text, no system reminder)")
	assert.Equal(t, "What did I just do?", userMsg.Contents[0].Text)
	assert.Empty(t, userMsg.Contents[0].SystemReminder)

	t.Log("✓ System reminder was filtered out when Session rebuilt history")
}
