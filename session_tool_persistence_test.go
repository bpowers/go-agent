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
