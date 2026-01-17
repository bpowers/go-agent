package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/persistence"
	"github.com/bpowers/go-agent/persistence/sqlitestore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func createTestDB(t *testing.T) (string, func()) {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	return dbPath, func() {}
}

func populateTestData(t *testing.T, dbPath string) {
	t.Helper()
	store, err := sqlitestore.New(dbPath)
	require.NoError(t, err)
	defer store.Close()

	sessionID := "session-abc123"
	baseTime := time.Date(2024, 1, 15, 10, 0, 0, 0, time.UTC)

	// User message with system reminder
	_, err = store.AddRecord(sessionID, persistence.Record{
		Role: chat.UserRole,
		Contents: []chat.Content{
			{Text: "What is 2+2?"},
			{SystemReminder: "Be helpful"},
		},
		Live:        true,
		Status:      persistence.RecordStatusSuccess,
		InputTokens: 10,
		Timestamp:   baseTime,
	})
	require.NoError(t, err)

	// Assistant message with tool call
	_, err = store.AddRecord(sessionID, persistence.Record{
		Role: chat.AssistantRole,
		Contents: []chat.Content{
			{Text: "Let me calculate."},
			{ToolCall: &chat.ToolCall{
				ID:        "call_123",
				Name:      "calculator",
				Arguments: json.RawMessage(`{"a": 2, "b": 2}`),
			}},
		},
		Live:         true,
		Status:       persistence.RecordStatusSuccess,
		OutputTokens: 20,
		Timestamp:    baseTime.Add(time.Second),
	})
	require.NoError(t, err)

	// Tool result
	_, err = store.AddRecord(sessionID, persistence.Record{
		Role: chat.ToolRole,
		Contents: []chat.Content{
			{ToolResult: &chat.ToolResult{
				ToolCallID: "call_123",
				Name:       "calculator",
				Content:    "4",
			}},
		},
		Live:      true,
		Status:    persistence.RecordStatusSuccess,
		Timestamp: baseTime.Add(2 * time.Second),
	})
	require.NoError(t, err)

	// Final assistant response
	_, err = store.AddRecord(sessionID, persistence.Record{
		Role: chat.AssistantRole,
		Contents: []chat.Content{
			{Text: "2+2 equals 4."},
		},
		Live:         true,
		Status:       persistence.RecordStatusSuccess,
		OutputTokens: 8,
		Timestamp:    baseTime.Add(3 * time.Second),
	})
	require.NoError(t, err)

	// Add a second session
	_, err = store.AddRecord("session-xyz789", persistence.Record{
		Role: chat.UserRole,
		Contents: []chat.Content{
			{Text: "Hello"},
		},
		Live:        true,
		Status:      persistence.RecordStatusSuccess,
		InputTokens: 5,
		Timestamp:   baseTime,
	})
	require.NoError(t, err)
}

func captureOutput(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	fn()

	w.Close()
	os.Stdout = old

	var buf bytes.Buffer
	io.Copy(&buf, r)
	return buf.String()
}

func TestRunList(t *testing.T) {
	dbPath, cleanup := createTestDB(t)
	defer cleanup()
	populateTestData(t, dbPath)

	output := captureOutput(t, func() {
		err := runList([]string{"--db", dbPath})
		require.NoError(t, err)
	})

	lines := strings.Split(strings.TrimSpace(output), "\n")
	assert.Len(t, lines, 2)
	assert.Contains(t, lines, "session-abc123")
	assert.Contains(t, lines, "session-xyz789")
}

func TestRunList_MissingDB(t *testing.T) {
	err := runList([]string{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "--db is required")
}

func TestRunList_NonexistentDB(t *testing.T) {
	err := runList([]string{"--db", "/nonexistent/path/db.sqlite"})
	assert.Error(t, err)
}

func TestRunShow_JSON(t *testing.T) {
	dbPath, cleanup := createTestDB(t)
	defer cleanup()
	populateTestData(t, dbPath)

	output := captureOutput(t, func() {
		err := runShow([]string{"--db", dbPath, "--session", "session-abc123", "--format", "json"})
		require.NoError(t, err)
	})

	var records []persistence.Record
	err := json.Unmarshal([]byte(output), &records)
	require.NoError(t, err)

	assert.Len(t, records, 4)

	// Verify chronological order
	assert.Equal(t, chat.UserRole, records[0].Role)
	assert.Equal(t, chat.AssistantRole, records[1].Role)
	assert.Equal(t, chat.ToolRole, records[2].Role)
	assert.Equal(t, chat.AssistantRole, records[3].Role)

	// Verify user message content
	assert.Equal(t, "What is 2+2?", records[0].Contents[0].Text)
	assert.Equal(t, "Be helpful", records[0].Contents[1].SystemReminder)

	// Verify tool call
	require.NotNil(t, records[1].Contents[1].ToolCall)
	assert.Equal(t, "calculator", records[1].Contents[1].ToolCall.Name)
	assert.Equal(t, "call_123", records[1].Contents[1].ToolCall.ID)

	// Verify tool result
	require.NotNil(t, records[2].Contents[0].ToolResult)
	assert.Equal(t, "4", records[2].Contents[0].ToolResult.Content)
}

func TestRunShow_JSONL(t *testing.T) {
	dbPath, cleanup := createTestDB(t)
	defer cleanup()
	populateTestData(t, dbPath)

	output := captureOutput(t, func() {
		err := runShow([]string{"--db", dbPath, "--session", "session-abc123", "--format", "jsonl"})
		require.NoError(t, err)
	})

	lines := strings.Split(strings.TrimSpace(output), "\n")
	assert.Len(t, lines, 4)

	// Each line should be valid JSON
	for i, line := range lines {
		var record persistence.Record
		err := json.Unmarshal([]byte(line), &record)
		require.NoError(t, err, "line %d should be valid JSON", i)
	}

	// Verify first record
	var first persistence.Record
	err := json.Unmarshal([]byte(lines[0]), &first)
	require.NoError(t, err)
	assert.Equal(t, chat.UserRole, first.Role)
}

func TestRunShow_MissingArgs(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{
			name: "missing db",
			args: []string{"--session", "abc"},
			want: "--db is required",
		},
		{
			name: "missing session",
			args: []string{"--db", "/tmp/test.db"},
			want: "--session is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := runShow(tt.args)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.want)
		})
	}
}

func TestRunShow_InvalidFormat(t *testing.T) {
	dbPath, cleanup := createTestDB(t)
	defer cleanup()

	err := runShow([]string{"--db", dbPath, "--session", "abc", "--format", "xml"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "--format must be 'json' or 'jsonl'")
}

func TestRunShow_EmptySession(t *testing.T) {
	dbPath, cleanup := createTestDB(t)
	defer cleanup()

	// Create empty DB
	store, err := sqlitestore.New(dbPath)
	require.NoError(t, err)
	store.Close()

	// Capture stderr for the "no records" message
	oldStderr := os.Stderr
	r, w, _ := os.Pipe()
	os.Stderr = w

	err = runShow([]string{"--db", dbPath, "--session", "nonexistent"})

	w.Close()
	os.Stderr = oldStderr

	var buf bytes.Buffer
	io.Copy(&buf, r)

	assert.NoError(t, err)
	assert.Contains(t, buf.String(), "no records found")
}

func TestRunShow_RecordsInChronologicalOrder(t *testing.T) {
	dbPath, cleanup := createTestDB(t)
	defer cleanup()
	populateTestData(t, dbPath)

	output := captureOutput(t, func() {
		err := runShow([]string{"--db", dbPath, "--session", "session-abc123"})
		require.NoError(t, err)
	})

	var records []persistence.Record
	err := json.Unmarshal([]byte(output), &records)
	require.NoError(t, err)

	// Verify timestamps are in order
	for i := 1; i < len(records); i++ {
		assert.True(t, records[i].Timestamp.After(records[i-1].Timestamp) ||
			records[i].Timestamp.Equal(records[i-1].Timestamp),
			"record %d should not be before record %d", i, i-1)
	}
}
