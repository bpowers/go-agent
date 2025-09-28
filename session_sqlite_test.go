package agent

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/llm/openai"
	"github.com/bpowers/go-agent/persistence/sqlitestore"
)

func TestSessionWithSQLiteStore(t *testing.T) {
	// Use a temporary file for this test
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "session.db")

	// Create SQLite store
	store, err := sqlitestore.New(dbPath)
	require.NoError(t, err)
	defer store.Close()

	// Use a fixed session ID for persistence testing
	sessionID := "test-session-persistence"

	// Create session with SQLite persistence
	client := &mockClient{}
	session := NewSession(client, "Persistent assistant",
		WithStore(store),
		WithRestoreSession(sessionID))

	// Test basic messaging
	ctx := context.Background()
	response, err := session.Message(ctx, chat.UserMessage("Hello persistent world"))
	require.NoError(t, err)
	assert.Contains(t, response.GetText(), "Hello persistent world")

	// Check records are persisted
	records := session.LiveRecords()
	assert.Len(t, records, 3) // System, user, assistant

	// Get metrics
	metrics := session.Metrics()
	assert.Equal(t, 3, metrics.RecordsLive)
	assert.Greater(t, metrics.CumulativeTokens, 0)

	// Create new session with same store and session ID to test persistence
	session2 := NewSession(client, "Should be ignored", // System prompt already in store
		WithStore(store),
		WithRestoreSession(sessionID))

	// Should have the same records
	records2 := session2.LiveRecords()
	assert.Len(t, records2, 3)
	assert.Equal(t, "Persistent assistant", records2[0].Content)
	assert.Equal(t, "Hello persistent world", records2[1].Content)
}

func TestSessionPersistenceAcrossRestarts(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "restart.db")

	// Use a fixed session ID for persistence testing
	sessionID := "test-session-restart"

	// First session
	store1, err := sqlitestore.New(dbPath)
	require.NoError(t, err)

	client := &mockClient{}
	session1 := NewSession(client, "Persistent system",
		WithStore(store1),
		WithRestoreSession(sessionID))

	ctx := context.Background()

	// Add some messages
	for i := 0; i < 3; i++ {
		_, err := session1.Message(ctx, chat.UserMessage("Message"))
		require.NoError(t, err)
	}

	metrics1 := session1.Metrics()
	tokens1 := metrics1.CumulativeTokens

	// Close first store
	store1.Close()

	// Create new store and session with same database and session ID
	store2, err := sqlitestore.New(dbPath)
	require.NoError(t, err)
	defer store2.Close()

	session2 := NewSession(client, "Persistent system",
		WithStore(store2),
		WithRestoreSession(sessionID))

	// Should have the same history
	records := session2.TotalRecords()
	assert.Equal(t, 7, len(records)) // System + 3*(user+assistant)

	// Metrics should be preserved
	metrics2 := session2.Metrics()
	assert.Equal(t, tokens1, metrics2.CumulativeTokens)
}

func TestSessionCompactionWithSQLite(t *testing.T) {
	store, err := sqlitestore.New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	client := &mockClient{}
	session := NewSession(client, "System",
		WithStore(store))

	// Lower threshold for testing
	session.SetCompactionThreshold(0.1)

	ctx := context.Background()

	// Add enough messages to trigger compaction
	for i := 0; i < 5; i++ {
		_, err := session.Message(ctx, chat.UserMessage("Test message"))
		require.NoError(t, err)
	}

	// Manually trigger compaction
	err = session.CompactNow()
	require.NoError(t, err)

	// Check that some records are dead
	allRecords := session.TotalRecords()
	liveRecords := session.LiveRecords()
	assert.Greater(t, len(allRecords), len(liveRecords))

	// Verify compaction metrics were saved
	metrics := session.Metrics()
	assert.Equal(t, 1, metrics.CompactionCount)
	assert.False(t, metrics.LastCompaction.IsZero())
}

func TestSessionResumption(t *testing.T) {
	// Constants for testing
	const userName = "bobby"
	const systemPrompt = "You are a helpful assistant. Remember important information from the conversation."

	// Use a temporary directory for this test
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "session_resume.db")

	var sessionID string

	// Phase 1: Create initial session and conversation
	{
		// Create SQLite store
		store, err := sqlitestore.New(dbPath)
		require.NoError(t, err)

		// Create a mock client for testing
		client := &mockClient{}

		// Create initial session
		session := NewSession(client, systemPrompt, WithStore(store))

		// Record the session ID for later restoration
		sessionID = session.SessionID()
		require.NotEmpty(t, sessionID, "Session ID should not be empty")

		ctx := context.Background()

		// Send first message introducing the name
		msg1 := fmt.Sprintf("Hello! My name is %s", userName)
		response1, err := session.Message(ctx, chat.UserMessage(msg1))
		require.NoError(t, err)
		assert.Contains(t, response1.GetText(), msg1)

		// Send second message with some other context
		msg2 := "I'm interested in learning about Go programming"
		response2, err := session.Message(ctx, chat.UserMessage(msg2))
		require.NoError(t, err)
		assert.Contains(t, response2.GetText(), msg2)

		// Verify we have the expected records
		records := session.LiveRecords()
		assert.Len(t, records, 5) // System + 2*(user+assistant)

		// Close the store to simulate session end
		store.Close()
	}

	// Phase 2: Force garbage collection to ensure no references remain
	for i := 0; i < 2; i++ {
		runtime.GC()
		runtime.Gosched()
	}

	// Phase 3: Resume session with same ID
	{
		// Create new store with same database
		store, err := sqlitestore.New(dbPath)
		require.NoError(t, err)
		defer store.Close()

		// Create a mock client for testing
		client := &mockClient{}

		// Resume session using WithRestoreSession
		resumedSession := NewSession(client, "Different prompt that should be ignored",
			WithStore(store),
			WithRestoreSession(sessionID))

		// Verify the session ID matches
		assert.Equal(t, sessionID, resumedSession.SessionID())

		// Verify the history was restored
		records := resumedSession.LiveRecords()
		assert.Len(t, records, 5) // Should have all the previous records

		// Verify the system prompt was preserved from the original session
		assert.Equal(t, systemPrompt, records[0].Content)

		// Verify the user's name is in the history
		foundNameMessage := false
		for _, record := range records {
			if record.Role == chat.UserRole && strings.Contains(record.Content, userName) {
				foundNameMessage = true
				break
			}
		}
		assert.True(t, foundNameMessage, "Should find the message with user's name in history")

		// Now ask the LLM to recall the user's name
		ctx := context.Background()
		response, err := resumedSession.Message(ctx, chat.UserMessage("What is my name? Reply with just the name in one word."))
		require.NoError(t, err)

		// The mock client echoes back the content, so check if it contains the question
		// In a real scenario with an actual LLM, we'd check if it recalls "bobby"
		// For our mock, we just verify the message was processed
		assert.Contains(t, response.GetText(), "What is my name")

		// Verify cumulative tokens were preserved
		metrics := resumedSession.Metrics()
		assert.Greater(t, metrics.CumulativeTokens, 0)
		assert.Equal(t, 7, metrics.RecordsTotal) // Original 5 + new question + answer
	}
}

func TestSessionResumptionWithLLM(t *testing.T) {
	// Skip if no API key
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("Skipping LLM integration test: OPENAI_API_KEY not set")
	}

	// Constants for testing
	const userName = "bobby"
	const systemPrompt = "You are a helpful assistant. Remember important information from the conversation."

	// Use a temporary directory for this test
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "session_llm_resume.db")

	var sessionID string

	// Phase 1: Create initial session and conversation with real LLM
	{
		// Create SQLite store
		store, err := sqlitestore.New(dbPath)
		require.NoError(t, err)

		// Create OpenAI client with gpt-4o-mini
		client, err := openai.NewClient(
			openai.OpenAIURL,
			os.Getenv("OPENAI_API_KEY"),
			openai.WithModel("gpt-4o-mini"),
		)
		require.NoError(t, err)

		// Create initial session
		session := NewSession(client, systemPrompt, WithStore(store))

		// Record the session ID for later restoration
		sessionID = session.SessionID()
		require.NotEmpty(t, sessionID, "Session ID should not be empty")

		ctx := context.Background()

		// Send first message introducing the name
		msg1 := fmt.Sprintf("Hello! My name is %s. Please remember this for later.", userName)
		response1, err := session.Message(ctx, chat.UserMessage(msg1))
		require.NoError(t, err)
		require.NotEmpty(t, response1.GetText())

		// Send second message with some other context
		msg2 := "I'm interested in learning about Go programming. What's a good starting point?"
		response2, err := session.Message(ctx, chat.UserMessage(msg2))
		require.NoError(t, err)
		require.NotEmpty(t, response2.GetText())

		// Verify we have the expected records
		records := session.LiveRecords()
		assert.Len(t, records, 5) // System + 2*(user+assistant)

		// Close the store to simulate session end
		store.Close()
	}

	// Phase 2: Force garbage collection to ensure no references remain
	for i := 0; i < 2; i++ {
		runtime.GC()
		runtime.Gosched()
	}

	// Phase 3: Resume session with same ID
	{
		// Create new store with same database
		store, err := sqlitestore.New(dbPath)
		require.NoError(t, err)
		defer store.Close()

		// Create OpenAI client with gpt-4o-mini
		client, err := openai.NewClient(
			openai.OpenAIURL,
			os.Getenv("OPENAI_API_KEY"),
			openai.WithModel("gpt-4o-mini"),
		)
		require.NoError(t, err)

		// Resume session using WithRestoreSession
		resumedSession := NewSession(client, "Different prompt that should be ignored",
			WithStore(store),
			WithRestoreSession(sessionID))

		// Verify the session ID matches
		assert.Equal(t, sessionID, resumedSession.SessionID())

		// Verify the history was restored
		records := resumedSession.LiveRecords()
		assert.Len(t, records, 5) // Should have all the previous records

		// Verify the system prompt was preserved from the original session
		assert.Equal(t, systemPrompt, records[0].Content)

		// Now ask the LLM to recall the user's name
		ctx := context.Background()
		response, err := resumedSession.Message(ctx, chat.UserMessage("What is my name? Reply with just the name in one word."))
		require.NoError(t, err)

		// The LLM should recall the name "bobby" from the previous conversation
		responseLower := strings.ToLower(response.GetText())
		assert.Contains(t, responseLower, strings.ToLower(userName),
			"LLM should recall the name %s from the resumed session, got: %s", userName, response.GetText())

		// Verify cumulative tokens were preserved and increased
		metrics := resumedSession.Metrics()
		assert.Greater(t, metrics.CumulativeTokens, 0)
		assert.Equal(t, 7, metrics.RecordsTotal) // Original 5 + new question + answer
	}
}
