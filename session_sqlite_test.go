package agent

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
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

	// Create session with SQLite persistence
	client := &mockClient{}
	session := NewSession(client, "Persistent assistant",
		WithStore(store))

	// Test basic messaging
	ctx := context.Background()
	response, err := session.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: "Hello persistent world",
	})
	require.NoError(t, err)
	assert.Contains(t, response.Content, "Hello persistent world")

	// Check records are persisted
	records := session.LiveRecords()
	assert.Len(t, records, 3) // System, user, assistant

	// Get metrics
	metrics := session.Metrics()
	assert.Equal(t, 3, metrics.RecordsLive)
	assert.Greater(t, metrics.CumulativeTokens, 0)

	// Create new session with same store to test persistence
	session2 := NewSession(client, "Should be ignored", // System prompt already in store
		WithStore(store))

	// Should have the same records
	records2 := session2.LiveRecords()
	assert.Len(t, records2, 3)
	assert.Equal(t, "Persistent assistant", records2[0].Content)
	assert.Equal(t, "Hello persistent world", records2[1].Content)
}

func TestSessionPersistenceAcrossRestarts(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "restart.db")

	// First session
	store1, err := sqlitestore.New(dbPath)
	require.NoError(t, err)

	client := &mockClient{}
	session1 := NewSession(client, "Persistent system",
		WithStore(store1))

	ctx := context.Background()

	// Add some messages
	for i := 0; i < 3; i++ {
		_, err := session1.Message(ctx, chat.Message{
			Role:    chat.UserRole,
			Content: "Message",
		})
		require.NoError(t, err)
	}

	metrics1 := session1.Metrics()
	tokens1 := metrics1.CumulativeTokens

	// Close first store
	store1.Close()

	// Create new store and session with same database
	store2, err := sqlitestore.New(dbPath)
	require.NoError(t, err)
	defer store2.Close()

	session2 := NewSession(client, "Persistent system", WithStore(store2))

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
		_, err := session.Message(ctx, chat.Message{
			Role:    chat.UserRole,
			Content: "Test message",
		})
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
