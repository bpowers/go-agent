package sqlitestore

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/persistence"
)

func TestSQLiteStoreBasics(t *testing.T) {
	// Use in-memory database for testing
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	sessionID := "test-session"

	// Test adding a record
	record := persistence.Record{
		Role: chat.UserRole,
		Contents: []chat.Content{
			{Text: "Test message"},
		},
		Live:         true,
		Status:       persistence.RecordStatusSuccess,
		InputTokens:  7,
		OutputTokens: 3,
		Timestamp:    time.Now(),
	}

	id, err := store.AddRecord(sessionID, record)
	require.NoError(t, err)
	assert.Greater(t, id, int64(0))

	// Test getting a single record by ID
	retrieved, err := store.GetRecord(sessionID, id)
	require.NoError(t, err)
	assert.Equal(t, id, retrieved.ID)
	assert.Equal(t, "Test message", retrieved.GetText())
	assert.Equal(t, chat.UserRole, retrieved.Role)
	assert.True(t, retrieved.Live)

	// Test getting non-existent record
	_, err = store.GetRecord(sessionID, 99999)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "record not found")

	// Test getting all records
	records, err := store.GetAllRecords(sessionID)
	require.NoError(t, err)
	assert.Len(t, records, 1)
	assert.Equal(t, "Test message", records[0].GetText())
	assert.Equal(t, chat.UserRole, records[0].Role)
	assert.True(t, records[0].Live)

	// Test getting live records
	liveRecords, err := store.GetLiveRecords(sessionID)
	require.NoError(t, err)
	assert.Len(t, liveRecords, 1)
}

func TestSQLiteStoreUpdateRecord(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	sessionID := "test-session"

	// Add a record
	record := persistence.Record{
		Role: chat.UserRole,
		Contents: []chat.Content{
			{Text: "Original"},
		},
		Live:         true,
		Status:       persistence.RecordStatusSuccess,
		InputTokens:  3,
		OutputTokens: 2,
		Timestamp:    time.Now(),
	}

	id, err := store.AddRecord(sessionID, record)
	require.NoError(t, err)

	// Update the record
	record.Contents = []chat.Content{{Text: "Updated"}}
	record.InputTokens = 5
	record.OutputTokens = 2
	err = store.UpdateRecord(sessionID, id, record)
	require.NoError(t, err)

	// Verify update
	records, err := store.GetAllRecords(sessionID)
	require.NoError(t, err)
	assert.Len(t, records, 1)
	assert.Equal(t, "Updated", records[0].GetText())
	assert.Equal(t, 5, records[0].InputTokens)
	assert.Equal(t, 2, records[0].OutputTokens)
}

func TestSQLiteStoreMarkLiveDead(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	sessionID := "test-session"

	// Add multiple records
	for i := 0; i < 3; i++ {
		record := persistence.Record{
			Role: chat.UserRole,
			Contents: []chat.Content{
				{Text: "Message"},
			},
			Live:         true,
			Status:       persistence.RecordStatusSuccess,
			InputTokens:  6,
			OutputTokens: 4,
			Timestamp:    time.Now(),
		}
		_, err := store.AddRecord(sessionID, record)
		require.NoError(t, err)
	}

	// Mark first record as dead
	err = store.MarkRecordDead(sessionID, 1)
	require.NoError(t, err)

	// Check live records
	liveRecords, err := store.GetLiveRecords(sessionID)
	require.NoError(t, err)
	assert.Len(t, liveRecords, 2)

	// Check all records
	allRecords, err := store.GetAllRecords(sessionID)
	require.NoError(t, err)
	assert.Len(t, allRecords, 3)
	assert.False(t, allRecords[0].Live)
	assert.True(t, allRecords[1].Live)
	assert.True(t, allRecords[2].Live)

	// Mark it live again
	err = store.MarkRecordLive(sessionID, 1)
	require.NoError(t, err)

	liveRecords, err = store.GetLiveRecords(sessionID)
	require.NoError(t, err)
	assert.Len(t, liveRecords, 3)
}

func TestSQLiteStoreDelete(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	sessionID := "test-session"

	// Add records
	for i := 0; i < 3; i++ {
		record := persistence.Record{
			Role: chat.UserRole,
			Contents: []chat.Content{
				{Text: "Message"},
			},
			Live:         true,
			Status:       persistence.RecordStatusSuccess,
			InputTokens:  6,
			OutputTokens: 4,
			Timestamp:    time.Now(),
		}
		_, err := store.AddRecord(sessionID, record)
		require.NoError(t, err)
	}

	// Delete middle record
	err = store.DeleteRecord(sessionID, 2)
	require.NoError(t, err)

	// Check remaining records
	records, err := store.GetAllRecords(sessionID)
	require.NoError(t, err)
	assert.Len(t, records, 2)
	assert.Equal(t, int64(1), records[0].ID)
	assert.Equal(t, int64(3), records[1].ID)
}

func TestSQLiteStoreClear(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	sessionID := "test-session"

	// Add records
	for i := 0; i < 5; i++ {
		record := persistence.Record{
			Role: chat.UserRole,
			Contents: []chat.Content{
				{Text: "Message"},
			},
			Live:         true,
			Status:       persistence.RecordStatusSuccess,
			InputTokens:  6,
			OutputTokens: 4,
			Timestamp:    time.Now(),
		}
		_, err := store.AddRecord(sessionID, record)
		require.NoError(t, err)
	}

	// Clear all records
	err = store.Clear(sessionID)
	require.NoError(t, err)

	// Check no records remain
	records, err := store.GetAllRecords(sessionID)
	require.NoError(t, err)
	assert.Len(t, records, 0)
}

func TestSQLiteStoreMetrics(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	sessionID := "test-session"

	// Save metrics
	metrics := persistence.SessionMetrics{
		CompactionCount:     5,
		LastCompaction:      time.Now(),
		CumulativeTokens:    1000,
		CompactionThreshold: 0.75,
	}

	err = store.SaveMetrics(sessionID, metrics)
	require.NoError(t, err)

	// Load metrics
	loaded, err := store.LoadMetrics(sessionID)
	require.NoError(t, err)

	assert.Equal(t, metrics.CompactionCount, loaded.CompactionCount)
	assert.Equal(t, metrics.CumulativeTokens, loaded.CumulativeTokens)
	assert.Equal(t, metrics.CompactionThreshold, loaded.CompactionThreshold)
	assert.WithinDuration(t, metrics.LastCompaction, loaded.LastCompaction, time.Second)
}

func TestSQLiteStorePersistence(t *testing.T) {
	// Use a temporary file for this test
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	sessionID := "test-session"

	// Create store and add records
	store1, err := New(dbPath)
	require.NoError(t, err)

	record := persistence.Record{
		Role: chat.AssistantRole,
		Contents: []chat.Content{
			{Text: "Persisted message"},
		},
		Live:         true,
		Status:       persistence.RecordStatusSuccess,
		InputTokens:  9,
		OutputTokens: 6,
		Timestamp:    time.Now(),
	}

	id, err := store1.AddRecord(sessionID, record)
	require.NoError(t, err)

	// Save metrics
	metrics := persistence.SessionMetrics{
		CompactionCount:  3,
		CumulativeTokens: 500,
	}
	err = store1.SaveMetrics(sessionID, metrics)
	require.NoError(t, err)

	// Close first store
	store1.Close()

	// Open new store with same file
	store2, err := New(dbPath)
	require.NoError(t, err)
	defer store2.Close()

	// Check records persisted
	records, err := store2.GetAllRecords(sessionID)
	require.NoError(t, err)
	assert.Len(t, records, 1)
	assert.Equal(t, "Persisted message", records[0].GetText())
	assert.Equal(t, id, records[0].ID)

	// Check metrics persisted
	loadedMetrics, err := store2.LoadMetrics(sessionID)
	require.NoError(t, err)
	assert.Equal(t, 3, loadedMetrics.CompactionCount)
	assert.Equal(t, 500, loadedMetrics.CumulativeTokens)
}

func TestSQLiteStoreOrdering(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	sessionID := "test-session"

	// Add records with specific timestamps
	baseTime := time.Now()
	times := []time.Duration{
		3 * time.Second,
		1 * time.Second,
		2 * time.Second,
	}

	for i, duration := range times {
		record := persistence.Record{
			Role: chat.UserRole,
			Contents: []chat.Content{
				{Text: string(rune('A' + i))}, // A, B, C
			},
			Live:         true,
			Status:       persistence.RecordStatusSuccess,
			InputTokens:  6,
			OutputTokens: 4,
			Timestamp:    baseTime.Add(duration),
		}
		_, err := store.AddRecord(sessionID, record)
		require.NoError(t, err)
	}

	// Get records - should be ordered by timestamp
	records, err := store.GetAllRecords(sessionID)
	require.NoError(t, err)
	assert.Len(t, records, 3)
	assert.Equal(t, "B", records[0].GetText()) // 1 second
	assert.Equal(t, "C", records[1].GetText()) // 2 seconds
	assert.Equal(t, "A", records[2].GetText()) // 3 seconds
}

func TestSQLiteStoreFileCreation(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "new.db")

	// Ensure file doesn't exist
	_, err := os.Stat(dbPath)
	assert.True(t, os.IsNotExist(err))

	// Create store - should create file
	store, err := New(dbPath)
	require.NoError(t, err)
	defer store.Close()

	// File should now exist
	info, err := os.Stat(dbPath)
	require.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0))
}

func TestSQLiteStoreMultipleSessions(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	// Add records for multiple sessions
	session1 := "session-1"
	session2 := "session-2"

	// Add records to session 1
	for i := 0; i < 3; i++ {
		record := persistence.Record{
			Role: chat.UserRole,
			Contents: []chat.Content{
				{Text: "Session 1 message"},
			},
			Live:         true,
			Status:       persistence.RecordStatusSuccess,
			InputTokens:  5,
			OutputTokens: 3,
			Timestamp:    time.Now(),
		}
		_, err := store.AddRecord(session1, record)
		require.NoError(t, err)
	}

	// Add records to session 2
	for i := 0; i < 2; i++ {
		record := persistence.Record{
			Role: chat.AssistantRole,
			Contents: []chat.Content{
				{Text: "Session 2 message"},
			},
			Live:         true,
			Status:       persistence.RecordStatusSuccess,
			InputTokens:  4,
			OutputTokens: 2,
			Timestamp:    time.Now(),
		}
		_, err := store.AddRecord(session2, record)
		require.NoError(t, err)
	}

	// Check sessions are isolated
	records1, err := store.GetAllRecords(session1)
	require.NoError(t, err)
	assert.Len(t, records1, 3)
	assert.Equal(t, "Session 1 message", records1[0].GetText())

	records2, err := store.GetAllRecords(session2)
	require.NoError(t, err)
	assert.Len(t, records2, 2)
	assert.Equal(t, "Session 2 message", records2[0].GetText())

	// Test ListSessions
	sessions, err := store.ListSessions()
	require.NoError(t, err)
	assert.Len(t, sessions, 2)
	assert.Contains(t, sessions, session1)
	assert.Contains(t, sessions, session2)

	// Test DeleteSession
	err = store.DeleteSession(session1)
	require.NoError(t, err)

	// Check session 1 is deleted
	records1, err = store.GetAllRecords(session1)
	require.NoError(t, err)
	assert.Len(t, records1, 0)

	// Check session 2 is still there
	records2, err = store.GetAllRecords(session2)
	require.NoError(t, err)
	assert.Len(t, records2, 2)

	// Check ListSessions now only returns session 2
	sessions, err = store.ListSessions()
	require.NoError(t, err)
	assert.Len(t, sessions, 1)
	assert.Equal(t, session2, sessions[0])
}
