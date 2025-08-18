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

	// Test adding a record
	record := persistence.Record{
		Role:         chat.UserRole,
		Content:      "Test message",
		Live:         true,
		InputTokens:  7,
		OutputTokens: 3,
		Timestamp:    time.Now(),
	}

	id, err := store.AddRecord(record)
	require.NoError(t, err)
	assert.Greater(t, id, int64(0))

	// Test getting all records
	records, err := store.GetAllRecords()
	require.NoError(t, err)
	assert.Len(t, records, 1)
	assert.Equal(t, "Test message", records[0].Content)
	assert.Equal(t, chat.UserRole, records[0].Role)
	assert.True(t, records[0].Live)

	// Test getting live records
	liveRecords, err := store.GetLiveRecords()
	require.NoError(t, err)
	assert.Len(t, liveRecords, 1)
}

func TestSQLiteStoreUpdateRecord(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	// Add a record
	record := persistence.Record{
		Role:         chat.UserRole,
		Content:      "Original",
		Live:         true,
		InputTokens:  3,
		OutputTokens: 2,
		Timestamp:    time.Now(),
	}

	id, err := store.AddRecord(record)
	require.NoError(t, err)

	// Update the record
	record.Content = "Updated"
	record.InputTokens = 5
	record.OutputTokens = 2
	err = store.UpdateRecord(id, record)
	require.NoError(t, err)

	// Verify update
	records, err := store.GetAllRecords()
	require.NoError(t, err)
	assert.Len(t, records, 1)
	assert.Equal(t, "Updated", records[0].Content)
	assert.Equal(t, 5, records[0].InputTokens)
	assert.Equal(t, 2, records[0].OutputTokens)
}

func TestSQLiteStoreMarkLiveDead(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	// Add multiple records
	for i := 0; i < 3; i++ {
		record := persistence.Record{
			Role:         chat.UserRole,
			Content:      "Message",
			Live:         true,
			InputTokens:  6,
			OutputTokens: 4,
			Timestamp:    time.Now(),
		}
		_, err := store.AddRecord(record)
		require.NoError(t, err)
	}

	// Mark first record as dead
	err = store.MarkRecordDead(1)
	require.NoError(t, err)

	// Check live records
	liveRecords, err := store.GetLiveRecords()
	require.NoError(t, err)
	assert.Len(t, liveRecords, 2)

	// Check all records
	allRecords, err := store.GetAllRecords()
	require.NoError(t, err)
	assert.Len(t, allRecords, 3)
	assert.False(t, allRecords[0].Live)
	assert.True(t, allRecords[1].Live)
	assert.True(t, allRecords[2].Live)

	// Mark it live again
	err = store.MarkRecordLive(1)
	require.NoError(t, err)

	liveRecords, err = store.GetLiveRecords()
	require.NoError(t, err)
	assert.Len(t, liveRecords, 3)
}

func TestSQLiteStoreDelete(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	// Add records
	for i := 0; i < 3; i++ {
		record := persistence.Record{
			Role:         chat.UserRole,
			Content:      "Message",
			Live:         true,
			InputTokens:  6,
			OutputTokens: 4,
			Timestamp:    time.Now(),
		}
		_, err := store.AddRecord(record)
		require.NoError(t, err)
	}

	// Delete middle record
	err = store.DeleteRecord(2)
	require.NoError(t, err)

	// Check remaining records
	records, err := store.GetAllRecords()
	require.NoError(t, err)
	assert.Len(t, records, 2)
	assert.Equal(t, int64(1), records[0].ID)
	assert.Equal(t, int64(3), records[1].ID)
}

func TestSQLiteStoreClear(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	// Add records
	for i := 0; i < 5; i++ {
		record := persistence.Record{
			Role:         chat.UserRole,
			Content:      "Message",
			Live:         true,
			InputTokens:  6,
			OutputTokens: 4,
			Timestamp:    time.Now(),
		}
		_, err := store.AddRecord(record)
		require.NoError(t, err)
	}

	// Clear all records
	err = store.Clear()
	require.NoError(t, err)

	// Check no records remain
	records, err := store.GetAllRecords()
	require.NoError(t, err)
	assert.Len(t, records, 0)
}

func TestSQLiteStoreMetrics(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	// Save metrics
	metrics := persistence.SessionMetrics{
		CompactionCount:     5,
		LastCompaction:      time.Now(),
		CumulativeTokens:    1000,
		CompactionThreshold: 0.75,
	}

	err = store.SaveMetrics(metrics)
	require.NoError(t, err)

	// Load metrics
	loaded, err := store.LoadMetrics()
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

	// Create store and add records
	store1, err := New(dbPath)
	require.NoError(t, err)

	record := persistence.Record{
		Role:         chat.AssistantRole,
		Content:      "Persisted message",
		Live:         true,
		InputTokens:  9,
		OutputTokens: 6,
		Timestamp:    time.Now(),
	}

	id, err := store1.AddRecord(record)
	require.NoError(t, err)

	// Save metrics
	metrics := persistence.SessionMetrics{
		CompactionCount:  3,
		CumulativeTokens: 500,
	}
	err = store1.SaveMetrics(metrics)
	require.NoError(t, err)

	// Close first store
	store1.Close()

	// Open new store with same file
	store2, err := New(dbPath)
	require.NoError(t, err)
	defer store2.Close()

	// Check records persisted
	records, err := store2.GetAllRecords()
	require.NoError(t, err)
	assert.Len(t, records, 1)
	assert.Equal(t, "Persisted message", records[0].Content)
	assert.Equal(t, id, records[0].ID)

	// Check metrics persisted
	loadedMetrics, err := store2.LoadMetrics()
	require.NoError(t, err)
	assert.Equal(t, 3, loadedMetrics.CompactionCount)
	assert.Equal(t, 500, loadedMetrics.CumulativeTokens)
}

func TestSQLiteStoreOrdering(t *testing.T) {
	store, err := New(":memory:")
	require.NoError(t, err)
	defer store.Close()

	// Add records with specific timestamps
	baseTime := time.Now()
	times := []time.Duration{
		3 * time.Second,
		1 * time.Second,
		2 * time.Second,
	}

	for i, duration := range times {
		record := persistence.Record{
			Role:         chat.UserRole,
			Content:      string(rune('A' + i)), // A, B, C
			Live:         true,
			InputTokens:  6,
			OutputTokens: 4,
			Timestamp:    baseTime.Add(duration),
		}
		_, err := store.AddRecord(record)
		require.NoError(t, err)
	}

	// Get records - should be ordered by timestamp
	records, err := store.GetAllRecords()
	require.NoError(t, err)
	assert.Len(t, records, 3)
	assert.Equal(t, "B", records[0].Content) // 1 second
	assert.Equal(t, "C", records[1].Content) // 2 seconds
	assert.Equal(t, "A", records[2].Content) // 3 seconds
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
