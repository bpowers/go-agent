// Package persistence provides storage interfaces for Session records.
package persistence

import (
	"sync"
	"time"

	"github.com/bpowers/go-agent/chat"
)

// Record represents a conversation turn that can be persisted.
type Record struct {
	ID           int64     `json:"id"`
	Role         chat.Role `json:"role"`
	Content      string    `json:"content"`
	Live         bool      `json:"live"`
	InputTokens  int       `json:"input_tokens"`  // Actual tokens from LLM
	OutputTokens int       `json:"output_tokens"` // Actual tokens from LLM
	Timestamp    time.Time `json:"timestamp"`
}

// Store defines the interface for persisting session records.
type Store interface {
	// AddRecord inserts a new record into the store.
	AddRecord(record Record) (int64, error)

	// GetAllRecords retrieves all records in chronological order.
	GetAllRecords() ([]Record, error)

	// GetLiveRecords retrieves only live records in chronological order.
	GetLiveRecords() ([]Record, error)

	// UpdateRecord updates an existing record by ID.
	UpdateRecord(id int64, record Record) error

	// MarkRecordDead marks a record as not live.
	MarkRecordDead(id int64) error

	// MarkRecordLive marks a record as live.
	MarkRecordLive(id int64) error

	// DeleteRecord removes a record by ID.
	DeleteRecord(id int64) error

	// Clear removes all records.
	Clear() error

	// Close closes the store and releases resources.
	Close() error

	// SaveMetrics persists session metrics.
	SaveMetrics(metrics SessionMetrics) error

	// LoadMetrics retrieves saved session metrics.
	LoadMetrics() (SessionMetrics, error)
}

// SessionMetrics represents session statistics that can be persisted.
type SessionMetrics struct {
	CompactionCount     int       `json:"compaction_count"`
	LastCompaction      time.Time `json:"last_compaction"`
	CumulativeTokens    int       `json:"cumulative_tokens"`
	CompactionThreshold float64   `json:"compaction_threshold"`
}

// MemoryStore provides an in-memory implementation of Store.
type MemoryStore struct {
	mu      sync.Mutex
	records []Record
	nextID  int64
	metrics SessionMetrics
}

// NewMemoryStore creates a new in-memory store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		records: make([]Record, 0),
		nextID:  1,
	}
}

// AddRecord implements Store.
func (m *MemoryStore) AddRecord(record Record) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	record.ID = m.nextID
	m.nextID++
	m.records = append(m.records, record)
	return record.ID, nil
}

// GetAllRecords implements Store.
func (m *MemoryStore) GetAllRecords() ([]Record, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := make([]Record, len(m.records))
	copy(result, m.records)
	return result, nil
}

// GetLiveRecords implements Store.
func (m *MemoryStore) GetLiveRecords() ([]Record, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var live []Record
	for _, r := range m.records {
		if r.Live {
			live = append(live, r)
		}
	}
	return live, nil
}

// UpdateRecord implements Store.
func (m *MemoryStore) UpdateRecord(id int64, record Record) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i, r := range m.records {
		if r.ID == id {
			record.ID = id // Preserve ID
			m.records[i] = record
			return nil
		}
	}
	return nil // Not found is not an error for memory store
}

// MarkRecordDead implements Store.
func (m *MemoryStore) MarkRecordDead(id int64) error {
	for i, r := range m.records {
		if r.ID == id {
			m.records[i].Live = false
			return nil
		}
	}
	return nil
}

// MarkRecordLive implements Store.
func (m *MemoryStore) MarkRecordLive(id int64) error {
	for i, r := range m.records {
		if r.ID == id {
			m.records[i].Live = true
			return nil
		}
	}
	return nil
}

// DeleteRecord implements Store.
func (m *MemoryStore) DeleteRecord(id int64) error {
	for i, r := range m.records {
		if r.ID == id {
			m.records = append(m.records[:i], m.records[i+1:]...)
			return nil
		}
	}
	return nil
}

// Clear implements Store.
func (m *MemoryStore) Clear() error {
	m.records = m.records[:0]
	m.nextID = 1
	m.metrics = SessionMetrics{}
	return nil
}

// Close implements Store.
func (m *MemoryStore) Close() error {
	// Nothing to close for memory store
	return nil
}

// SaveMetrics implements Store.
func (m *MemoryStore) SaveMetrics(metrics SessionMetrics) error {
	m.metrics = metrics
	return nil
}

// LoadMetrics implements Store.
func (m *MemoryStore) LoadMetrics() (SessionMetrics, error) {
	return m.metrics, nil
}
