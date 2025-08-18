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
	AddRecord(sessionID string, record Record) (int64, error)

	// GetAllRecords retrieves all records in chronological order.
	GetAllRecords(sessionID string) ([]Record, error)

	// GetLiveRecords retrieves only live records in chronological order.
	GetLiveRecords(sessionID string) ([]Record, error)

	// UpdateRecord updates an existing record by ID.
	UpdateRecord(sessionID string, id int64, record Record) error

	// MarkRecordDead marks a record as not live.
	MarkRecordDead(sessionID string, id int64) error

	// MarkRecordLive marks a record as live.
	MarkRecordLive(sessionID string, id int64) error

	// DeleteRecord removes a record by ID.
	DeleteRecord(sessionID string, id int64) error

	// Clear removes all records for a session.
	Clear(sessionID string) error

	// Close closes the store and releases resources.
	Close() error

	// SaveMetrics persists session metrics.
	SaveMetrics(sessionID string, metrics SessionMetrics) error

	// LoadMetrics retrieves saved session metrics.
	LoadMetrics(sessionID string) (SessionMetrics, error)

	// ListSessions returns all session IDs in the store.
	ListSessions() ([]string, error)

	// DeleteSession removes all data for a session.
	DeleteSession(sessionID string) error
}

// SessionMetrics represents session statistics that can be persisted.
type SessionMetrics struct {
	CompactionCount     int       `json:"compaction_count"`
	LastCompaction      time.Time `json:"last_compaction"`
	CumulativeTokens    int       `json:"cumulative_tokens"`
	CompactionThreshold float64   `json:"compaction_threshold"`
}

// sessionData holds data for a single session
type sessionData struct {
	records []Record
	nextID  int64
	metrics SessionMetrics
}

// MemoryStore provides an in-memory implementation of Store.
type MemoryStore struct {
	mu       sync.Mutex
	sessions map[string]*sessionData
}

// NewMemoryStore creates a new in-memory store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		sessions: make(map[string]*sessionData),
	}
}

// AddRecord adds a new record to the in-memory store and returns its assigned ID.
func (m *MemoryStore) AddRecord(sessionID string, record Record) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	sess := m.getOrCreateSessionLocked(sessionID)
	record.ID = sess.nextID
	sess.nextID++
	sess.records = append(sess.records, record)
	return record.ID, nil
}

// getOrCreateSessionLocked gets or creates a session (mutex must be held)
func (m *MemoryStore) getOrCreateSessionLocked(sessionID string) *sessionData {
	if sess, ok := m.sessions[sessionID]; ok {
		return sess
	}
	sess := &sessionData{
		records: make([]Record, 0),
		nextID:  1,
	}
	m.sessions[sessionID] = sess
	return sess
}

// GetAllRecords returns a copy of all records in the store, both live and dead.
func (m *MemoryStore) GetAllRecords(sessionID string) ([]Record, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	sess := m.getOrCreateSessionLocked(sessionID)
	result := make([]Record, len(sess.records))
	copy(result, sess.records)
	return result, nil
}

// GetLiveRecords returns only the records marked as live in the current context window.
func (m *MemoryStore) GetLiveRecords(sessionID string) ([]Record, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	sess := m.getOrCreateSessionLocked(sessionID)
	var live []Record
	for _, r := range sess.records {
		if r.Live {
			live = append(live, r)
		}
	}
	return live, nil
}

// UpdateRecord updates an existing record with the given ID in the store.
func (m *MemoryStore) UpdateRecord(sessionID string, id int64, record Record) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	sess := m.getOrCreateSessionLocked(sessionID)
	for i, r := range sess.records {
		if r.ID == id {
			record.ID = id // Preserve ID
			sess.records[i] = record
			return nil
		}
	}
	return nil // Not found is not an error for memory store
}

// MarkRecordDead marks a record as dead, removing it from the active context window.
func (m *MemoryStore) MarkRecordDead(sessionID string, id int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	sess := m.getOrCreateSessionLocked(sessionID)
	for i, r := range sess.records {
		if r.ID == id {
			sess.records[i].Live = false
			return nil
		}
	}
	return nil
}

// MarkRecordLive marks a record as live, adding it back to the active context window.
func (m *MemoryStore) MarkRecordLive(sessionID string, id int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	sess := m.getOrCreateSessionLocked(sessionID)
	for i, r := range sess.records {
		if r.ID == id {
			sess.records[i].Live = true
			return nil
		}
	}
	return nil
}

// DeleteRecord permanently removes a record with the given ID from the store.
func (m *MemoryStore) DeleteRecord(sessionID string, id int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	sess := m.getOrCreateSessionLocked(sessionID)
	for i, r := range sess.records {
		if r.ID == id {
			sess.records = append(sess.records[:i], sess.records[i+1:]...)
			return nil
		}
	}
	return nil
}

// Clear removes all records for a session.
func (m *MemoryStore) Clear(sessionID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if sess, ok := m.sessions[sessionID]; ok {
		sess.records = sess.records[:0]
		sess.nextID = 1
		sess.metrics = SessionMetrics{}
	}
	return nil
}

// Close is a no-op for the in-memory store as there are no resources to release.
func (m *MemoryStore) Close() error {
	// Nothing to close for memory store
	return nil
}

// SaveMetrics stores the session metrics in memory for later retrieval.
func (m *MemoryStore) SaveMetrics(sessionID string, metrics SessionMetrics) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	sess := m.getOrCreateSessionLocked(sessionID)
	sess.metrics = metrics
	return nil
}

// LoadMetrics retrieves the previously saved session metrics from memory.
func (m *MemoryStore) LoadMetrics(sessionID string) (SessionMetrics, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	sess := m.getOrCreateSessionLocked(sessionID)
	return sess.metrics, nil
}

// ListSessions returns all session IDs in the store.
func (m *MemoryStore) ListSessions() ([]string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	var sessions []string
	for id := range m.sessions {
		sessions = append(sessions, id)
	}
	return sessions, nil
}

// DeleteSession removes all data for a session.
func (m *MemoryStore) DeleteSession(sessionID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	delete(m.sessions, sessionID)
	return nil
}
