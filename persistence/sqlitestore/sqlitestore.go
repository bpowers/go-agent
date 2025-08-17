// Package sqlitestore provides SQLite-based persistence for Session records.
package sqlitestore

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	_ "modernc.org/sqlite"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/persistence"
)

// SQLiteStore implements persistence.Store using SQLite.
type SQLiteStore struct {
	db *sql.DB
}

// New creates a new SQLite-based store at the given path.
// Use ":memory:" for an in-memory database.
func New(dbPath string) (*SQLiteStore, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}

	store := &SQLiteStore{db: db}
	if err := store.initSchema(); err != nil {
		db.Close()
		return nil, fmt.Errorf("init schema: %w", err)
	}

	return store, nil
}

// initSchema creates the necessary tables if they don't exist.
func (s *SQLiteStore) initSchema() error {
	const schema = `
CREATE TABLE IF NOT EXISTS records (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    live       BOOLEAN NOT NULL,
    tokens     INTEGER NOT NULL,
    timestamp  DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_live ON records(live);
CREATE INDEX IF NOT EXISTS idx_records_timestamp ON records(timestamp);

CREATE TABLE IF NOT EXISTS metrics (
    id                    INTEGER PRIMARY KEY CHECK (id = 1),
    compaction_count      INTEGER NOT NULL DEFAULT 0,
    last_compaction       DATETIME,
    cumulative_tokens     INTEGER NOT NULL DEFAULT 0,
    compaction_threshold  REAL NOT NULL DEFAULT 0.8,
    data                  TEXT
);

-- Ensure metrics table has exactly one row
INSERT OR IGNORE INTO metrics (id) VALUES (1);
`
	_, err := s.db.Exec(schema)
	return err
}

// AddRecord implements persistence.Store.
func (s *SQLiteStore) AddRecord(record persistence.Record) (int64, error) {
	result, err := s.db.Exec(
		`INSERT INTO records (role, content, live, tokens, timestamp) VALUES (?, ?, ?, ?, ?)`,
		string(record.Role), record.Content, record.Live, record.Tokens, record.Timestamp,
	)
	if err != nil {
		return 0, fmt.Errorf("insert record: %w", err)
	}

	id, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("get insert id: %w", err)
	}

	return id, nil
}

// GetAllRecords implements persistence.Store.
func (s *SQLiteStore) GetAllRecords() ([]persistence.Record, error) {
	rows, err := s.db.Query(
		`SELECT id, role, content, live, tokens, timestamp FROM records ORDER BY timestamp, id`,
	)
	if err != nil {
		return nil, fmt.Errorf("query records: %w", err)
	}
	defer rows.Close()

	var records []persistence.Record
	for rows.Next() {
		var r persistence.Record
		var roleStr string
		if err := rows.Scan(&r.ID, &roleStr, &r.Content, &r.Live, &r.Tokens, &r.Timestamp); err != nil {
			return nil, fmt.Errorf("scan record: %w", err)
		}
		r.Role = chat.Role(roleStr)
		records = append(records, r)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate records: %w", err)
	}

	return records, nil
}

// GetLiveRecords implements persistence.Store.
func (s *SQLiteStore) GetLiveRecords() ([]persistence.Record, error) {
	rows, err := s.db.Query(
		`SELECT id, role, content, live, tokens, timestamp FROM records WHERE live = 1 ORDER BY timestamp, id`,
	)
	if err != nil {
		return nil, fmt.Errorf("query live records: %w", err)
	}
	defer rows.Close()

	var records []persistence.Record
	for rows.Next() {
		var r persistence.Record
		var roleStr string
		if err := rows.Scan(&r.ID, &roleStr, &r.Content, &r.Live, &r.Tokens, &r.Timestamp); err != nil {
			return nil, fmt.Errorf("scan record: %w", err)
		}
		r.Role = chat.Role(roleStr)
		records = append(records, r)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate records: %w", err)
	}

	return records, nil
}

// UpdateRecord implements persistence.Store.
func (s *SQLiteStore) UpdateRecord(id int64, record persistence.Record) error {
	_, err := s.db.Exec(
		`UPDATE records SET role = ?, content = ?, live = ?, tokens = ?, timestamp = ? WHERE id = ?`,
		string(record.Role), record.Content, record.Live, record.Tokens, record.Timestamp, id,
	)
	if err != nil {
		return fmt.Errorf("update record: %w", err)
	}
	return nil
}

// MarkRecordDead implements persistence.Store.
func (s *SQLiteStore) MarkRecordDead(id int64) error {
	_, err := s.db.Exec(`UPDATE records SET live = 0 WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("mark record dead: %w", err)
	}
	return nil
}

// MarkRecordLive implements persistence.Store.
func (s *SQLiteStore) MarkRecordLive(id int64) error {
	_, err := s.db.Exec(`UPDATE records SET live = 1 WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("mark record live: %w", err)
	}
	return nil
}

// DeleteRecord implements persistence.Store.
func (s *SQLiteStore) DeleteRecord(id int64) error {
	_, err := s.db.Exec(`DELETE FROM records WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("delete record: %w", err)
	}
	return nil
}

// Clear implements persistence.Store.
func (s *SQLiteStore) Clear() error {
	_, err := s.db.Exec(`DELETE FROM records`)
	if err != nil {
		return fmt.Errorf("clear records: %w", err)
	}

	// Reset metrics
	_, err = s.db.Exec(
		`UPDATE metrics SET compaction_count = 0, last_compaction = NULL, cumulative_tokens = 0, compaction_threshold = 0.8 WHERE id = 1`,
	)
	if err != nil {
		return fmt.Errorf("reset metrics: %w", err)
	}

	return nil
}

// Close implements persistence.Store.
func (s *SQLiteStore) Close() error {
	return s.db.Close()
}

// SaveMetrics implements persistence.Store.
func (s *SQLiteStore) SaveMetrics(metrics persistence.SessionMetrics) error {
	// Store as JSON for extensibility
	data, err := json.Marshal(metrics)
	if err != nil {
		return fmt.Errorf("marshal metrics: %w", err)
	}

	var lastCompaction *time.Time
	if !metrics.LastCompaction.IsZero() {
		lastCompaction = &metrics.LastCompaction
	}

	_, err = s.db.Exec(
		`UPDATE metrics SET 
			compaction_count = ?, 
			last_compaction = ?, 
			cumulative_tokens = ?, 
			compaction_threshold = ?,
			data = ?
		WHERE id = 1`,
		metrics.CompactionCount,
		lastCompaction,
		metrics.CumulativeTokens,
		metrics.CompactionThreshold,
		string(data),
	)
	if err != nil {
		return fmt.Errorf("save metrics: %w", err)
	}

	return nil
}

// LoadMetrics implements persistence.Store.
func (s *SQLiteStore) LoadMetrics() (persistence.SessionMetrics, error) {
	var metrics persistence.SessionMetrics
	var lastCompaction sql.NullTime
	var data sql.NullString

	err := s.db.QueryRow(
		`SELECT compaction_count, last_compaction, cumulative_tokens, compaction_threshold, data FROM metrics WHERE id = 1`,
	).Scan(&metrics.CompactionCount, &lastCompaction, &metrics.CumulativeTokens, &metrics.CompactionThreshold, &data)
	if err != nil {
		if err == sql.ErrNoRows {
			// Return default metrics
			return persistence.SessionMetrics{CompactionThreshold: 0.8}, nil
		}
		return metrics, fmt.Errorf("load metrics: %w", err)
	}

	if lastCompaction.Valid {
		metrics.LastCompaction = lastCompaction.Time
	}

	// If we have JSON data, use it (for future extensibility)
	if data.Valid && data.String != "" {
		json.Unmarshal([]byte(data.String), &metrics)
	}

	return metrics, nil
}

// BeginTransaction starts a new transaction.
func (s *SQLiteStore) BeginTransaction() (*sql.Tx, error) {
	return s.db.Begin()
}

// ExecInTransaction executes a function within a transaction.
func (s *SQLiteStore) ExecInTransaction(fn func(*sql.Tx) error) error {
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}

	if err := fn(tx); err != nil {
		tx.Rollback()
		return err
	}

	return tx.Commit()
}
