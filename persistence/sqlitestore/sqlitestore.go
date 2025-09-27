// Package sqlitestore provides SQLite-based persistence for Session records.
package sqlitestore

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
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
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    tool_calls    TEXT,
    tool_results  TEXT,
    live          BOOLEAN NOT NULL,
    status        TEXT NOT NULL DEFAULT 'success',
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    timestamp     DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_session ON records(session_id);
CREATE INDEX IF NOT EXISTS idx_records_live ON records(session_id, live);
CREATE INDEX IF NOT EXISTS idx_records_timestamp ON records(session_id, timestamp);

CREATE TABLE IF NOT EXISTS metrics (
    session_id            TEXT PRIMARY KEY,
    compaction_count      INTEGER NOT NULL DEFAULT 0,
    last_compaction       DATETIME,
    cumulative_tokens     INTEGER NOT NULL DEFAULT 0,
    compaction_threshold  REAL NOT NULL DEFAULT 0.8,
    data                  TEXT
);
`
	_, err := s.db.Exec(schema)
	if err != nil {
		return err
	}

	// Make sure tool_calls column exists even on pre-existing databases.
	if err := addColumnIfMissing(s.db, "records", "tool_calls", "TEXT"); err != nil {
		return err
	}
	if err := addColumnIfMissing(s.db, "records", "tool_results", "TEXT"); err != nil {
		return err
	}

	return nil
}

func addColumnIfMissing(db *sql.DB, table, column, colType string) error {
	_, err := db.Exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", table, column, colType))
	if err == nil {
		return nil
	}
	if strings.Contains(err.Error(), "duplicate column name") {
		return nil
	}
	return err
}

func encodeToolCalls(calls []chat.ToolCall) (sql.NullString, error) {
	if len(calls) == 0 {
		return sql.NullString{}, nil
	}
	data, err := json.Marshal(calls)
	if err != nil {
		return sql.NullString{}, err
	}
	return sql.NullString{String: string(data), Valid: true}, nil
}

func encodeToolResults(results []chat.ToolResult) (sql.NullString, error) {
	if len(results) == 0 {
		return sql.NullString{}, nil
	}
	data, err := json.Marshal(results)
	if err != nil {
		return sql.NullString{}, err
	}
	return sql.NullString{String: string(data), Valid: true}, nil
}

func decodeToolCalls(src sql.NullString, dest *[]chat.ToolCall) error {
	if !src.Valid || src.String == "" {
		*dest = nil
		return nil
	}
	return json.Unmarshal([]byte(src.String), dest)
}

func decodeToolResults(src sql.NullString, dest *[]chat.ToolResult) error {
	if !src.Valid || src.String == "" {
		*dest = nil
		return nil
	}
	return json.Unmarshal([]byte(src.String), dest)
}

// AddRecord implements persistence.Store.
func (s *SQLiteStore) AddRecord(sessionID string, record persistence.Record) (int64, error) {
	// Default to success if status not specified
	if record.Status == "" {
		record.Status = "success"
	}

	toolCallsJSON, err := encodeToolCalls(record.ToolCalls)
	if err != nil {
		return 0, fmt.Errorf("encode tool calls: %w", err)
	}
	toolResultsJSON, err := encodeToolResults(record.ToolResults)
	if err != nil {
		return 0, fmt.Errorf("encode tool results: %w", err)
	}
	result, err := s.db.Exec(
		`INSERT INTO records (session_id, role, content, tool_calls, tool_results, live, status, input_tokens, output_tokens, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		sessionID, string(record.Role), record.Content, toolCallsJSON, toolResultsJSON, record.Live, record.Status, record.InputTokens, record.OutputTokens, record.Timestamp,
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

// GetRecord implements persistence.Store.
func (s *SQLiteStore) GetRecord(sessionID string, id int64) (persistence.Record, error) {
	var r persistence.Record
	var roleStr string
	var toolCallsJSON sql.NullString
	var toolResultsJSON sql.NullString
	err := s.db.QueryRow(
		`SELECT id, role, content, tool_calls, tool_results, live, status, input_tokens, output_tokens, timestamp FROM records WHERE session_id = ? AND id = ?`,
		sessionID, id,
	).Scan(&r.ID, &roleStr, &r.Content, &toolCallsJSON, &toolResultsJSON, &r.Live, &r.Status, &r.InputTokens, &r.OutputTokens, &r.Timestamp)
	if err != nil {
		if err == sql.ErrNoRows {
			return persistence.Record{}, fmt.Errorf("record not found: %d", id)
		}
		return persistence.Record{}, fmt.Errorf("query record: %w", err)
	}
	r.Role = chat.Role(roleStr)
	if err := decodeToolCalls(toolCallsJSON, &r.ToolCalls); err != nil {
		return persistence.Record{}, fmt.Errorf("decode tool calls: %w", err)
	}
	if err := decodeToolResults(toolResultsJSON, &r.ToolResults); err != nil {
		return persistence.Record{}, fmt.Errorf("decode tool results: %w", err)
	}
	return r, nil
}

// GetAllRecords implements persistence.Store.
func (s *SQLiteStore) GetAllRecords(sessionID string) ([]persistence.Record, error) {
	rows, err := s.db.Query(
		`SELECT id, role, content, tool_calls, tool_results, live, status, input_tokens, output_tokens, timestamp FROM records WHERE session_id = ? ORDER BY timestamp, id`,
		sessionID,
	)
	if err != nil {
		return nil, fmt.Errorf("query records: %w", err)
	}
	defer rows.Close()

	var records []persistence.Record
	for rows.Next() {
		var r persistence.Record
		var roleStr string
		var toolCallsJSON sql.NullString
		var toolResultsJSON sql.NullString
		if err := rows.Scan(&r.ID, &roleStr, &r.Content, &toolCallsJSON, &toolResultsJSON, &r.Live, &r.Status, &r.InputTokens, &r.OutputTokens, &r.Timestamp); err != nil {
			return nil, fmt.Errorf("scan record: %w", err)
		}
		r.Role = chat.Role(roleStr)
		if err := decodeToolCalls(toolCallsJSON, &r.ToolCalls); err != nil {
			return nil, fmt.Errorf("decode tool calls: %w", err)
		}
		if err := decodeToolResults(toolResultsJSON, &r.ToolResults); err != nil {
			return nil, fmt.Errorf("decode tool results: %w", err)
		}
		records = append(records, r)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate records: %w", err)
	}

	return records, nil
}

// GetLiveRecords implements persistence.Store.
func (s *SQLiteStore) GetLiveRecords(sessionID string) ([]persistence.Record, error) {
	rows, err := s.db.Query(
		`SELECT id, role, content, tool_calls, tool_results, live, status, input_tokens, output_tokens, timestamp FROM records WHERE session_id = ? AND live = 1 ORDER BY timestamp, id`,
		sessionID,
	)
	if err != nil {
		return nil, fmt.Errorf("query live records: %w", err)
	}
	defer rows.Close()

	var records []persistence.Record
	for rows.Next() {
		var r persistence.Record
		var roleStr string
		var toolCallsJSON sql.NullString
		var toolResultsJSON sql.NullString
		if err := rows.Scan(&r.ID, &roleStr, &r.Content, &toolCallsJSON, &toolResultsJSON, &r.Live, &r.Status, &r.InputTokens, &r.OutputTokens, &r.Timestamp); err != nil {
			return nil, fmt.Errorf("scan record: %w", err)
		}
		r.Role = chat.Role(roleStr)
		if err := decodeToolCalls(toolCallsJSON, &r.ToolCalls); err != nil {
			return nil, fmt.Errorf("decode tool calls: %w", err)
		}
		if err := decodeToolResults(toolResultsJSON, &r.ToolResults); err != nil {
			return nil, fmt.Errorf("decode tool results: %w", err)
		}
		records = append(records, r)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate records: %w", err)
	}

	return records, nil
}

// UpdateRecord implements persistence.Store.
func (s *SQLiteStore) UpdateRecord(sessionID string, id int64, record persistence.Record) error {
	toolCallsJSON, err := encodeToolCalls(record.ToolCalls)
	if err != nil {
		return fmt.Errorf("encode tool calls: %w", err)
	}
	toolResultsJSON, err := encodeToolResults(record.ToolResults)
	if err != nil {
		return fmt.Errorf("encode tool results: %w", err)
	}
	_, err = s.db.Exec(
		`UPDATE records SET role = ?, content = ?, tool_calls = ?, tool_results = ?, live = ?, status = ?, input_tokens = ?, output_tokens = ?, timestamp = ? WHERE session_id = ? AND id = ?`,
		string(record.Role), record.Content, toolCallsJSON, toolResultsJSON, record.Live, record.Status, record.InputTokens, record.OutputTokens, record.Timestamp, sessionID, id,
	)
	if err != nil {
		return fmt.Errorf("update record: %w", err)
	}
	return nil
}

// MarkRecordDead implements persistence.Store.
func (s *SQLiteStore) MarkRecordDead(sessionID string, id int64) error {
	_, err := s.db.Exec(`UPDATE records SET live = 0 WHERE session_id = ? AND id = ?`, sessionID, id)
	if err != nil {
		return fmt.Errorf("mark record dead: %w", err)
	}
	return nil
}

// MarkRecordLive implements persistence.Store.
func (s *SQLiteStore) MarkRecordLive(sessionID string, id int64) error {
	_, err := s.db.Exec(`UPDATE records SET live = 1 WHERE session_id = ? AND id = ?`, sessionID, id)
	if err != nil {
		return fmt.Errorf("mark record live: %w", err)
	}
	return nil
}

// DeleteRecord implements persistence.Store.
func (s *SQLiteStore) DeleteRecord(sessionID string, id int64) error {
	_, err := s.db.Exec(`DELETE FROM records WHERE session_id = ? AND id = ?`, sessionID, id)
	if err != nil {
		return fmt.Errorf("delete record: %w", err)
	}
	return nil
}

// Clear implements persistence.Store.
func (s *SQLiteStore) Clear(sessionID string) error {
	_, err := s.db.Exec(`DELETE FROM records WHERE session_id = ?`, sessionID)
	if err != nil {
		return fmt.Errorf("clear records: %w", err)
	}

	// Reset metrics for this session
	_, err = s.db.Exec(`DELETE FROM metrics WHERE session_id = ?`, sessionID)
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
func (s *SQLiteStore) SaveMetrics(sessionID string, metrics persistence.SessionMetrics) error {
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
		`INSERT INTO metrics (session_id, compaction_count, last_compaction, cumulative_tokens, compaction_threshold, data) 
		VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT(session_id) DO UPDATE SET
			compaction_count = excluded.compaction_count,
			last_compaction = excluded.last_compaction,
			cumulative_tokens = excluded.cumulative_tokens,
			compaction_threshold = excluded.compaction_threshold,
			data = excluded.data`,
		sessionID,
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
func (s *SQLiteStore) LoadMetrics(sessionID string) (persistence.SessionMetrics, error) {
	var metrics persistence.SessionMetrics
	var lastCompaction sql.NullTime
	var data sql.NullString

	err := s.db.QueryRow(
		`SELECT compaction_count, last_compaction, cumulative_tokens, compaction_threshold, data FROM metrics WHERE session_id = ?`,
		sessionID,
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

// ListSessions implements persistence.Store.
func (s *SQLiteStore) ListSessions() ([]string, error) {
	rows, err := s.db.Query(`SELECT DISTINCT session_id FROM records ORDER BY session_id`)
	if err != nil {
		return nil, fmt.Errorf("query sessions: %w", err)
	}
	defer rows.Close()

	var sessions []string
	for rows.Next() {
		var sessionID string
		if err := rows.Scan(&sessionID); err != nil {
			return nil, fmt.Errorf("scan session: %w", err)
		}
		sessions = append(sessions, sessionID)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate sessions: %w", err)
	}

	return sessions, nil
}

// DeleteSession implements persistence.Store.
func (s *SQLiteStore) DeleteSession(sessionID string) error {
	// Start a transaction to delete both records and metrics
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Delete records
	if _, err := tx.Exec(`DELETE FROM records WHERE session_id = ?`, sessionID); err != nil {
		return fmt.Errorf("delete records: %w", err)
	}

	// Delete metrics
	if _, err := tx.Exec(`DELETE FROM metrics WHERE session_id = ?`, sessionID); err != nil {
		return fmt.Errorf("delete metrics: %w", err)
	}

	return tx.Commit()
}
