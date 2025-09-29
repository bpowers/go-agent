package agent

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/persistence"
)

// generateSessionID creates a unique session identifier
func generateSessionID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		// Fallback to timestamp if random fails
		return fmt.Sprintf("session-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

// Session manages the conversation lifecycle with automatic context compaction.
// It embeds chat.Chat for full compatibility while adding persistence and
// automatic summarization capabilities. When the context window approaches
// capacity (default 80%), older messages are automatically compacted into
// summaries to maintain conversation continuity.
type Session interface {
	chat.Chat // a session is a chat that has been enhanced with context window management.

	// SessionID returns the unique identifier for this session.
	SessionID() string

	// LiveRecords returns all records marked as live (in active context window).
	LiveRecords() []Record

	// TotalRecords returns all records (both live and dead).
	TotalRecords() []Record

	// CompactNow manually triggers context compaction.
	CompactNow() error

	// SetCompactionThreshold sets the threshold for automatic compaction (0.0-1.0).
	// A value of 0.8 means compact when 80% of the context window is used.
	// A value of 0.0 means never compact automatically.
	SetCompactionThreshold(float64)

	// Metrics returns usage statistics for the session.
	Metrics() SessionMetrics
}

// RecordStatus represents the status of a record in the conversation.
type RecordStatus string

const (
	RecordStatusPending RecordStatus = "pending" // Request sent, awaiting response
	RecordStatusSuccess RecordStatus = "success" // Successfully processed
	RecordStatusFailed  RecordStatus = "failed"  // Failed to process
)

// Record represents a conversation turn in the session history.
type Record struct {
	ID           int64             `json:"id,omitzero"`
	Role         chat.Role         `json:"role"`
	Content      string            `json:"content"`
	ToolCalls    []chat.ToolCall   `json:"tool_calls,omitzero"`
	ToolResults  []chat.ToolResult `json:"tool_results,omitzero"`
	Live         bool              `json:"live"`          // In active context window
	Status       RecordStatus      `json:"status"`        // Processing status
	InputTokens  int               `json:"input_tokens"`  // Actual tokens from LLM
	OutputTokens int               `json:"output_tokens"` // Actual tokens from LLM
	Timestamp    time.Time         `json:"timestamp"`
}

// SessionMetrics provides usage statistics for the session.
type SessionMetrics struct {
	CumulativeTokens int       `json:"cumulative_tokens"` // Total tokens used across all messages
	LiveTokens       int       `json:"live_tokens"`       // Tokens in active context window
	MaxTokens        int       `json:"max_tokens"`        // Model's max context size
	CompactionCount  int       `json:"compaction_count"`  // Number of compactions performed
	LastCompaction   time.Time `json:"last_compaction"`   // When last compacted
	RecordsLive      int       `json:"records_live"`      // Number of live records
	RecordsTotal     int       `json:"records_total"`     // Total records (live + dead)
	PercentFull      float64   `json:"percent_full"`      // LiveTokens/MaxTokens ratio
}

// SessionOption configures a Session.
type SessionOption func(*sessionOptions)

type sessionOptions struct {
	sessionID       string
	store           persistence.Store
	initialMessages []chat.Message
	summarizer      Summarizer
}

// WithRestoreSession restores a session with the given ID.
// This allows resuming a previous conversation by loading its history
// and state from the configured persistence store.
// If not provided, a new UUID will be generated for a fresh session.
func WithRestoreSession(id string) SessionOption {
	return func(opts *sessionOptions) {
		opts.sessionID = id
	}
}

// WithStore sets a custom persistence store for the session.
// If not provided, an in-memory store is used.
func WithStore(store persistence.Store) SessionOption {
	return func(opts *sessionOptions) {
		opts.store = store
	}
}

// WithInitialMessages sets the initial messages for the session.
func WithInitialMessages(msgs ...chat.Message) SessionOption {
	return func(opts *sessionOptions) {
		opts.initialMessages = msgs
	}
}

// WithSummarizer sets a custom summarizer for context compaction.
// If not provided, a default LLM-based summarizer is used.
func WithSummarizer(summarizer Summarizer) SessionOption {
	return func(opts *sessionOptions) {
		opts.summarizer = summarizer
	}
}

// NewSession creates a new Session with the given client, system prompt, and options.
func NewSession(client chat.Client, systemPrompt string, opts ...SessionOption) Session {
	// Apply options
	var options sessionOptions
	for _, opt := range opts {
		if opt != nil {
			opt(&options)
		}
	}

	// Generate session ID if not provided
	if options.sessionID == "" {
		options.sessionID = generateSessionID()
	}

	// Default to memory store if not specified
	if options.store == nil {
		options.store = persistence.NewMemoryStore()
	}

	// Default to LLM summarizer if not specified
	if options.summarizer == nil {
		// Use the same client - users can provide WithSummarizer() with a client
		// configured for a cheaper model if desired
		options.summarizer = NewSummarizer(client)
	}

	// Load existing metrics if available
	metrics, _ := options.store.LoadMetrics(options.sessionID)

	// Check if we have existing records in the store
	existingRecords, _ := options.store.GetAllRecords(options.sessionID)
	hasExistingRecords := len(existingRecords) > 0

	// If we have existing records, use the system prompt from the store
	// Otherwise, use the provided system prompt
	actualSystemPrompt := systemPrompt
	if hasExistingRecords {
		// Find the system prompt from existing records
		for _, r := range existingRecords {
			if r.Role == "system" {
				actualSystemPrompt = r.Content
				break
			}
		}
	}

	// Create base chat
	baseChat := client.NewChat(actualSystemPrompt, options.initialMessages...)

	// Only add initial records if the store is empty
	if !hasExistingRecords {
		// Create initial records from system prompt and initial messages
		if systemPrompt != "" {
			options.store.AddRecord(options.sessionID, persistence.Record{
				Role:         "system",
				Content:      systemPrompt,
				Live:         true,
				InputTokens:  0, // System prompt tokens counted with first message
				OutputTokens: 0,
				Timestamp:    time.Now(),
			})
		}

		for _, msg := range options.initialMessages {
			options.store.AddRecord(options.sessionID, persistence.Record{
				Role:         chat.Role(msg.Role),
				Content:      msg.GetText(),
				ToolCalls:    msg.GetToolCalls(),
				ToolResults:  msg.GetToolResults(),
				Live:         true,
				InputTokens:  0, // Initial messages' tokens counted with first query
				OutputTokens: 0,
				Timestamp:    time.Now(),
			})
		}
	}

	// Use loaded threshold if valid, otherwise default to 0.8
	// A threshold of 0.0 is valid (means never compact)
	compactionThreshold := metrics.CompactionThreshold
	if !hasExistingRecords && compactionThreshold == 0 {
		// Only set default for new sessions without explicit threshold
		compactionThreshold = 0.8
	}

	return &session{
		sessionID:           options.sessionID,
		chat:                baseChat,
		client:              client,
		systemPrompt:        actualSystemPrompt,
		store:               options.store,
		summarizer:          options.summarizer,
		compactionThreshold: compactionThreshold,
		compactionCount:     metrics.CompactionCount,
		lastCompaction:      metrics.LastCompaction,
		cumulativeTokens:    metrics.CumulativeTokens,
		tools:               make(map[string]registeredTool),
	}
}

// session is the implementation of Session with pluggable storage.
type session struct {
	sessionID    string
	chat         chat.Chat
	client       chat.Client
	systemPrompt string
	store        persistence.Store
	summarizer   Summarizer

	mu                  sync.Mutex
	compactionThreshold float64
	compactionCount     int
	lastCompaction      time.Time
	cumulativeTokens    int
	lastUsage           chat.TokenUsageDetails

	// Tool tracking - use single mutex for simplicity as per CLAUDE.md
	tools             map[string]registeredTool
	lastUserMessageID int64
	lastUserMessage   chat.Message
	lastHistoryLen    int
}

type registeredTool struct {
	def chat.ToolDef
	fn  func(context.Context, string) string
}

func cloneToolCalls(calls []chat.ToolCall) []chat.ToolCall {
	if len(calls) == 0 {
		return nil
	}
	cloned := make([]chat.ToolCall, len(calls))
	for i, tc := range calls {
		cloned[i] = chat.ToolCall{
			ID:   tc.ID,
			Name: tc.Name,
		}
		if tc.Arguments != nil {
			cloned[i].Arguments = append([]byte(nil), tc.Arguments...)
		}
	}
	return cloned
}

func cloneToolResults(results []chat.ToolResult) []chat.ToolResult {
	if len(results) == 0 {
		return nil
	}
	cloned := make([]chat.ToolResult, len(results))
	for i, tr := range results {
		cloned[i] = chat.ToolResult{
			ToolCallID: tr.ToolCallID,
			Name:       tr.Name,
			Content:    tr.Content,
			Error:      tr.Error,
		}
	}
	return cloned
}

func recordFromPersistence(r persistence.Record) Record {
	return Record{
		ID:           r.ID,
		Role:         chat.Role(r.Role),
		Content:      r.Content,
		ToolCalls:    cloneToolCalls(r.ToolCalls),
		ToolResults:  cloneToolResults(r.ToolResults),
		Live:         r.Live,
		Status:       RecordStatus(r.Status),
		InputTokens:  r.InputTokens,
		OutputTokens: r.OutputTokens,
		Timestamp:    r.Timestamp,
	}
}

// SessionID implements Session
func (s *session) SessionID() string {
	return s.sessionID
}

// Message implements chat.Chat
func (s *session) Message(ctx context.Context, msg chat.Message, opts ...chat.Option) (chat.Message, error) {
	// Add user message and check compaction
	tempChat, err := s.prepareForMessage(ctx, msg)
	if err != nil {
		return chat.Message{}, err
	}

	// Send message
	response, err := tempChat.Message(ctx, msg, opts...)
	if err != nil {
		return response, err
	}

	// Track response
	s.trackResponse(tempChat, response)
	return response, nil
}

// prepareForMessage adds the user message, checks for compaction, and returns a prepared chat.
// This method expects the mutex is NOT held and will handle locking internally.
func (s *session) prepareForMessage(ctx context.Context, msg chat.Message) (chat.Chat, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Build the message history from live records BEFORE adding the new message
	// This prevents the message from being duplicated when we call tempChat.Message()
	systemPrompt, msgs := s.buildChatHistoryLocked()
	s.lastHistoryLen = len(msgs)

	// Add user message to records (tokens will be updated after response)
	userMsgID, _ := s.store.AddRecord(s.sessionID, persistence.Record{
		Role:         chat.Role(msg.Role),
		Content:      msg.GetText(),
		ToolCalls:    msg.GetToolCalls(),
		ToolResults:  msg.GetToolResults(),
		Live:         true,
		InputTokens:  0, // Will be updated after response
		OutputTokens: 0,
		Timestamp:    time.Now(),
	})

	// Store message ID for later token update
	s.lastUserMessageID = userMsgID
	s.lastUserMessage = msg

	// Check if we need to compact before sending
	if s.shouldCompactLocked() {
		// We need to compact, but CompactNow needs the lock too
		// So we use a locked variant
		if err := s.compactNowLocked(ctx); err != nil {
			return nil, fmt.Errorf("auto-compaction failed: %w", err)
		}
	}

	// Recreate chat with history (not including the just-added message)
	tempChat := s.client.NewChat(systemPrompt, msgs...)

	// Re-register tools
	for _, tool := range s.tools {
		if err := tempChat.RegisterTool(tool.def, tool.fn); err != nil {
			return nil, fmt.Errorf("failed to re-register tool %s: %w", tool.def.Name(), err)
		}
	}

	return tempChat, nil
}

// trackResponse records the response and updates metrics with actual token counts.
// This method expects the mutex is NOT held and will handle locking internally.
func (s *session) trackResponse(tempChat chat.Chat, response chat.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Get actual token usage from the LLM
	usage, err := tempChat.TokenUsage()
	if err != nil {
		log.Printf("Warning: Failed to get token usage from LLM: %v", err)
	}
	s.lastUsage = usage.LastMessage

	// Log if we're missing expected token values
	if usage.LastMessage.InputTokens == 0 {
		log.Printf("Warning: LLM returned 0 input tokens for message")
	}
	if usage.LastMessage.OutputTokens == 0 {
		log.Printf("Warning: LLM returned 0 output tokens for response")
	}
	if usage.LastMessage.TotalTokens == 0 {
		log.Printf("Warning: LLM returned 0 total tokens for exchange")
	}

	s.cumulativeTokens += usage.LastMessage.TotalTokens

	// Update the user message with actual input tokens
	if s.lastUserMessageID > 0 && usage.LastMessage.InputTokens > 0 {
		// Get the specific record to update
		record, err := s.store.GetRecord(s.sessionID, s.lastUserMessageID)
		if err == nil {
			record.InputTokens = usage.LastMessage.InputTokens
			record.Status = string(RecordStatusSuccess) // Mark as successful
			s.store.UpdateRecord(s.sessionID, s.lastUserMessageID, record)
		}
	}

	// Persist assistant/tool responses in the order returned by the provider.
	// Skip the user message since we already added it in prepareForMessage.
	_, history := tempChat.History()
	if s.lastHistoryLen > len(history) {
		s.lastHistoryLen = len(history)
	}
	newMessages := history[s.lastHistoryLen:]

	// Skip the first message if it's the user message we already added
	if len(newMessages) > 0 && newMessages[0].Role == s.lastUserMessage.Role &&
		newMessages[0].GetText() == s.lastUserMessage.GetText() {
		newMessages = newMessages[1:]
	}

	now := time.Now()
	for i, m := range newMessages {
		rec := persistence.Record{
			Role:        m.Role,
			Content:     m.GetText(),
			ToolCalls:   cloneToolCalls(m.GetToolCalls()),
			ToolResults: cloneToolResults(m.GetToolResults()),
			Live:        true,
			Timestamp:   now.Add(time.Millisecond * time.Duration(i)),
		}

		// Assign output tokens to the final assistant message in the exchange.
		if m.Role == chat.AssistantRole && i == len(newMessages)-1 {
			rec.OutputTokens = usage.LastMessage.OutputTokens
		}

		if _, err := s.store.AddRecord(s.sessionID, rec); err != nil {
			log.Printf("Warning: failed to add record for role %s: %v", rec.Role, err)
		}
	}
	s.lastHistoryLen = len(history)

	// Save metrics
	s.saveMetricsLocked()
}

// History implements chat.Chat
func (s *session) History() (systemPrompt string, msgs []chat.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.buildChatHistoryLocked()
}

// TokenUsage implements chat.Chat
func (s *session) TokenUsage() (chat.TokenUsage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Return actual usage information:
	// - LastMessage: The token counts from the most recent exchange
	// - Cumulative: Total tokens used across all messages in the session
	return chat.TokenUsage{
		LastMessage: s.lastUsage,
		Cumulative: chat.TokenUsageDetails{
			InputTokens:  0, // Not tracked separately at session level
			OutputTokens: 0, // Not tracked separately at session level
			TotalTokens:  s.cumulativeTokens,
		},
	}, nil
}

// MaxTokens implements chat.Chat
func (s *session) MaxTokens() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Query the current chat for max tokens dynamically
	return s.chat.MaxTokens()
}

// RegisterTool implements chat.Chat
func (s *session) RegisterTool(def chat.ToolDef, fn func(context.Context, string) string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.tools == nil {
		s.tools = make(map[string]registeredTool)
	}

	s.tools[def.Name()] = registeredTool{
		def: def,
		fn:  fn,
	}

	// Also register with underlying chat
	return s.chat.RegisterTool(def, fn)
}

// DeregisterTool implements chat.Chat
func (s *session) DeregisterTool(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.tools, name)
	s.chat.DeregisterTool(name)
}

// ListTools implements chat.Chat
func (s *session) ListTools() []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	var names []string
	for name := range s.tools {
		names = append(names, name)
	}
	return names
}

// LiveRecords returns all records marked as live (in active context window).
func (s *session) LiveRecords() []Record {
	s.mu.Lock()
	defer s.mu.Unlock()

	records, _ := s.store.GetLiveRecords(s.sessionID)
	var result []Record
	for _, r := range records {
		result = append(result, recordFromPersistence(r))
	}
	return result
}

// TotalRecords returns all records (both live and dead).
func (s *session) TotalRecords() []Record {
	s.mu.Lock()
	defer s.mu.Unlock()

	records, _ := s.store.GetAllRecords(s.sessionID)
	var result []Record
	for _, r := range records {
		result = append(result, recordFromPersistence(r))
	}
	return result
}

// CompactNow manually triggers context compaction.
func (s *session) CompactNow() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Use a reasonable timeout for manual compaction
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return s.compactNowLocked(ctx)
}

// compactNowLocked performs compaction with the mutex already held.
func (s *session) compactNowLocked(ctx context.Context) error {
	// Find live records to compact
	liveRecords, _ := s.store.GetLiveRecords(s.sessionID)

	if len(liveRecords) < 3 { // Need at least a few messages to summarize
		return nil
	}

	// Keep last 2 messages, summarize the rest
	recordsToSummarize := liveRecords[:len(liveRecords)-2]

	// Convert persistence.Record to agent.Record for summarizer
	var agentRecords []Record
	for _, r := range recordsToSummarize {
		agentRecords = append(agentRecords, recordFromPersistence(r))
	}

	// Use the configured summarizer with context from the request
	summary, err := s.summarizer.Summarize(ctx, agentRecords)
	if err != nil {
		return fmt.Errorf("summarization failed: %w", err)
	}

	// Mark old records as dead (except last 2)
	for i, r := range liveRecords {
		if i < len(liveRecords)-2 {
			s.store.MarkRecordDead(s.sessionID, r.ID)
		}
	}

	// Add summary as assistant message with tag (safer than system message)
	s.store.AddRecord(s.sessionID, persistence.Record{
		Role:         "assistant",
		Content:      fmt.Sprintf("[Previous conversation summary]\n%s", summary),
		Live:         true,
		Status:       string(RecordStatusSuccess), // Summaries are successful
		InputTokens:  0,                           // Summary tokens will be counted with next message
		OutputTokens: 0,
		Timestamp:    time.Now(),
	})

	// Update compaction metrics
	s.compactionCount++
	s.lastCompaction = time.Now()
	s.saveMetricsLocked()

	return nil
}

// SetCompactionThreshold sets the threshold for automatic compaction (0.0-1.0).
func (s *session) SetCompactionThreshold(threshold float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if threshold < 0 {
		threshold = 0
	}
	if threshold > 1 {
		threshold = 1
	}
	s.compactionThreshold = threshold
	s.saveMetricsLocked()
}

// Metrics returns usage statistics for the session.
func (s *session) Metrics() SessionMetrics {
	s.mu.Lock()
	defer s.mu.Unlock()

	liveTokens := s.calculateLiveTokensLocked()
	liveRecords, _ := s.store.GetLiveRecords(s.sessionID)
	allRecords, _ := s.store.GetAllRecords(s.sessionID)

	// Query max tokens dynamically from current chat
	maxTokens := s.chat.MaxTokens()
	percentFull := 0.0
	if maxTokens > 0 {
		percentFull = float64(liveTokens) / float64(maxTokens)
	}

	return SessionMetrics{
		CumulativeTokens: s.cumulativeTokens,
		LiveTokens:       liveTokens,
		MaxTokens:        maxTokens,
		CompactionCount:  s.compactionCount,
		LastCompaction:   s.lastCompaction,
		RecordsLive:      len(liveRecords),
		RecordsTotal:     len(allRecords),
		PercentFull:      percentFull,
	}
}

// Helper methods - all expect mutex to be held

// shouldCompactLocked checks if compaction is needed (mutex must be held).
func (s *session) shouldCompactLocked() bool {
	// Threshold of 0.0 means never compact
	if s.compactionThreshold == 0.0 {
		return false
	}

	liveTokens := s.calculateLiveTokensLocked()
	// Query max tokens dynamically from current chat
	maxTokens := s.chat.MaxTokens()
	if maxTokens <= 0 {
		return false
	}
	percentFull := float64(liveTokens) / float64(maxTokens)
	return percentFull >= s.compactionThreshold
}

// calculateLiveTokensLocked calculates live token count (mutex must be held).
func (s *session) calculateLiveTokensLocked() int {
	records, _ := s.store.GetLiveRecords(s.sessionID)
	total := 0
	for _, r := range records {
		// Count both input and output tokens
		total += r.InputTokens + r.OutputTokens
	}
	return total
}

// buildChatHistoryLocked builds the chat history (mutex must be held).
func (s *session) buildChatHistoryLocked() (string, []chat.Message) {
	var systemPrompt string
	var msgs []chat.Message

	records, _ := s.store.GetLiveRecords(s.sessionID)
	for _, r := range records {
		if r.Role == "system" {
			if systemPrompt == "" {
				systemPrompt = r.Content
			} else {
				// Append additional system messages
				systemPrompt += "\n\n" + r.Content
			}
		} else {
			msg := chat.Message{Role: chat.Role(r.Role)}
			if r.Content != "" {
				msg.AddText(r.Content)
			}
			for _, tc := range cloneToolCalls(r.ToolCalls) {
				msg.AddToolCall(tc)
			}
			for _, tr := range cloneToolResults(r.ToolResults) {
				msg.AddToolResult(tr)
			}
			msgs = append(msgs, msg)
		}
	}

	return systemPrompt, msgs
}

// saveMetricsLocked saves metrics to store (mutex must be held).
func (s *session) saveMetricsLocked() {
	s.store.SaveMetrics(s.sessionID, persistence.SessionMetrics{
		CompactionCount:     s.compactionCount,
		LastCompaction:      s.lastCompaction,
		CumulativeTokens:    s.cumulativeTokens,
		CompactionThreshold: s.compactionThreshold,
	})
}
