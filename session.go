package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/persistence"
)

// Session manages conversation lifecycle with automatic context compaction.
// It wraps a chat.Chat instance and provides additional features like
// token tracking, record management, and automatic summarization.
type Session interface {
	// Embeds chat.Chat for full compatibility
	chat.Chat

	// Session-specific methods
	LiveRecords() []Record          // Get active context window records
	TotalRecords() []Record         // Get all records (live + dead)
	CompactNow() error              // Manually trigger compaction
	SetCompactionThreshold(float64) // Set when to compact (0.0-1.0, default 0.8)

	// Metrics
	SessionMetrics() SessionMetrics // Token usage, compaction stats, etc.
}

// Record represents a conversation turn in the session history.
type Record struct {
	ID        int64     `json:"id,omitzero"`
	Role      chat.Role `json:"role"`
	Content   string    `json:"content"`
	Live      bool      `json:"live"`   // In active context window
	Tokens    int       `json:"tokens"` // Estimated token count
	Timestamp time.Time `json:"timestamp"`
}

// SessionMetrics provides usage statistics for the session.
type SessionMetrics struct {
	TotalTokens     int       `json:"total_tokens"`     // Cumulative tokens used
	LiveTokens      int       `json:"live_tokens"`      // Tokens in active window
	MaxTokens       int       `json:"max_tokens"`       // Model's max context size
	CompactionCount int       `json:"compaction_count"` // Number of compactions
	LastCompaction  time.Time `json:"last_compaction"`  // When last compacted
	RecordsLive     int       `json:"records_live"`     // Number of live records
	RecordsTotal    int       `json:"records_total"`    // Total records (live + dead)
	PercentFull     float64   `json:"percent_full"`     // LiveTokens/MaxTokens
}

// SessionOption configures a Session.
type SessionOption func(*sessionOptions)

type sessionOptions struct {
	store           persistence.Store
	initialMessages []chat.Message
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

// NewSession creates a new Session with the given client, system prompt, and options.
func NewSession(client chat.Client, systemPrompt string, opts ...SessionOption) Session {
	// Apply options
	var options sessionOptions
	for _, opt := range opts {
		if opt != nil {
			opt(&options)
		}
	}

	// Default to memory store if not specified
	if options.store == nil {
		options.store = persistence.NewMemoryStore()
	}

	// Load existing metrics if available
	metrics, _ := options.store.LoadMetrics()
	if metrics.CompactionThreshold == 0 {
		metrics.CompactionThreshold = 0.8
	}

	// Check if we have existing records in the store
	existingRecords, _ := options.store.GetAllRecords()
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
			tokens := estimateTokens(systemPrompt)
			options.store.AddRecord(persistence.Record{
				Role:      "system",
				Content:   systemPrompt,
				Live:      true,
				Tokens:    tokens,
				Timestamp: time.Now(),
			})
		}

		for _, msg := range options.initialMessages {
			tokens := estimateTokens(msg.Content)
			options.store.AddRecord(persistence.Record{
				Role:      chat.Role(msg.Role),
				Content:   msg.Content,
				Live:      true,
				Tokens:    tokens,
				Timestamp: time.Now(),
			})
		}
	}

	return &persistentSession{
		chat:                baseChat,
		client:              client,
		systemPrompt:        actualSystemPrompt,
		store:               options.store,
		compactionThreshold: metrics.CompactionThreshold,
		compactionCount:     metrics.CompactionCount,
		lastCompaction:      metrics.LastCompaction,
		cumulativeTokens:    metrics.CumulativeTokens,
		maxTokens:           baseChat.MaxTokens(),
		tools:               make(map[string]registeredTool),
	}
}

// persistentSession is the implementation of Session with pluggable storage.
type persistentSession struct {
	chat         chat.Chat
	client       chat.Client
	systemPrompt string
	store        persistence.Store

	mu                  sync.Mutex
	compactionThreshold float64
	compactionCount     int
	lastCompaction      time.Time
	cumulativeTokens    int
	maxTokens           int

	// Tool tracking - use single mutex for simplicity as per CLAUDE.md
	tools map[string]registeredTool
}

type registeredTool struct {
	def chat.ToolDef
	fn  func(context.Context, string) string
}

// Message implements chat.Chat
func (s *persistentSession) Message(ctx context.Context, msg chat.Message, opts ...chat.Option) (chat.Message, error) {
	// Add user message and check compaction
	tempChat, err := s.prepareForMessage(msg)
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

// MessageStream implements chat.Chat
func (s *persistentSession) MessageStream(ctx context.Context, msg chat.Message, callback chat.StreamCallback, opts ...chat.Option) (chat.Message, error) {
	// Add user message and check compaction
	tempChat, err := s.prepareForMessage(msg)
	if err != nil {
		return chat.Message{}, err
	}

	// Send message with streaming
	response, err := tempChat.MessageStream(ctx, msg, callback, opts...)
	if err != nil {
		return response, err
	}

	// Track response
	s.trackResponse(tempChat, response)
	return response, nil
}

// prepareForMessage adds the user message, checks for compaction, and returns a prepared chat.
// This method expects the mutex is NOT held and will handle locking internally.
func (s *persistentSession) prepareForMessage(msg chat.Message) (chat.Chat, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Add user message to records
	userTokens := estimateTokens(msg.Content)
	s.store.AddRecord(persistence.Record{
		Role:      chat.Role(msg.Role),
		Content:   msg.Content,
		Live:      true,
		Tokens:    userTokens,
		Timestamp: time.Now(),
	})

	// Check if we need to compact before sending
	if s.shouldCompactLocked() {
		// We need to compact, but CompactNow needs the lock too
		// So we use a locked variant
		if err := s.compactNowLocked(); err != nil {
			return nil, fmt.Errorf("auto-compaction failed: %w", err)
		}
	}

	// Build the message history from live records
	systemPrompt, msgs := s.buildChatHistoryLocked()

	// Recreate chat with current history
	tempChat := s.client.NewChat(systemPrompt, msgs...)

	// Re-register tools
	for _, tool := range s.tools {
		if err := tempChat.RegisterTool(tool.def, tool.fn); err != nil {
			return nil, fmt.Errorf("failed to re-register tool %s: %w", tool.def.Name(), err)
		}
	}

	return tempChat, nil
}

// trackResponse records the response and updates metrics.
// This method expects the mutex is NOT held and will handle locking internally.
func (s *persistentSession) trackResponse(tempChat chat.Chat, response chat.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()

	responseTokens := estimateTokens(response.Content)
	s.store.AddRecord(persistence.Record{
		Role:      chat.Role(response.Role),
		Content:   response.Content,
		Live:      true,
		Tokens:    responseTokens,
		Timestamp: time.Now(),
	})

	// Update token usage
	usage, _ := tempChat.TokenUsage()
	s.cumulativeTokens += usage.LastMessage.TotalTokens

	// Save metrics
	s.saveMetricsLocked()
}

// History implements chat.Chat
func (s *persistentSession) History() (systemPrompt string, msgs []chat.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.buildChatHistoryLocked()
}

// TokenUsage implements chat.Chat
func (s *persistentSession) TokenUsage() (chat.TokenUsage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	liveTokens := s.calculateLiveTokensLocked()

	return chat.TokenUsage{
		LastMessage: chat.TokenUsageDetails{
			InputTokens:  0, // Not tracked at session level
			OutputTokens: 0, // Not tracked at session level
			TotalTokens:  0, // Not tracked at session level
		},
		Cumulative: chat.TokenUsageDetails{
			InputTokens:  liveTokens,
			OutputTokens: 0, // Not tracked separately at session level
			TotalTokens:  s.cumulativeTokens,
		},
	}, nil
}

// MaxTokens implements chat.Chat
func (s *persistentSession) MaxTokens() int {
	return s.maxTokens
}

// RegisterTool implements chat.Chat
func (s *persistentSession) RegisterTool(def chat.ToolDef, fn func(context.Context, string) string) error {
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
func (s *persistentSession) DeregisterTool(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.tools, name)
	s.chat.DeregisterTool(name)
}

// ListTools implements chat.Chat
func (s *persistentSession) ListTools() []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	var names []string
	for name := range s.tools {
		names = append(names, name)
	}
	return names
}

// LiveRecords returns all records marked as live (in active context window).
func (s *persistentSession) LiveRecords() []Record {
	s.mu.Lock()
	defer s.mu.Unlock()

	records, _ := s.store.GetLiveRecords()
	var result []Record
	for _, r := range records {
		result = append(result, Record{
			ID:        r.ID,
			Role:      chat.Role(r.Role),
			Content:   r.Content,
			Live:      r.Live,
			Tokens:    r.Tokens,
			Timestamp: r.Timestamp,
		})
	}
	return result
}

// TotalRecords returns all records (both live and dead).
func (s *persistentSession) TotalRecords() []Record {
	s.mu.Lock()
	defer s.mu.Unlock()

	records, _ := s.store.GetAllRecords()
	var result []Record
	for _, r := range records {
		result = append(result, Record{
			ID:        r.ID,
			Role:      chat.Role(r.Role),
			Content:   r.Content,
			Live:      r.Live,
			Tokens:    r.Tokens,
			Timestamp: r.Timestamp,
		})
	}
	return result
}

// CompactNow manually triggers context compaction.
func (s *persistentSession) CompactNow() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.compactNowLocked()
}

// compactNowLocked performs compaction with the mutex already held.
func (s *persistentSession) compactNowLocked() error {
	// Find live records to compact
	liveRecords, _ := s.store.GetLiveRecords()

	if len(liveRecords) < 3 { // Need at least a few messages to summarize
		return nil
	}

	// Build conversation text for summarization
	var conversation strings.Builder
	for _, r := range liveRecords[:len(liveRecords)-2] { // Keep last 2 messages
		conversation.WriteString(fmt.Sprintf("%s: %s\n\n", r.Role, r.Content))
	}

	// Create summarization prompt
	summaryPrompt := fmt.Sprintf(`Please provide a concise summary of the following conversation that preserves the key information, decisions made, and any important context. The summary should be suitable for continuing the conversation later.

Conversation to summarize:
%s

Provide only the summary, no additional commentary.`, conversation.String())

	// Use a cheaper model for summarization
	summaryClient := s.client // In real impl, would create client with cheaper model
	summaryChat := summaryClient.NewChat("You are a helpful assistant that creates concise conversation summaries.")

	summaryMsg, err := summaryChat.Message(context.Background(), chat.Message{
		Role:    chat.UserRole,
		Content: summaryPrompt,
	})
	if err != nil {
		return fmt.Errorf("summarization failed: %w", err)
	}

	// Mark old records as dead (except last 2)
	for i, r := range liveRecords {
		if i < len(liveRecords)-2 {
			s.store.MarkRecordDead(r.ID)
		}
	}

	// Add summary as new record
	summaryTokens := estimateTokens(summaryMsg.Content)
	s.store.AddRecord(persistence.Record{
		Role:      "system",
		Content:   fmt.Sprintf("[Previous conversation summary]\n%s", summaryMsg.Content),
		Live:      true,
		Tokens:    summaryTokens,
		Timestamp: time.Now(),
	})

	// Update compaction metrics
	s.compactionCount++
	s.lastCompaction = time.Now()
	s.saveMetricsLocked()

	return nil
}

// SetCompactionThreshold sets the threshold for automatic compaction (0.0-1.0).
func (s *persistentSession) SetCompactionThreshold(threshold float64) {
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

// SessionMetrics returns usage statistics for the session.
func (s *persistentSession) SessionMetrics() SessionMetrics {
	s.mu.Lock()
	defer s.mu.Unlock()

	liveTokens := s.calculateLiveTokensLocked()
	liveRecords, _ := s.store.GetLiveRecords()
	allRecords, _ := s.store.GetAllRecords()

	percentFull := 0.0
	if s.maxTokens > 0 {
		percentFull = float64(liveTokens) / float64(s.maxTokens)
	}

	return SessionMetrics{
		TotalTokens:     s.cumulativeTokens,
		LiveTokens:      liveTokens,
		MaxTokens:       s.maxTokens,
		CompactionCount: s.compactionCount,
		LastCompaction:  s.lastCompaction,
		RecordsLive:     len(liveRecords),
		RecordsTotal:    len(allRecords),
		PercentFull:     percentFull,
	}
}

// Helper methods - all expect mutex to be held

// shouldCompactLocked checks if compaction is needed (mutex must be held).
func (s *persistentSession) shouldCompactLocked() bool {
	liveTokens := s.calculateLiveTokensLocked()
	if s.maxTokens <= 0 {
		return false
	}
	percentFull := float64(liveTokens) / float64(s.maxTokens)
	return percentFull >= s.compactionThreshold
}

// calculateLiveTokensLocked calculates live token count (mutex must be held).
func (s *persistentSession) calculateLiveTokensLocked() int {
	records, _ := s.store.GetLiveRecords()
	total := 0
	for _, r := range records {
		total += r.Tokens
	}
	return total
}

// buildChatHistoryLocked builds the chat history (mutex must be held).
func (s *persistentSession) buildChatHistoryLocked() (string, []chat.Message) {
	var systemPrompt string
	var msgs []chat.Message

	records, _ := s.store.GetLiveRecords()
	for _, r := range records {
		if r.Role == "system" {
			if systemPrompt == "" {
				systemPrompt = r.Content
			} else {
				// Append additional system messages
				systemPrompt += "\n\n" + r.Content
			}
		} else {
			msgs = append(msgs, chat.Message{
				Role:    chat.Role(r.Role),
				Content: r.Content,
			})
		}
	}

	return systemPrompt, msgs
}

// saveMetricsLocked saves metrics to store (mutex must be held).
func (s *persistentSession) saveMetricsLocked() {
	s.store.SaveMetrics(persistence.SessionMetrics{
		CompactionCount:     s.compactionCount,
		LastCompaction:      s.lastCompaction,
		CumulativeTokens:    s.cumulativeTokens,
		CompactionThreshold: s.compactionThreshold,
	})
}

// estimateTokens provides a rough token count estimate.
// In production, this would use a proper tokenizer.
func estimateTokens(text string) int {
	// Rough estimate: ~4 characters per token
	return len(text) / 4
}
