package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/persistence"
)

// Session manages conversation lifecycle with automatic context compaction.
// It embeds chat.Chat for full compatibility while adding persistence and
// automatic summarization capabilities. When the context window approaches
// capacity (default 80%), older messages are automatically compacted into
// summaries to maintain conversation continuity.
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
	ID           int64     `json:"id,omitzero"`
	Role         chat.Role `json:"role"`
	Content      string    `json:"content"`
	Live         bool      `json:"live"`          // In active context window
	InputTokens  int       `json:"input_tokens"`  // Actual tokens from LLM
	OutputTokens int       `json:"output_tokens"` // Actual tokens from LLM
	Timestamp    time.Time `json:"timestamp"`
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
	summarizer      Summarizer
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

	// Default to memory store if not specified
	if options.store == nil {
		options.store = persistence.NewMemoryStore()
	}

	// Default to LLM summarizer if not specified
	if options.summarizer == nil {
		// Use same client but could specify a cheaper model here
		options.summarizer = NewLLMSummarizer(client, "")
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
			options.store.AddRecord(persistence.Record{
				Role:         "system",
				Content:      systemPrompt,
				Live:         true,
				InputTokens:  0, // System prompt tokens counted with first message
				OutputTokens: 0,
				Timestamp:    time.Now(),
			})
		}

		for _, msg := range options.initialMessages {
			options.store.AddRecord(persistence.Record{
				Role:         chat.Role(msg.Role),
				Content:      msg.Content,
				Live:         true,
				InputTokens:  0, // Initial messages' tokens counted with first query
				OutputTokens: 0,
				Timestamp:    time.Now(),
			})
		}
	}

	return &persistentSession{
		chat:                baseChat,
		client:              client,
		systemPrompt:        actualSystemPrompt,
		store:               options.store,
		summarizer:          options.summarizer,
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
	summarizer   Summarizer

	mu                  sync.Mutex
	compactionThreshold float64
	compactionCount     int
	lastCompaction      time.Time
	cumulativeTokens    int
	maxTokens           int
	lastUsage           chat.TokenUsageDetails

	// Tool tracking - use single mutex for simplicity as per CLAUDE.md
	tools             map[string]registeredTool
	lastUserMessageID int64
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

	// Add user message to records (tokens will be updated after response)
	userMsgID, _ := s.store.AddRecord(persistence.Record{
		Role:         chat.Role(msg.Role),
		Content:      msg.Content,
		Live:         true,
		InputTokens:  0, // Will be updated after response
		OutputTokens: 0,
		Timestamp:    time.Now(),
	})

	// Store message ID for later token update
	s.lastUserMessageID = userMsgID

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

// trackResponse records the response and updates metrics with actual token counts.
// This method expects the mutex is NOT held and will handle locking internally.
func (s *persistentSession) trackResponse(tempChat chat.Chat, response chat.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Get actual token usage from the LLM
	usage, _ := tempChat.TokenUsage()
	s.lastUsage = usage.LastMessage
	s.cumulativeTokens += usage.LastMessage.TotalTokens

	// Update the user message with actual input tokens
	if s.lastUserMessageID > 0 {
		userRec, _ := s.store.GetAllRecords()
		for i := len(userRec) - 2; i >= 0 && i < len(userRec); i++ {
			if userRec[i].ID == s.lastUserMessageID {
				userRec[i].InputTokens = usage.LastMessage.InputTokens
				s.store.UpdateRecord(s.lastUserMessageID, userRec[i])
				break
			}
		}
	}

	// Add response with actual output tokens
	s.store.AddRecord(persistence.Record{
		Role:         chat.Role(response.Role),
		Content:      response.Content,
		Live:         true,
		InputTokens:  0, // Input tokens are on the user message
		OutputTokens: usage.LastMessage.OutputTokens,
		Timestamp:    time.Now(),
	})

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
			ID:           r.ID,
			Role:         chat.Role(r.Role),
			Content:      r.Content,
			Live:         r.Live,
			InputTokens:  r.InputTokens,
			OutputTokens: r.OutputTokens,
			Timestamp:    r.Timestamp,
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
			ID:           r.ID,
			Role:         chat.Role(r.Role),
			Content:      r.Content,
			Live:         r.Live,
			InputTokens:  r.InputTokens,
			OutputTokens: r.OutputTokens,
			Timestamp:    r.Timestamp,
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

	// Keep last 2 messages, summarize the rest
	recordsToSummarize := liveRecords[:len(liveRecords)-2]

	// Convert persistence.Record to agent.Record for summarizer
	var agentRecords []Record
	for _, r := range recordsToSummarize {
		agentRecords = append(agentRecords, Record{
			ID:           r.ID,
			Role:         chat.Role(r.Role),
			Content:      r.Content,
			Live:         r.Live,
			InputTokens:  r.InputTokens,
			OutputTokens: r.OutputTokens,
			Timestamp:    r.Timestamp,
		})
	}

	// Use the configured summarizer
	summary, err := s.summarizer.Summarize(context.Background(), agentRecords)
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
	s.store.AddRecord(persistence.Record{
		Role:         "system",
		Content:      fmt.Sprintf("[Previous conversation summary]\n%s", summary),
		Live:         true,
		InputTokens:  0, // Summary tokens will be counted with next message
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
		// Count both input and output tokens
		total += r.InputTokens + r.OutputTokens
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
