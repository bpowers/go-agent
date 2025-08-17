package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/bpowers/go-agent/chat"
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

// NewSession creates a new Session with the given client and system prompt.
func NewSession(client chat.Client, systemPrompt string, initialMsgs ...chat.Message) Session {
	baseChat := client.NewChat(systemPrompt, initialMsgs...)

	// Create initial records from system prompt and initial messages
	var records []Record
	if systemPrompt != "" {
		tokens := estimateTokens(systemPrompt)
		records = append(records, Record{
			Role:      "system",
			Content:   systemPrompt,
			Live:      true,
			Tokens:    tokens,
			Timestamp: time.Now(),
		})
	}

	for _, msg := range initialMsgs {
		tokens := estimateTokens(msg.Content)
		records = append(records, Record{
			Role:      msg.Role,
			Content:   msg.Content,
			Live:      true,
			Tokens:    tokens,
			Timestamp: time.Now(),
		})
	}

	return &memorySession{
		chat:                baseChat,
		client:              client,
		systemPrompt:        systemPrompt,
		records:             records,
		compactionThreshold: 0.8,
		maxTokens:           baseChat.MaxTokens(),
	}
}

// memorySession is the in-memory implementation of Session.
type memorySession struct {
	chat         chat.Chat
	client       chat.Client
	systemPrompt string

	mu                  sync.Mutex
	records             []Record
	compactionThreshold float64
	compactionCount     int
	lastCompaction      time.Time
	cumulativeTokens    int
	maxTokens           int

	// Tool tracking
	tools     map[string]registeredTool
	toolsLock sync.RWMutex
}

type registeredTool struct {
	def chat.ToolDef
	fn  func(context.Context, string) string
}

// Message implements chat.Chat
func (s *memorySession) Message(ctx context.Context, msg chat.Message, opts ...chat.Option) (chat.Message, error) {
	s.mu.Lock()

	// Add user message to records
	userTokens := estimateTokens(msg.Content)
	s.records = append(s.records, Record{
		Role:      msg.Role,
		Content:   msg.Content,
		Live:      true,
		Tokens:    userTokens,
		Timestamp: time.Now(),
	})

	// Check if we need to compact before sending
	if s.shouldCompact() {
		s.mu.Unlock()
		if err := s.CompactNow(); err != nil {
			return chat.Message{}, fmt.Errorf("auto-compaction failed: %w", err)
		}
		s.mu.Lock()
	}

	// Build the message history from live records
	systemPrompt, msgs := s.buildChatHistory()
	s.mu.Unlock()

	// Recreate chat with current history
	tempChat := s.client.NewChat(systemPrompt, msgs...)

	// Re-register tools
	s.toolsLock.RLock()
	for _, tool := range s.tools {
		if err := tempChat.RegisterTool(tool.def, tool.fn); err != nil {
			s.toolsLock.RUnlock()
			return chat.Message{}, fmt.Errorf("failed to re-register tool %s: %w", tool.def.Name(), err)
		}
	}
	s.toolsLock.RUnlock()

	// Send message
	response, err := tempChat.Message(ctx, msg, opts...)
	if err != nil {
		return response, err
	}

	// Track response
	s.mu.Lock()
	responseTokens := estimateTokens(response.Content)
	s.records = append(s.records, Record{
		Role:      response.Role,
		Content:   response.Content,
		Live:      true,
		Tokens:    responseTokens,
		Timestamp: time.Now(),
	})

	// Update token usage
	usage, _ := tempChat.TokenUsage()
	s.cumulativeTokens += usage.TotalTokens
	s.mu.Unlock()

	return response, nil
}

// MessageStream implements chat.Chat
func (s *memorySession) MessageStream(ctx context.Context, msg chat.Message, callback chat.StreamCallback, opts ...chat.Option) (chat.Message, error) {
	s.mu.Lock()

	// Add user message to records
	userTokens := estimateTokens(msg.Content)
	s.records = append(s.records, Record{
		Role:      msg.Role,
		Content:   msg.Content,
		Live:      true,
		Tokens:    userTokens,
		Timestamp: time.Now(),
	})

	// Check if we need to compact before sending
	if s.shouldCompact() {
		s.mu.Unlock()
		if err := s.CompactNow(); err != nil {
			return chat.Message{}, fmt.Errorf("auto-compaction failed: %w", err)
		}
		s.mu.Lock()
	}

	// Build the message history from live records
	systemPrompt, msgs := s.buildChatHistory()
	s.mu.Unlock()

	// Recreate chat with current history
	tempChat := s.client.NewChat(systemPrompt, msgs...)

	// Re-register tools
	s.toolsLock.RLock()
	for _, tool := range s.tools {
		if err := tempChat.RegisterTool(tool.def, tool.fn); err != nil {
			s.toolsLock.RUnlock()
			return chat.Message{}, fmt.Errorf("failed to re-register tool %s: %w", tool.def.Name(), err)
		}
	}
	s.toolsLock.RUnlock()

	// Send message with streaming
	response, err := tempChat.MessageStream(ctx, msg, callback, opts...)
	if err != nil {
		return response, err
	}

	// Track response
	s.mu.Lock()
	responseTokens := estimateTokens(response.Content)
	s.records = append(s.records, Record{
		Role:      response.Role,
		Content:   response.Content,
		Live:      true,
		Tokens:    responseTokens,
		Timestamp: time.Now(),
	})

	// Update token usage
	usage, _ := tempChat.TokenUsage()
	s.cumulativeTokens += usage.TotalTokens
	s.mu.Unlock()

	return response, nil
}

// History implements chat.Chat
func (s *memorySession) History() (systemPrompt string, msgs []chat.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.buildChatHistory()
}

// TokenUsage implements chat.Chat
func (s *memorySession) TokenUsage() (chat.TokenUsage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	liveTokens := s.calculateLiveTokens()

	return chat.TokenUsage{
		InputTokens:  liveTokens,
		OutputTokens: 0, // Not tracked separately at session level
		TotalTokens:  s.cumulativeTokens,
	}, nil
}

// MaxTokens implements chat.Chat
func (s *memorySession) MaxTokens() int {
	return s.maxTokens
}

// RegisterTool implements chat.Chat
func (s *memorySession) RegisterTool(def chat.ToolDef, fn func(context.Context, string) string) error {
	s.toolsLock.Lock()
	defer s.toolsLock.Unlock()

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
func (s *memorySession) DeregisterTool(name string) {
	s.toolsLock.Lock()
	defer s.toolsLock.Unlock()

	delete(s.tools, name)
	s.chat.DeregisterTool(name)
}

// ListTools implements chat.Chat
func (s *memorySession) ListTools() []string {
	s.toolsLock.RLock()
	defer s.toolsLock.RUnlock()

	var names []string
	for name := range s.tools {
		names = append(names, name)
	}
	return names
}

// LiveRecords returns all records marked as live (in active context window).
func (s *memorySession) LiveRecords() []Record {
	s.mu.Lock()
	defer s.mu.Unlock()

	var live []Record
	for _, r := range s.records {
		if r.Live {
			live = append(live, r)
		}
	}
	return live
}

// TotalRecords returns all records (both live and dead).
func (s *memorySession) TotalRecords() []Record {
	s.mu.Lock()
	defer s.mu.Unlock()

	result := make([]Record, len(s.records))
	copy(result, s.records)
	return result
}

// CompactNow manually triggers context compaction.
func (s *memorySession) CompactNow() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Find live records to compact
	var liveRecords []Record
	for _, r := range s.records {
		if r.Live {
			liveRecords = append(liveRecords, r)
		}
	}

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

	// Mark old records as dead
	for i := range s.records {
		if s.records[i].Live && i < len(s.records)-2 {
			s.records[i].Live = false
		}
	}

	// Add summary as new record
	summaryTokens := estimateTokens(summaryMsg.Content)
	summaryRecord := Record{
		Role:      "system",
		Content:   fmt.Sprintf("[Previous conversation summary]\n%s", summaryMsg.Content),
		Live:      true,
		Tokens:    summaryTokens,
		Timestamp: time.Now(),
	}

	// Insert summary before the last 2 messages
	newRecords := make([]Record, 0, len(s.records)+1)
	newRecords = append(newRecords, s.records[:len(s.records)-2]...)
	newRecords = append(newRecords, summaryRecord)
	newRecords = append(newRecords, s.records[len(s.records)-2:]...)
	s.records = newRecords

	// Update compaction metrics
	s.compactionCount++
	s.lastCompaction = time.Now()

	return nil
}

// SetCompactionThreshold sets the threshold for automatic compaction (0.0-1.0).
func (s *memorySession) SetCompactionThreshold(threshold float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if threshold < 0 {
		threshold = 0
	}
	if threshold > 1 {
		threshold = 1
	}
	s.compactionThreshold = threshold
}

// SessionMetrics returns usage statistics for the session.
func (s *memorySession) SessionMetrics() SessionMetrics {
	s.mu.Lock()
	defer s.mu.Unlock()

	liveTokens := s.calculateLiveTokens()
	liveCount := 0
	for _, r := range s.records {
		if r.Live {
			liveCount++
		}
	}

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
		RecordsLive:     liveCount,
		RecordsTotal:    len(s.records),
		PercentFull:     percentFull,
	}
}

// Helper methods

func (s *memorySession) shouldCompact() bool {
	liveTokens := s.calculateLiveTokens()
	if s.maxTokens <= 0 {
		return false
	}
	percentFull := float64(liveTokens) / float64(s.maxTokens)
	return percentFull >= s.compactionThreshold
}

func (s *memorySession) calculateLiveTokens() int {
	total := 0
	for _, r := range s.records {
		if r.Live {
			total += r.Tokens
		}
	}
	return total
}

func (s *memorySession) buildChatHistory() (string, []chat.Message) {
	var systemPrompt string
	var msgs []chat.Message

	for _, r := range s.records {
		if !r.Live {
			continue
		}

		if r.Role == "system" {
			if systemPrompt == "" {
				systemPrompt = r.Content
			} else {
				// Append additional system messages
				systemPrompt += "\n\n" + r.Content
			}
		} else {
			msgs = append(msgs, chat.Message{
				Role:    r.Role,
				Content: r.Content,
			})
		}
	}

	return systemPrompt, msgs
}

// estimateTokens provides a rough token count estimate.
// In production, this would use a proper tokenizer.
func estimateTokens(text string) int {
	// Rough estimate: ~4 characters per token
	return len(text) / 4
}
