package common

import (
	"sync"

	"github.com/bpowers/go-agent/chat"
)

// State manages message history and token usage with thread-safe operations.
// This simple struct extracts the common pattern all providers share.
type State struct {
	mu sync.Mutex

	systemPrompt string
	messages     []chat.Message

	lastMessageUsage chat.TokenUsageDetails
	cumulativeUsage  chat.TokenUsageDetails
}

// NewState creates a new state manager.
func NewState(systemPrompt string, initialMessages []chat.Message) *State {
	msgs := make([]chat.Message, len(initialMessages))
	copy(msgs, initialMessages)
	return &State{
		systemPrompt: systemPrompt,
		messages:     msgs,
	}
}

// Snapshot returns a copy of the system prompt and message history.
// This allows streaming operations to work with a consistent view of the state
// without holding locks during long-running operations.
func (s *State) Snapshot() (systemPrompt string, messages []chat.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()

	systemPrompt = s.systemPrompt
	messages = make([]chat.Message, len(s.messages))
	copy(messages, s.messages)
	return systemPrompt, messages
}

// AppendMessages adds messages to the history and optionally updates token usage.
func (s *State) AppendMessages(msgs []chat.Message, usage *chat.TokenUsageDetails) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.messages = append(s.messages, msgs...)

	if usage != nil && usage.TotalTokens > 0 {
		s.lastMessageUsage = *usage
		s.cumulativeUsage.InputTokens += usage.InputTokens
		s.cumulativeUsage.OutputTokens += usage.OutputTokens
		s.cumulativeUsage.TotalTokens += usage.TotalTokens
		s.cumulativeUsage.CachedTokens += usage.CachedTokens
	}
}

// History returns the system prompt and a copy of the message history.
func (s *State) History() (string, []chat.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()

	msgs := make([]chat.Message, len(s.messages))
	copy(msgs, s.messages)
	return s.systemPrompt, msgs
}

// TokenUsage returns the token usage statistics.
func (s *State) TokenUsage() (chat.TokenUsage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return chat.TokenUsage{
		LastMessage: s.lastMessageUsage,
		Cumulative:  s.cumulativeUsage,
	}, nil
}

// UpdateUsage updates only the token usage without adding messages.
func (s *State) UpdateUsage(usage chat.TokenUsageDetails) {
	if usage.TotalTokens == 0 {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.lastMessageUsage = usage
	s.cumulativeUsage.InputTokens += usage.InputTokens
	s.cumulativeUsage.OutputTokens += usage.OutputTokens
	s.cumulativeUsage.TotalTokens += usage.TotalTokens
	s.cumulativeUsage.CachedTokens += usage.CachedTokens
}
