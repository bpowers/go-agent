package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/bpowers/go-agent/chat"
)

// Summarizer defines the interface for conversation summarization strategies.
// Implementations can use LLMs, extractive summarization, or other techniques
// to compress conversation history while preserving important context.
type Summarizer interface {
	// Summarize compresses a list of records into a concise summary.
	// The summary should preserve key information, decisions made, and important context.
	Summarize(ctx context.Context, records []Record) (string, error)

	// SetPrompt allows customization of the summarization prompt for LLM-based summarizers.
	SetPrompt(prompt string)
}

// LLMSummarizer uses an LLM to create intelligent conversation summaries.
type LLMSummarizer struct {
	client chat.Client
	model  string
	prompt string
}

// NewLLMSummarizer creates a new LLM-based summarizer.
// The model parameter can specify a different (usually cheaper) model for summarization.
func NewLLMSummarizer(client chat.Client, model string) *LLMSummarizer {
	return &LLMSummarizer{
		client: client,
		model:  model,
		prompt: defaultSummarizationPrompt,
	}
}

// SetPrompt updates the summarization prompt.
func (s *LLMSummarizer) SetPrompt(prompt string) {
	s.prompt = prompt
}

// Summarize uses an LLM to create a concise summary of the conversation.
func (s *LLMSummarizer) Summarize(ctx context.Context, records []Record) (string, error) {
	if len(records) == 0 {
		return "", nil
	}

	// Build conversation text
	var conversation strings.Builder
	for _, r := range records {
		conversation.WriteString(fmt.Sprintf("%s: %s\n\n", r.Role, r.Content))
	}

	// Create summarization request
	summaryPrompt := fmt.Sprintf("%s\n\nConversation to summarize:\n%s", s.prompt, conversation.String())

	// Create a chat session with the summarization model
	summaryChat := s.client.NewChat("You are an assistant tasked with summarizing conversations.")

	// Get the summary
	response, err := summaryChat.Message(ctx, chat.Message{
		Role:    chat.UserRole,
		Content: summaryPrompt,
	})
	if err != nil {
		return "", fmt.Errorf("summarization failed: %w", err)
	}

	return response.Content, nil
}

// defaultSummarizationPrompt is the default prompt for LLM-based summarization.
const defaultSummarizationPrompt = `Please provide a concise summary of the following conversation that preserves the key information, decisions made, and any important context. The summary should be suitable for continuing the conversation later.

Focus on:
- Main topics discussed
- Key decisions or conclusions reached
- Important context that affects future conversation
- Any unresolved questions or action items

The summary must be in markdown format.

Provide only the summary, no additional commentary, relying **strictly** on the provided text.`

// SimpleSummarizer provides a basic extractive summarization strategy.
// It keeps the first and last N messages without compression.
type SimpleSummarizer struct {
	keepFirst int
	keepLast  int
}

// NewSimpleSummarizer creates a basic summarizer that keeps first and last messages.
func NewSimpleSummarizer(keepFirst, keepLast int) *SimpleSummarizer {
	return &SimpleSummarizer{
		keepFirst: keepFirst,
		keepLast:  keepLast,
	}
}

// SetPrompt is a no-op for SimpleSummarizer.
func (s *SimpleSummarizer) SetPrompt(prompt string) {
	// No-op for simple summarizer
}

// Summarize returns a simple extraction of first and last messages.
func (s *SimpleSummarizer) Summarize(ctx context.Context, records []Record) (string, error) {
	if len(records) == 0 {
		return "", nil
	}

	var result strings.Builder
	result.WriteString("[Previous conversation summary]\n")

	// Keep first N messages
	firstCount := s.keepFirst
	if firstCount > len(records) {
		firstCount = len(records)
	}

	for i := 0; i < firstCount; i++ {
		result.WriteString(fmt.Sprintf("%s: %s\n", records[i].Role, records[i].Content))
	}

	// If we have more messages than we're keeping, add ellipsis
	if len(records) > s.keepFirst+s.keepLast {
		result.WriteString("\n... [middle portion omitted] ...\n\n")

		// Keep last N messages
		for i := len(records) - s.keepLast; i < len(records); i++ {
			result.WriteString(fmt.Sprintf("%s: %s\n", records[i].Role, records[i].Content))
		}
	}

	return result.String(), nil
}
