package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/bpowers/go-agent/chat"
)

func TestSimpleSummarizer(t *testing.T) {
	summarizer := NewSimpleSummarizer(2, 2)

	records := []Record{
		{Role: chat.UserRole, Content: "First message"},
		{Role: chat.AssistantRole, Content: "First response"},
		{Role: chat.UserRole, Content: "Second message"},
		{Role: chat.AssistantRole, Content: "Second response"},
		{Role: chat.UserRole, Content: "Third message"},
		{Role: chat.AssistantRole, Content: "Third response"},
		{Role: chat.UserRole, Content: "Fourth message"},
		{Role: chat.AssistantRole, Content: "Fourth response"},
	}

	summary, err := summarizer.Summarize(context.Background(), records)
	assert.NoError(t, err)

	// Should keep first 2 and last 2 messages
	assert.Contains(t, summary, "First message")
	assert.Contains(t, summary, "First response")
	assert.Contains(t, summary, "Fourth message")
	assert.Contains(t, summary, "Fourth response")
	assert.Contains(t, summary, "middle portion omitted")
}

func TestSimpleSummarizerWithFewRecords(t *testing.T) {
	summarizer := NewSimpleSummarizer(2, 2)

	records := []Record{
		{Role: chat.UserRole, Content: "Only message"},
		{Role: chat.AssistantRole, Content: "Only response"},
	}

	summary, err := summarizer.Summarize(context.Background(), records)
	assert.NoError(t, err)

	// Should include all messages when total is less than keep threshold
	assert.Contains(t, summary, "Only message")
	assert.Contains(t, summary, "Only response")
	assert.NotContains(t, summary, "middle portion omitted")
}

func TestSimpleSummarizerEmptyRecords(t *testing.T) {
	summarizer := NewSimpleSummarizer(2, 2)

	summary, err := summarizer.Summarize(context.Background(), []Record{})
	assert.NoError(t, err)
	assert.Empty(t, summary)
}

// mockSummarizerClient for testing LLMSummarizer
type mockSummarizerClient struct {
	response string
}

func (m *mockSummarizerClient) NewChat(systemPrompt string, initialMsgs ...chat.Message) chat.Chat {
	return &mockSummarizerChat{
		systemPrompt: systemPrompt,
		response:     m.response,
	}
}

type mockSummarizerChat struct {
	systemPrompt string
	response     string
}

func (m *mockSummarizerChat) Message(ctx context.Context, msg chat.Message, opts ...chat.Option) (chat.Message, error) {
	return chat.Message{
		Role:    chat.AssistantRole,
		Content: m.response,
	}, nil
}

func (m *mockSummarizerChat) MessageStream(ctx context.Context, msg chat.Message, callback chat.StreamCallback, opts ...chat.Option) (chat.Message, error) {
	return m.Message(ctx, msg, opts...)
}

func (m *mockSummarizerChat) History() (systemPrompt string, msgs []chat.Message) {
	return m.systemPrompt, nil
}

func (m *mockSummarizerChat) TokenUsage() (chat.TokenUsage, error) {
	return chat.TokenUsage{}, nil
}

func (m *mockSummarizerChat) MaxTokens() int {
	return 4096
}

func (m *mockSummarizerChat) RegisterTool(def chat.ToolDef, fn func(context.Context, string) string) error {
	return nil
}

func (m *mockSummarizerChat) DeregisterTool(name string) {}

func (m *mockSummarizerChat) ListTools() []string {
	return nil
}

func TestLLMSummarizer(t *testing.T) {
	mockClient := &mockSummarizerClient{
		response: "The user asked about Go and received information about its concurrency features.",
	}

	summarizer := NewLLMSummarizer(mockClient, "")

	records := []Record{
		{Role: chat.UserRole, Content: "Tell me about Go"},
		{Role: chat.AssistantRole, Content: "Go is a programming language with great concurrency support through goroutines and channels."},
	}

	summary, err := summarizer.Summarize(context.Background(), records)
	assert.NoError(t, err)
	assert.Equal(t, "The user asked about Go and received information about its concurrency features.", summary)
}

func TestLLMSummarizerCustomPrompt(t *testing.T) {
	mockClient := &mockSummarizerClient{
		response: "Brief summary",
	}

	summarizer := NewLLMSummarizer(mockClient, "")
	customPrompt := "Make it very brief"
	summarizer.SetPrompt(customPrompt)

	records := []Record{
		{Role: chat.UserRole, Content: "Long conversation"},
	}

	// The mock will return the predefined response
	summary, err := summarizer.Summarize(context.Background(), records)
	assert.NoError(t, err)
	assert.Equal(t, "Brief summary", summary)
}

func TestLLMSummarizerEmptyRecords(t *testing.T) {
	mockClient := &mockSummarizerClient{
		response: "",
	}

	summarizer := NewLLMSummarizer(mockClient, "")

	summary, err := summarizer.Summarize(context.Background(), []Record{})
	assert.NoError(t, err)
	assert.Empty(t, summary)
}

func TestSummarizerBuildsCorrectPrompt(t *testing.T) {
	// This test verifies that the summarizer correctly formats the conversation
	records := []Record{
		{Role: chat.UserRole, Content: "Hello"},
		{Role: chat.AssistantRole, Content: "Hi there"},
		{Role: chat.UserRole, Content: "How are you?"},
		{Role: chat.AssistantRole, Content: "I'm doing well"},
	}

	// We can't easily test the actual prompt sent to the LLM without a more complex mock,
	// but we can verify the conversation building logic by checking SimpleSummarizer
	simple := NewSimpleSummarizer(10, 10)
	summary, err := simple.Summarize(context.Background(), records)
	assert.NoError(t, err)

	// Should include all messages in order
	lines := strings.Split(summary, "\n")
	var foundMessages []string
	for _, line := range lines {
		if strings.HasPrefix(line, "user:") || strings.HasPrefix(line, "assistant:") {
			foundMessages = append(foundMessages, line)
		}
	}

	assert.Equal(t, 4, len(foundMessages))
	assert.Contains(t, foundMessages[0], "Hello")
	assert.Contains(t, foundMessages[1], "Hi there")
	assert.Contains(t, foundMessages[2], "How are you")
	assert.Contains(t, foundMessages[3], "doing well")
}