package chat

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStreamEventTypes(t *testing.T) {
	t.Parallel()
	// Test that all event types are properly defined
	assert.Equal(t, StreamEventType("content"), StreamEventTypeContent)
	assert.Equal(t, StreamEventType("thinking"), StreamEventTypeThinking)
	assert.Equal(t, StreamEventType("thinking_summary"), StreamEventTypeThinkingSummary)
	assert.Equal(t, StreamEventType("done"), StreamEventTypeDone)
}

func TestStreamEvent(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		event    StreamEvent
		validate func(t *testing.T, e StreamEvent)
	}{
		{
			name: "Content event with text",
			event: StreamEvent{
				Type:    StreamEventTypeContent,
				Content: "Hello, world!",
			},
			validate: func(t *testing.T, e StreamEvent) {
				assert.Equal(t, StreamEventTypeContent, e.Type)
				assert.Equal(t, "Hello, world!", e.Content)
				assert.Nil(t, e.ThinkingStatus)
			},
		},
		{
			name: "Thinking event with status",
			event: StreamEvent{
				Type:           StreamEventTypeThinking,
				ThinkingStatus: &ThinkingStatus{},
			},
			validate: func(t *testing.T, e StreamEvent) {
				assert.Equal(t, StreamEventTypeThinking, e.Type)
				assert.NotNil(t, e.ThinkingStatus)
			},
		},
		{
			name: "Thinking event with content",
			event: StreamEvent{
				Type:           StreamEventTypeThinking,
				Content:        "Processing request...",
				ThinkingStatus: &ThinkingStatus{},
			},
			validate: func(t *testing.T, e StreamEvent) {
				assert.Equal(t, StreamEventTypeThinking, e.Type)
				assert.Equal(t, "Processing request...", e.Content)
				assert.NotNil(t, e.ThinkingStatus)
			},
		},
		{
			name: "Thinking summary event",
			event: StreamEvent{
				Type: StreamEventTypeThinkingSummary,
				ThinkingStatus: &ThinkingStatus{
					Summary: "Analyzed the user's request for help",
				},
			},
			validate: func(t *testing.T, e StreamEvent) {
				assert.Equal(t, StreamEventTypeThinkingSummary, e.Type)
				assert.NotNil(t, e.ThinkingStatus)
				assert.Equal(t, "Analyzed the user's request for help", e.ThinkingStatus.Summary)
			},
		},
		{
			name: "Done event",
			event: StreamEvent{
				Type:         StreamEventTypeDone,
				FinishReason: "stop",
			},
			validate: func(t *testing.T, e StreamEvent) {
				assert.Equal(t, StreamEventTypeDone, e.Type)
				assert.Equal(t, "stop", e.FinishReason)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			tt.validate(t, tt.event)
		})
	}
}

func TestStreamCallback(t *testing.T) {
	t.Parallel()
	t.Run("Successful callback", func(t *testing.T) {
		t.Parallel()
		var receivedEvents []StreamEvent
		callback := func(event StreamEvent) error {
			receivedEvents = append(receivedEvents, event)
			return nil
		}

		events := []StreamEvent{
			{Type: StreamEventTypeThinking, ThinkingStatus: &ThinkingStatus{}},
			{Type: StreamEventTypeContent, Content: "Hello"},
			{Type: StreamEventTypeDone},
		}

		for _, event := range events {
			err := callback(event)
			assert.NoError(t, err)
		}

		assert.Equal(t, len(events), len(receivedEvents))
		for i, event := range events {
			assert.Equal(t, event.Type, receivedEvents[i].Type)
		}
	})

	t.Run("Callback returns error", func(t *testing.T) {
		t.Parallel()
		expectedErr := errors.New("user cancelled")
		callback := func(event StreamEvent) error {
			if event.Type == StreamEventTypeContent {
				return expectedErr
			}
			return nil
		}

		err := callback(StreamEvent{Type: StreamEventTypeThinking})
		assert.NoError(t, err)

		err = callback(StreamEvent{Type: StreamEventTypeContent, Content: "test"})
		assert.Equal(t, expectedErr, err)
	})
}

func TestMessage(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		message Message
		role    Role
		content string
	}{
		{
			name:    "User message",
			message: UserMessage("What is the weather?"),
			role:    UserRole,
			content: "What is the weather?",
		},
		{
			name:    "Assistant message",
			message: AssistantMessage("I can help you with that."),
			role:    AssistantRole,
			content: "I can help you with that.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.role, tt.message.Role)
			assert.Equal(t, tt.content, tt.message.GetText())
		})
	}
}

func TestOptions(t *testing.T) {
	t.Parallel()
	t.Run("WithTemperature", func(t *testing.T) {
		t.Parallel()
		opts := ApplyOptions(WithTemperature(0.8))
		assert.NotNil(t, opts.Temperature)
		assert.Equal(t, 0.8, *opts.Temperature)
	})

	t.Run("WithMaxTokens", func(t *testing.T) {
		t.Parallel()
		opts := ApplyOptions(WithMaxTokens(2048))
		assert.Equal(t, 2048, opts.MaxTokens)
	})

	t.Run("WithReasoningEffort", func(t *testing.T) {
		t.Parallel()
		opts := ApplyOptions(WithReasoningEffort("high"))
		assert.Equal(t, "high", opts.ReasoningEffort)
	})

	t.Run("Multiple options", func(t *testing.T) {
		t.Parallel()
		opts := ApplyOptions(
			WithTemperature(0.5),
			WithMaxTokens(1024),
			WithReasoningEffort("medium"),
		)
		assert.NotNil(t, opts.Temperature)
		assert.Equal(t, 0.5, *opts.Temperature)
		assert.Equal(t, 1024, opts.MaxTokens)
		assert.Equal(t, "medium", opts.ReasoningEffort)
	})

	t.Run("No options", func(t *testing.T) {
		t.Parallel()
		opts := ApplyOptions()
		assert.Nil(t, opts.Temperature)
		assert.Equal(t, 0, opts.MaxTokens)
		assert.Equal(t, "", opts.ReasoningEffort)
	})
}

func TestDebugDir(t *testing.T) {
	t.Parallel()

	t.Run("No debug dir", func(t *testing.T) {
		t.Parallel()
		ctx := context.Background()
		// This will panic if we try to assert on it directly
		// So we use a recover pattern
		defer func() {
			if r := recover(); r != nil {
				// Expected - no debug dir was set
			}
		}()
		_ = DebugDir(ctx)
		t.Error("Expected panic when no debug dir is set")
	})

	t.Run("With debug dir", func(t *testing.T) {
		t.Parallel()
		ctx := WithDebugDir(context.Background(), "/tmp/debug")
		dir := DebugDir(ctx)
		assert.Equal(t, "/tmp/debug", dir)
	})
}

// MockChat implements the Chat interface for testing
type MockChat struct {
	systemPrompt   string
	messages       []Message
	nextResponse   Message
	streamCallback StreamCallback
	err            error
}

func (m *MockChat) Message(ctx context.Context, msg Message, opts ...Option) (Message, error) {
	if m.err != nil {
		return Message{}, m.err
	}
	m.messages = append(m.messages, msg)

	// Apply options to get callback if provided
	appliedOpts := ApplyOptions(opts...)
	callback := appliedOpts.StreamingCb
	m.streamCallback = callback

	// Simulate streaming if callback provided
	if callback != nil {
		// Simulate thinking
		err := callback(StreamEvent{
			Type:           StreamEventTypeThinking,
			ThinkingStatus: &ThinkingStatus{},
		})
		if err != nil {
			return Message{}, err
		}

		// Simulate content
		err = callback(StreamEvent{
			Type:    StreamEventTypeContent,
			Content: m.nextResponse.GetText(),
		})
		if err != nil {
			return Message{}, err
		}
	}

	return m.nextResponse, nil
}

func (m *MockChat) History() (systemPrompt string, msgs []Message) {
	return m.systemPrompt, m.messages
}

// TokenUsage returns mock token usage
func (m *MockChat) TokenUsage() (TokenUsage, error) {
	return TokenUsage{}, nil
}

// MaxTokens returns mock max tokens
func (m *MockChat) MaxTokens() int {
	return 4096
}

// RegisterTool registers a mock tool
func (m *MockChat) RegisterTool(def ToolDef, fn func(context.Context, string) string) error {
	return nil
}

// DeregisterTool removes a mock tool
func (m *MockChat) DeregisterTool(name string) {
	// No-op for mock
}

// ListTools returns mock tool list
func (m *MockChat) ListTools() []string {
	return []string{}
}

func TestChatInterface(t *testing.T) {
	t.Parallel()

	t.Run("Message method", func(t *testing.T) {
		t.Parallel()
		mock := &MockChat{
			systemPrompt: "You are a helpful assistant",
			nextResponse: AssistantMessage("I can help with that!"),
		}

		ctx := context.Background()
		userMsg := UserMessage("Hello")

		resp, err := mock.Message(ctx, userMsg)
		assert.NoError(t, err)
		assert.Equal(t, AssistantRole, resp.Role)
		assert.Equal(t, "I can help with that!", resp.GetText())
		assert.Len(t, mock.messages, 1)
	})

	t.Run("Message with streaming callback", func(t *testing.T) {
		t.Parallel()
		mock := &MockChat{
			systemPrompt: "You are a helpful assistant",
			nextResponse: AssistantMessage("I can help with that!"),
		}

		ctx := context.Background()
		userMsg := UserMessage("Hello")

		var receivedEvents []StreamEvent
		callback := func(event StreamEvent) error {
			receivedEvents = append(receivedEvents, event)
			return nil
		}

		resp, err := mock.Message(ctx, userMsg, WithStreamingCb(callback))
		assert.NoError(t, err)
		assert.Equal(t, AssistantRole, resp.Role)
		assert.Equal(t, "I can help with that!", resp.GetText())
		assert.Len(t, mock.messages, 1)
		assert.Len(t, receivedEvents, 2) // Thinking + Content
	})

	t.Run("History method", func(t *testing.T) {
		t.Parallel()
		mock := &MockChat{
			systemPrompt: "You are a helpful assistant",
			messages: []Message{
				UserMessage("Hello"),
			},
		}

		system, msgs := mock.History()
		assert.Equal(t, "You are a helpful assistant", system)
		assert.Len(t, msgs, 1)
	})
}
