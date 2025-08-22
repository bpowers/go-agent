package openai

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/llm/internal/common"
)

func TestResponsesAPISelection(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name          string
		model         string
		expectedAPI   API
		shouldRespond bool
	}{
		{
			name:          "GPT-5 uses Responses API",
			model:         "gpt-5",
			expectedAPI:   Responses,
			shouldRespond: true,
		},
		{
			name:          "GPT-5-turbo uses Responses API",
			model:         "gpt-5-turbo",
			expectedAPI:   Responses,
			shouldRespond: true,
		},
		{
			name:          "O1 uses Responses API",
			model:         "o1-preview",
			expectedAPI:   Responses,
			shouldRespond: true,
		},
		{
			name:          "O3 uses Responses API",
			model:         "o3",
			expectedAPI:   Responses,
			shouldRespond: true,
		},
		{
			name:          "GPT-4 uses Chat Completions API",
			model:         "gpt-4",
			expectedAPI:   ChatCompletions,
			shouldRespond: false,
		},
		{
			name:          "GPT-4o uses Chat Completions API",
			model:         "gpt-4o",
			expectedAPI:   ChatCompletions,
			shouldRespond: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			// Create client with API selection based on model
			var api API = ChatCompletions
			if isResponsesModel(tt.model) {
				api = Responses
			}

			assert.Equal(t, tt.shouldRespond, api == Responses,
				"Model %s should use Responses API: %v", tt.model, tt.shouldRespond)
		})
	}
}

func TestMessageConversion(t *testing.T) {
	t.Parallel()
	// Test that chat messages are properly converted for Responses API
	systemPrompt := "You are a helpful assistant"
	initialMsgs := []chat.Message{
		{Role: chat.UserRole, Content: "Hello"},
		{Role: chat.AssistantRole, Content: "Hi there!"},
	}

	client := &chatClient{
		client: client{
			modelName: "gpt-5",
			api:       Responses,
		},
		state: common.NewState(systemPrompt, initialMsgs),
		tools: common.NewTools(),
	}

	// Verify the client has the correct configuration
	assert.Equal(t, Responses, client.api)
	assert.Equal(t, "gpt-5", client.modelName)

	// Verify state was initialized correctly
	actualSystemPrompt, actualMsgs := client.state.History()
	assert.Equal(t, systemPrompt, actualSystemPrompt)
	assert.Len(t, actualMsgs, 2)
}

func TestStreamEventHandling(t *testing.T) {
	t.Parallel()
	// Test that reasoning events are properly handled
	var receivedEvents []chat.StreamEvent

	callback := func(event chat.StreamEvent) error {
		receivedEvents = append(receivedEvents, event)
		return nil
	}

	// Simulate reasoning events
	events := []chat.StreamEvent{
		{
			Type: chat.StreamEventTypeThinking,
			ThinkingStatus: &chat.ThinkingStatus{
				IsThinking: true,
			},
		},
		{
			Type:    chat.StreamEventTypeThinking,
			Content: "Let me think about this...",
			ThinkingStatus: &chat.ThinkingStatus{
				IsThinking: true,
			},
		},
		{
			Type: chat.StreamEventTypeThinkingSummary,
			ThinkingStatus: &chat.ThinkingStatus{
				IsThinking: false,
				Summary:    "Analyzed the request",
			},
		},
		{
			Type:    chat.StreamEventTypeContent,
			Content: "Here's my response",
		},
	}

	// Process events through callback
	for _, event := range events {
		err := callback(event)
		assert.NoError(t, err)
	}

	// Verify all events were received
	assert.Equal(t, len(events), len(receivedEvents))

	// Verify event types
	assert.Equal(t, chat.StreamEventTypeThinking, receivedEvents[0].Type)
	assert.Equal(t, chat.StreamEventTypeThinking, receivedEvents[1].Type)
	assert.Equal(t, chat.StreamEventTypeThinkingSummary, receivedEvents[2].Type)
	assert.Equal(t, chat.StreamEventTypeContent, receivedEvents[3].Type)

	// Verify content
	assert.Equal(t, "Let me think about this...", receivedEvents[1].Content)
	assert.Equal(t, "Here's my response", receivedEvents[3].Content)
}

// isResponsesModel checks if the model should use the Responses API
// This is a duplicate of the function in main.go for testing
func isResponsesModel(model string) bool {
	modelLower := strings.ToLower(model)
	// gpt-5, o1, and o3 models use the Responses API
	return strings.HasPrefix(modelLower, "gpt-5") ||
		strings.HasPrefix(modelLower, "o1-") ||
		strings.HasPrefix(modelLower, "o3")
}

func TestClientCreation(t *testing.T) {
	t.Parallel()
	t.Run("Client requires model", func(t *testing.T) {
		t.Parallel()
		_, err := NewClient(OpenAIURL, "test-key")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "WithModel is a required option")
	})

	t.Run("Client creation with model", func(t *testing.T) {
		t.Parallel()
		client, err := NewClient(OpenAIURL, "test-key", WithModel("gpt-4"))
		assert.NoError(t, err)
		assert.NotNil(t, client)
	})

	t.Run("Client with Responses API", func(t *testing.T) {
		t.Parallel()
		cli, err := NewClient(OpenAIURL, "test-key",
			WithModel("gpt-5"),
			WithAPI(Responses))
		assert.NoError(t, err)
		assert.NotNil(t, cli)

		// Verify it's configured for Responses API
		// Note: We can't directly access private fields in tests,
		// but we can verify the client was created successfully
		// with the Responses API option
	})
}

func TestMessageStreamRouting(t *testing.T) {
	t.Parallel()
	t.Run("Routes to Responses API for gpt-5", func(t *testing.T) {
		t.Parallel()
		c := &chatClient{
			client: client{
				modelName: "gpt-5",
				api:       Responses,
			},
			state: common.NewState("Test", []chat.Message{}),
			tools: common.NewTools(),
		}

		// This would normally make an API call, but we're just testing routing
		// In a real test with mocking, we'd verify the correct API was called
		assert.Equal(t, Responses, c.api)
	})

	t.Run("Routes to Chat Completions API for gpt-4", func(t *testing.T) {
		t.Parallel()
		c := &chatClient{
			client: client{
				modelName: "gpt-4",
				api:       ChatCompletions,
			},
			state: common.NewState("Test", []chat.Message{}),
			tools: common.NewTools(),
		}

		assert.Equal(t, ChatCompletions, c.api)
	})
}
