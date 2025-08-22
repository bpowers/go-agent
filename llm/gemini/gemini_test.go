package gemini

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/bpowers/go-agent/chat"
)

// TestSystemPromptPlaceholder tests whether the placeholder response
// after system prompts is necessary for Gemini API.
func TestSystemPromptPlaceholder(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set")
	}

	ctx := context.Background()

	t.Run("with placeholder response", func(t *testing.T) {
		client, err := NewClient(apiKey, WithModel("gemini-2.5-flash"))
		require.NoError(t, err)

		// Create chat with system prompt
		chatClient := client.NewChat("You are a helpful assistant. Always be concise.")

		// Send a simple message
		resp, err := chatClient.Message(ctx, chat.Message{
			Role:    chat.UserRole,
			Content: "What is 2+2?",
		})

		require.NoError(t, err, "Should work with placeholder")
		assert.NotEmpty(t, resp.Content)
		assert.Contains(t, strings.ToLower(resp.Content), "4")
	})

	t.Run("without placeholder response", func(t *testing.T) {
		// Test direct API usage without our wrapper to see if placeholder is required
		clientConfig := &genai.ClientConfig{
			APIKey: apiKey,
		}

		genaiClient, err := genai.NewClient(ctx, clientConfig)
		require.NoError(t, err)

		// Build conversation WITHOUT placeholder
		contents := []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "You are a helpful assistant. Always be concise."},
				},
			},
			// No placeholder "model" response here
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "What is 2+2?"},
				},
			},
		}

		config := &genai.GenerateContentConfig{}
		resp, err := genaiClient.Models.GenerateContent(ctx, "gemini-2.5-flash", contents, config)

		// Check if this works or fails
		if err != nil {
			t.Logf("Without placeholder failed: %v", err)
			assert.Contains(t, err.Error(), "role", "Error should mention role alternation issue")
		} else {
			require.NotNil(t, resp)
			require.NotEmpty(t, resp.Candidates)
			require.NotNil(t, resp.Candidates[0].Content)
			require.NotEmpty(t, resp.Candidates[0].Content.Parts)

			responseText := resp.Candidates[0].Content.Parts[0].Text
			t.Logf("Without placeholder succeeded: %s", responseText)
			assert.Contains(t, strings.ToLower(responseText), "4")
		}
	})

	t.Run("with alternating roles", func(t *testing.T) {
		// Test with proper alternation but using system content as first user message
		clientConfig := &genai.ClientConfig{
			APIKey: apiKey,
		}

		genaiClient, err := genai.NewClient(ctx, clientConfig)
		require.NoError(t, err)

		// Build conversation WITH alternation
		contents := []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "You are a helpful assistant. Always be concise."},
				},
			},
			{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "Understood."}, // Different placeholder text
				},
			},
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "What is 2+2?"},
				},
			},
		}

		config := &genai.GenerateContentConfig{}
		resp, err := genaiClient.Models.GenerateContent(ctx, "gemini-2.5-flash", contents, config)

		require.NoError(t, err, "Should work with alternating roles")
		require.NotNil(t, resp)
		require.NotEmpty(t, resp.Candidates)
		require.NotNil(t, resp.Candidates[0].Content)
		require.NotEmpty(t, resp.Candidates[0].Content.Parts)

		responseText := resp.Candidates[0].Content.Parts[0].Text
		t.Logf("With alternation succeeded: %s", responseText)
		assert.Contains(t, strings.ToLower(responseText), "4")
	})
}
