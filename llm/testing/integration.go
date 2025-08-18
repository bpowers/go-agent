package testing

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/bpowers/go-agent/chat"
)

// testToolDef is a simple implementation of chat.ToolDef for testing
type testToolDef struct {
	name        string
	description string
	jsonSchema  string
}

func (t *testToolDef) MCPJsonSchema() string {
	return t.jsonSchema
}

func (t *testToolDef) Name() string {
	return t.name
}

func (t *testToolDef) Description() string {
	return t.description
}

// IntegrationConfig holds configuration for integration tests
type IntegrationConfig struct {
	Provider    string // "openai", "claude", "gemini"
	APIKey      string // from environment
	Model       string // defaults to cheapest model
	SkipIfNoKey bool   // skip test if API key missing
}

// SkipIfNoAPIKey skips the test if the API key for the provider is not set
func SkipIfNoAPIKey(t testing.TB, provider string) {
	var envVar string
	switch provider {
	case "openai":
		envVar = "OPENAI_API_KEY"
	case "claude":
		envVar = "ANTHROPIC_API_KEY"
	case "gemini":
		envVar = "GEMINI_API_KEY"
	default:
		t.Fatalf("unknown provider: %s", provider)
	}

	if os.Getenv(envVar) == "" {
		t.Skipf("Skipping test: %q not set in environment", envVar)
	}
}

// TestStreamingResponse tests streaming functionality with system dynamics prompt
func TestStreamingResponse(t testing.TB, client chat.Client) {
	prompt := "generate a 4 paragraph description of the system dynamics modeling methodology as it applies to strategy"

	chatSession := client.NewChat("You are a helpful assistant that explains complex concepts clearly.", nil...)

	// Test non-streaming first
	response, err := chatSession.Message(context.Background(), chat.Message{
		Role:    chat.UserRole,
		Content: prompt,
	})
	if err != nil {
		t.Fatalf("Failed to get response: %v", err)
	}

	// Validate response content
	if response.Content == "" {
		t.Error("Expected non-empty response content")
	}

	// Check for system dynamics keywords
	content := strings.ToLower(response.Content)
	expectedKeywords := []string{"system", "dynamics", "feedback", "model"}
	missingKeywords := []string{}
	for _, keyword := range expectedKeywords {
		if !strings.Contains(content, keyword) {
			missingKeywords = append(missingKeywords, keyword)
		}
	}
	if len(missingKeywords) > 0 {
		t.Errorf("Response missing expected keywords: %v", missingKeywords)
	}

	// Count paragraphs (rough approximation)
	paragraphs := strings.Split(response.Content, "\n\n")
	if len(paragraphs) < 3 {
		t.Errorf("Expected at least 3 paragraphs, got %d", len(paragraphs))
	}

	// Test streaming
	var streamedContent strings.Builder
	var chunkCount int

	streamResponse, err := chatSession.Message(
		context.Background(),
		chat.Message{
			Role:    chat.UserRole,
			Content: "Now explain reinforcement loops in one paragraph.",
		},
		chat.WithStreamingCb(func(event chat.StreamEvent) error {
			if event.Type == chat.StreamEventTypeContent {
				streamedContent.WriteString(event.Content)
				chunkCount++
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("Failed to get streaming response: %v", err)
	}

	// Validate streaming produced content
	if streamedContent.String() == "" {
		t.Error("Expected non-empty streamed content")
	}

	// Validate final response matches streamed content
	if streamResponse.Content != streamedContent.String() {
		t.Error("Final response doesn't match streamed content")
	}

	// Validate we got multiple chunks (streaming actually happened)
	if chunkCount < 2 {
		t.Errorf("Expected multiple stream chunks, got %d", chunkCount)
	}

	t.Logf("Streaming test passed: received %d chunks", chunkCount)
}

// TestStreaming is an alias for TestStreamingResponse for consistency
func TestStreaming(t testing.TB, client chat.Client) {
	TestStreamingResponse(t, client)
}

// TestTokenUsageCumulative tests that TokenUsage returns cumulative token counts across multiple messages
func TestTokenUsageCumulative(t testing.TB, client chat.Client) {
	chatSession := client.NewChat("You are a helpful assistant. Be concise.")

	// First message: request a long response (4 paragraphs)
	_, err := chatSession.Message(context.Background(), chat.Message{
		Role:    chat.UserRole,
		Content: "Write exactly 4 paragraphs explaining the water cycle. Each paragraph should be at least 3 sentences long.",
	})
	if err != nil {
		t.Fatalf("Failed to get first response: %v", err)
	}

	// Get token usage after first message
	usage1, err := chatSession.TokenUsage()
	if err != nil {
		t.Fatalf("Failed to get token usage after first message: %v", err)
	}

	// Validate first usage has tokens - both LastMessage and Cumulative
	if usage1.LastMessage.InputTokens <= 0 {
		t.Errorf("Expected positive LastMessage input tokens after first message, got %d", usage1.LastMessage.InputTokens)
	}
	if usage1.LastMessage.OutputTokens <= 0 {
		t.Errorf("Expected positive LastMessage output tokens after first message, got %d", usage1.LastMessage.OutputTokens)
	}
	if usage1.LastMessage.TotalTokens <= 0 {
		t.Errorf("Expected positive LastMessage total tokens after first message, got %d", usage1.LastMessage.TotalTokens)
	}

	if usage1.Cumulative.InputTokens <= 0 {
		t.Errorf("Expected positive cumulative input tokens after first message, got %d", usage1.Cumulative.InputTokens)
	}
	if usage1.Cumulative.OutputTokens <= 0 {
		t.Errorf("Expected positive cumulative output tokens after first message, got %d", usage1.Cumulative.OutputTokens)
	}
	if usage1.Cumulative.TotalTokens <= 0 {
		t.Errorf("Expected positive cumulative total tokens after first message, got %d", usage1.Cumulative.TotalTokens)
	}

	t.Logf("First message LastMessage usage - Input: %d, Output: %d, Total: %d",
		usage1.LastMessage.InputTokens, usage1.LastMessage.OutputTokens, usage1.LastMessage.TotalTokens)
	t.Logf("First message Cumulative usage - Input: %d, Output: %d, Total: %d",
		usage1.Cumulative.InputTokens, usage1.Cumulative.OutputTokens, usage1.Cumulative.TotalTokens)

	// Second message: request a very short response (1 word)
	_, err = chatSession.Message(context.Background(), chat.Message{
		Role:    chat.UserRole,
		Content: "Reply with exactly one word: yes or no. Is water important?",
	})
	if err != nil {
		t.Fatalf("Failed to get second response: %v", err)
	}

	// Get token usage after second message
	usage2, err := chatSession.TokenUsage()
	if err != nil {
		t.Fatalf("Failed to get token usage after second message: %v", err)
	}

	// Validate second message LastMessage tokens
	if usage2.LastMessage.InputTokens <= 0 {
		t.Errorf("Expected positive LastMessage input tokens after second message, got %d", usage2.LastMessage.InputTokens)
	}
	if usage2.LastMessage.OutputTokens <= 0 {
		t.Errorf("Expected positive LastMessage output tokens after second message, got %d", usage2.LastMessage.OutputTokens)
	}
	if usage2.LastMessage.TotalTokens <= 0 {
		t.Errorf("Expected positive LastMessage total tokens after second message, got %d", usage2.LastMessage.TotalTokens)
	}

	// The second response should be much shorter than the first (it's 1 word vs 4 paragraphs)
	if usage2.LastMessage.OutputTokens >= usage1.LastMessage.OutputTokens {
		t.Errorf("Expected second message output tokens (%d) to be less than first message (%d) since it's just 1 word",
			usage2.LastMessage.OutputTokens, usage1.LastMessage.OutputTokens)
	}

	t.Logf("Second message LastMessage usage - Input: %d, Output: %d, Total: %d",
		usage2.LastMessage.InputTokens, usage2.LastMessage.OutputTokens, usage2.LastMessage.TotalTokens)
	t.Logf("Second message Cumulative usage - Input: %d, Output: %d, Total: %d",
		usage2.Cumulative.InputTokens, usage2.Cumulative.OutputTokens, usage2.Cumulative.TotalTokens)

	// Check for monotonic behavior - usage should be cumulative
	// Input tokens should increase (includes conversation history)
	if usage2.Cumulative.InputTokens <= usage1.Cumulative.InputTokens {
		t.Errorf("Expected input tokens to increase (cumulative): first=%d, second=%d",
			usage1.Cumulative.InputTokens, usage2.Cumulative.InputTokens)
	}

	// Output tokens should increase (cumulative across responses)
	if usage2.Cumulative.OutputTokens <= usage1.Cumulative.OutputTokens {
		t.Errorf("Expected output tokens to increase (cumulative): first=%d, second=%d",
			usage1.Cumulative.OutputTokens, usage2.Cumulative.OutputTokens)
	}

	// Total tokens should increase
	if usage2.Cumulative.TotalTokens <= usage1.Cumulative.TotalTokens {
		t.Errorf("Expected total tokens to increase (cumulative): first=%d, second=%d",
			usage1.Cumulative.TotalTokens, usage2.Cumulative.TotalTokens)
	}

	// The second response should be much shorter than the first
	// But if usage is cumulative, the difference in output tokens should be small relative to the first response
	outputDiff := usage2.Cumulative.OutputTokens - usage1.Cumulative.OutputTokens
	if outputDiff > usage1.Cumulative.OutputTokens/4 {
		t.Logf("Warning: second response added %d output tokens, which seems high for a 1-word response", outputDiff)
	}
}

// TestToolCallStreamEvents tests that tool call events are emitted during streaming
func TestToolCallStreamEvents(t testing.TB, client chat.Client) {
	chatSession := client.NewChat("You are a helpful assistant with access to tools.")

	// Register a simple echo tool
	toolDef := &testToolDef{
		name:        "echo",
		description: "Echo back the provided message",
		jsonSchema: `{
			"name": "echo",
			"description": "Echo back the provided message",
			"inputSchema": {
				"type": "object",
				"properties": {
					"message": {
						"type": "string",
						"description": "The message to echo back"
					}
				},
				"required": ["message"]
			}
		}`,
	}

	toolCalled := false
	err := chatSession.RegisterTool(toolDef, func(ctx context.Context, input string) string {
		toolCalled = true
		return `{"result": "Echo successful"}`
	})
	if err != nil {
		t.Fatalf("Failed to register tool: %v", err)
	}

	// Track events during streaming
	var toolCallEvents []chat.StreamEvent
	var contentEvents []chat.StreamEvent
	streamedContent := strings.Builder{}

	_, err = chatSession.Message(
		context.Background(),
		chat.Message{
			Role:    chat.UserRole,
			Content: "Please use the echo tool to echo the message 'Hello World'",
		},
		chat.WithStreamingCb(func(event chat.StreamEvent) error {
			switch event.Type {
			case chat.StreamEventTypeToolCall:
				toolCallEvents = append(toolCallEvents, event)
				t.Logf("Tool call event: %+v", event)
			case chat.StreamEventTypeContent:
				contentEvents = append(contentEvents, event)
				streamedContent.WriteString(event.Content)
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("Failed to get streaming response: %v", err)
	}

	// Verify tool was actually called
	if !toolCalled {
		t.Error("Expected tool to be called, but it wasn't")
	}

	// Verify we received at least one tool call event
	if len(toolCallEvents) == 0 {
		t.Error("Expected at least one tool call event during streaming")
	}

	// Verify tool call events contain proper information
	for _, event := range toolCallEvents {
		if len(event.ToolCalls) == 0 {
			t.Error("Tool call event has no ToolCalls")
		}
		for _, tc := range event.ToolCalls {
			if tc.Name == "" {
				t.Error("Tool call has no name")
			}
			t.Logf("Tool invoked: %s with args: %s", tc.Name, string(tc.Arguments))
		}
	}

	// Verify we still got content events
	if len(contentEvents) == 0 {
		t.Error("Expected content events in addition to tool call events")
	}

	// Verify final content is not empty
	if streamedContent.String() == "" {
		t.Error("Expected non-empty streamed content")
	}

	t.Logf("Tool call streaming test passed: received %d tool call events and %d content events",
		len(toolCallEvents), len(contentEvents))
}
