package testing

import (
	"context"
	"encoding/json"
	"fmt"
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

// TestToolCallAndResultStreamEvents tests that both tool call and tool result events are emitted during streaming
func TestToolCallAndResultStreamEvents(t testing.TB, client chat.Client) {
	chatSession := client.NewChat("You are a helpful assistant with access to tools.")

	// Register a simple echo tool that returns a specific result
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
		// Parse the input to get the message
		var args struct {
			Message string `json:"message"`
		}
		if err := json.Unmarshal([]byte(input), &args); err != nil {
			return fmt.Sprintf(`{"error": "Failed to parse input: %v"}`, err)
		}
		return fmt.Sprintf("Echo: %s", args.Message)
	})
	if err != nil {
		t.Fatalf("Failed to register tool: %v", err)
	}

	// Track events during streaming
	var toolCallEvents []chat.StreamEvent
	var toolResultEvents []chat.StreamEvent
	var contentEvents []chat.StreamEvent
	toolCallIDs := make(map[string]bool)

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
				// Track tool call IDs
				for _, tc := range event.ToolCalls {
					if tc.ID != "" {
						toolCallIDs[tc.ID] = true
					}
				}
				t.Logf("Tool call event: %+v", event)
			case chat.StreamEventTypeToolResult:
				toolResultEvents = append(toolResultEvents, event)
				t.Logf("Tool result event: %+v", event)
			case chat.StreamEventTypeContent:
				contentEvents = append(contentEvents, event)
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

	// Verify we received at least one tool result event
	if len(toolResultEvents) == 0 {
		t.Error("Expected at least one tool result event during streaming")
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
			if tc.ID == "" {
				t.Error("Tool call has no ID")
			}
			t.Logf("Tool invoked: %s (ID: %s) with args: %s", tc.Name, tc.ID, string(tc.Arguments))
		}
	}

	// Verify tool result events match the tool calls
	for _, event := range toolResultEvents {
		if len(event.ToolResults) == 0 {
			t.Error("Tool result event has no ToolResults")
		}
		for _, tr := range event.ToolResults {
			if tr.ToolCallID == "" {
				t.Error("Tool result has no ToolCallID")
			}
			if !toolCallIDs[tr.ToolCallID] {
				t.Errorf("Tool result ID %s doesn't match any tool call", tr.ToolCallID)
			}
			resultStr := tr.Content
			if tr.Error != "" {
				resultStr = tr.Error
			}
			t.Logf("Tool result for ID %s: %s", tr.ToolCallID, resultStr)

			if tr.Error == "" && !strings.Contains(tr.Content, "Echo:") {
				t.Errorf("Tool result doesn't contain expected output: %s", tr.Content)
			}
		}
	}

	// Verify we still got content events
	if len(contentEvents) == 0 {
		t.Error("Expected content events in addition to tool events")
	}

	// Verify the number of tool calls matches tool results
	if len(toolCallEvents) != len(toolResultEvents) {
		t.Errorf("Mismatch: %d tool call events but %d tool result events",
			len(toolCallEvents), len(toolResultEvents))
	}

	t.Logf("Tool call and result streaming test passed: received %d tool call events, %d tool result events, and %d content events",
		len(toolCallEvents), len(toolResultEvents), len(contentEvents))
}

// BaseURLValidator is an interface that clients can implement to expose their base URL for testing
type BaseURLValidator interface {
	BaseURL() string
}

// TestBaseURLConfiguration validates that BaseURL is properly configured in the client
func TestBaseURLConfiguration(t testing.TB, client chat.Client, expectedURL string) {
	// Import note: caller should import testify packages
	// We use testing.TB interface here for flexibility

	// Try to cast to BaseURLValidator interface
	validator, ok := client.(BaseURLValidator)
	if !ok {
		t.Fatal("Client must implement BaseURLValidator interface")
	}

	actualURL := validator.BaseURL()
	if actualURL != expectedURL {
		t.Errorf("BaseURL mismatch: expected %q, got %q", expectedURL, actualURL)
	}
}

// HeadersValidator is an interface that clients can implement to expose their custom headers for testing
type HeadersValidator interface {
	Headers() map[string]string
}

// TestHeaderConfiguration validates that custom headers are properly configured in the client
func TestHeaderConfiguration(t testing.TB, client chat.Client, expectedHeaders map[string]string) {
	// Import note: caller should import testify packages
	// We use testing.TB interface here for flexibility

	// Try to cast to HeadersValidator interface
	validator, ok := client.(HeadersValidator)
	if !ok {
		t.Fatal("Client must implement HeadersValidator interface")
	}

	actualHeaders := validator.Headers()

	// Check that all expected headers are present with correct values
	for key, expectedValue := range expectedHeaders {
		actualValue, ok := actualHeaders[key]
		if !ok {
			t.Errorf("Missing header %q", key)
			continue
		}
		if actualValue != expectedValue {
			t.Errorf("Header %q mismatch: expected %q, got %q", key, expectedValue, actualValue)
		}
	}

	// Check for unexpected headers
	for key := range actualHeaders {
		if _, expected := expectedHeaders[key]; !expected {
			t.Errorf("Unexpected header %q with value %q", key, actualHeaders[key])
		}
	}
}

// TestEmptyToolResultsHandling tests that clients properly handle empty tool results without causing API errors
func TestEmptyToolResultsHandling(t testing.TB, client chat.Client) {
	chatSession := client.NewChat("You are a helpful assistant with access to tools.")

	// Register a tool that simulates returning empty results
	emptyResultTool := &testToolDef{
		name:        "empty_result_tool",
		description: "A tool that returns an empty result",
		jsonSchema: `{
			"name": "empty_result_tool",
			"description": "A tool that returns an empty result",
			"inputSchema": {
				"type": "object",
				"properties": {
					"action": {
						"type": "string",
						"description": "The action to perform"
					}
				},
				"required": ["action"]
			}
		}`,
	}

	// Track if the tool was called
	toolCalled := false
	err := chatSession.RegisterTool(emptyResultTool, func(ctx context.Context, input string) string {
		toolCalled = true
		// Simulate an empty result scenario
		return ""
	})
	if err != nil {
		t.Fatalf("Failed to register empty result tool: %v", err)
	}

	// Test with a message that should trigger tool use
	response, err := chatSession.Message(
		context.Background(),
		chat.Message{
			Role:    chat.UserRole,
			Content: "Please use the empty_result_tool to perform a test action.",
		},
	)
	// The request should succeed even with empty tool results
	if err != nil {
		t.Fatalf("Client failed to handle empty tool results properly: %v", err)
	}

	// Verify we got a response
	if response.Content == "" {
		t.Error("Expected non-empty response content")
	}

	// Verify the tool was called
	if !toolCalled {
		t.Error("Expected tool to be called, but it wasn't")
	}

	t.Logf("Non-streaming test passed, tool was called: %v", toolCalled)

	// Create a new session for streaming test to avoid history complications
	streamingSession := client.NewChat("You are a helpful assistant with access to tools.")

	// Register the same tool in the new session
	err = streamingSession.RegisterTool(emptyResultTool, func(ctx context.Context, input string) string {
		toolCalled = true
		// Simulate an empty result scenario
		return ""
	})
	if err != nil {
		t.Fatalf("Failed to register empty result tool in streaming session: %v", err)
	}

	// Test with streaming to ensure it also handles empty results properly
	toolCalled = false
	var streamedContent strings.Builder
	streamResponse, err := streamingSession.Message(
		context.Background(),
		chat.Message{
			Role:    chat.UserRole,
			Content: "Please use the empty_result_tool to perform a test action.",
		},
		chat.WithStreamingCb(func(event chat.StreamEvent) error {
			if event.Type == chat.StreamEventTypeContent {
				streamedContent.WriteString(event.Content)
			}
			return nil
		}),
	)
	// The streaming request should also succeed with empty tool results
	if err != nil {
		t.Fatalf("Client failed to handle empty tool results properly during streaming: %v", err)
	}

	// Verify we got a response
	if streamResponse.Content == "" {
		t.Error("Expected non-empty response content from streaming")
	}

	// Verify streaming produced content
	if streamedContent.String() == "" {
		t.Error("Expected non-empty streamed content")
	}

	// Verify the tool was called again
	if !toolCalled {
		t.Error("Expected tool to be called during streaming, but it wasn't")
	}

	t.Log("Empty tool results handling test passed")
}
