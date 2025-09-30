package testing

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	agent "github.com/bpowers/go-agent"
	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/persistence"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	response, err := chatSession.Message(context.Background(), chat.UserMessage(prompt))
	if err != nil {
		t.Fatalf("Failed to get response: %v", err)
	}

	// Validate response content
	if response.GetText() == "" {
		t.Error("Expected non-empty response content")
	}

	// Check for system dynamics keywords
	content := strings.ToLower(response.GetText())
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
	paragraphs := strings.Split(response.GetText(), "\n\n")
	if len(paragraphs) < 3 {
		t.Errorf("Expected at least 3 paragraphs, got %d", len(paragraphs))
	}

	// Test streaming
	var streamedContent strings.Builder
	var chunkCount int

	streamResponse, err := chatSession.Message(
		context.Background(),
		chat.UserMessage("Now explain reinforcement loops in one paragraph."),
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
	if streamResponse.GetText() != streamedContent.String() {
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
	_, err := chatSession.Message(context.Background(), chat.UserMessage("Write exactly 4 paragraphs explaining the water cycle. Each paragraph should be at least 3 sentences long."))
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
	_, err = chatSession.Message(context.Background(), chat.UserMessage("Reply with exactly one word: yes or no. Is water important?"))
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
		chat.UserMessage("Please use the echo tool to echo the message 'Hello World'"),
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
		chat.UserMessage("Please use the echo tool to echo the message 'Hello World'"),
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
		chat.UserMessage("Please use the empty_result_tool to perform a test action."),
	)
	// The request should succeed even with empty tool results
	if err != nil {
		t.Fatalf("Client failed to handle empty tool results properly: %v", err)
	}

	// Verify we got a response
	if response.GetText() == "" {
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
		chat.UserMessage("Please use the empty_result_tool to perform a test action."),
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
	if streamResponse.GetText() == "" {
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

// TestSystemReminderWithToolCalls tests that system reminders are properly injected after tool execution
func TestSystemReminderWithToolCalls(t testing.TB, client chat.Client) {
	chatSession := client.NewChat("You are a helpful assistant with access to tools.")

	// Track tool execution
	toolExecutionCount := 0
	var lastToolInput string

	// Register a simple calculation tool
	calcTool := &testToolDef{
		name:        "add_numbers",
		description: "Add two numbers together",
		jsonSchema: `{
			"name": "add_numbers",
			"description": "Add two numbers together",
			"inputSchema": {
				"type": "object",
				"properties": {
					"a": {"type": "number"},
					"b": {"type": "number"}
				},
				"required": ["a", "b"]
			}
		}`,
	}

	err := chatSession.RegisterTool(calcTool, func(ctx context.Context, input string) string {
		toolExecutionCount++
		lastToolInput = input

		// Parse input and calculate result
		var params struct {
			A float64 `json:"a"`
			B float64 `json:"b"`
		}
		if err := json.Unmarshal([]byte(input), &params); err != nil {
			return fmt.Sprintf(`{"error": "Failed to parse input: %v"}`, err)
		}
		result := params.A + params.B
		return fmt.Sprintf(`{"result": %v}`, result)
	})
	if err != nil {
		t.Fatalf("Failed to register calculation tool: %v", err)
	}

	// Create context with system reminder that executes after tools
	ctx := chat.WithSystemReminder(context.Background(), func() string {
		if toolExecutionCount > 0 {
			return fmt.Sprintf("<system-reminder>Tool 'add_numbers' was executed %d times. Last input: %s. Please mention this in your response.</system-reminder>",
				toolExecutionCount, lastToolInput)
		}
		return ""
	})

	// Test with a message that should trigger tool use
	response, err := chatSession.Message(
		ctx,
		chat.UserMessage("Please add 42 and 58 for me."),
	)
	if err != nil {
		t.Fatalf("Failed to get response with system reminder: %v", err)
	}

	// Verify we got a response
	if response.GetText() == "" {
		t.Error("Expected non-empty response content")
	}

	// Verify the tool was called
	if toolExecutionCount == 0 {
		t.Error("Expected tool to be called, but it wasn't")
	}

	// The response should contain the result (100)
	if !strings.Contains(response.GetText(), "100") {
		t.Error("Expected response to contain the calculation result '100'")
	}

	t.Logf("Tool executed %d times, response length: %d chars", toolExecutionCount, len(response.GetText()))

	// Test with streaming to ensure reminders work with streaming too
	streamingSession := client.NewChat("You are a helpful assistant with access to tools.")

	// Reset counters for streaming test
	toolExecutionCount = 0
	streamToolCalled := false

	err = streamingSession.RegisterTool(calcTool, func(ctx context.Context, input string) string {
		toolExecutionCount++
		streamToolCalled = true
		lastToolInput = input

		// Parse and calculate
		var params struct {
			A float64 `json:"a"`
			B float64 `json:"b"`
		}
		if err := json.Unmarshal([]byte(input), &params); err != nil {
			return fmt.Sprintf(`{"error": "Failed to parse input: %v"}`, err)
		}
		result := params.A + params.B
		return fmt.Sprintf(`{"result": %v}`, result)
	})
	if err != nil {
		t.Fatalf("Failed to register calculation tool for streaming: %v", err)
	}

	// Create context with system reminder for streaming
	streamCtx := chat.WithSystemReminder(context.Background(), func() string {
		if streamToolCalled {
			return "<system-reminder>Streaming mode: Tool was called successfully.</system-reminder>"
		}
		return ""
	})

	var streamedContent strings.Builder
	streamCallback := func(event chat.StreamEvent) error {
		if event.Type == chat.StreamEventTypeContent && event.Content != "" {
			streamedContent.WriteString(event.Content)
		}
		return nil
	}

	// Test with streaming
	streamResponse, err := streamingSession.Message(
		streamCtx,
		chat.UserMessage("Calculate 15 plus 25 for me."),
		chat.WithStreamingCb(streamCallback),
	)
	if err != nil {
		t.Fatalf("Failed to get streaming response with system reminder: %v", err)
	}

	// Verify streaming worked
	if streamResponse.GetText() == "" {
		t.Error("Expected non-empty response content from streaming")
	}

	if !streamToolCalled {
		t.Error("Expected tool to be called during streaming")
	}

	// The response should contain the result (40)
	if !strings.Contains(streamResponse.GetText(), "40") {
		t.Error("Expected streaming response to contain the calculation result '40'")
	}

	t.Log("System reminder test passed for both regular and streaming modes")
}

// TestNoDuplicateMessages tests that messages are not duplicated in history
// This is a regression test for a bug where chat clients were adding messages
// to their state even though the messages were already in the initial history
func TestNoDuplicateMessages(t *testing.T, client chat.Client) {
	// Test 1: Simple message flow without tools
	t.Run("SimpleMessages", func(t *testing.T) {
		// Create a chat with initial messages
		initialMessages := []chat.Message{
			chat.UserMessage("Hello"),
			chat.AssistantMessage("Hi there! How can I help you today?"),
		}

		chatSession := client.NewChat("You are a helpful assistant.", initialMessages...)

		// Send a new message
		userMsg := "What is 2+2?"
		response, err := chatSession.Message(context.Background(), chat.UserMessage(userMsg))
		require.NoError(t, err, "Failed to get response")
		require.NotEmpty(t, response.GetText(), "Expected non-empty response")

		// Get the history
		_, history := chatSession.History()

		// Count occurrences of each message
		helloCount := 0
		hiThereCount := 0
		mathQuestionCount := 0

		for _, msg := range history {
			content := msg.GetText()
			if content == "Hello" {
				helloCount++
			}
			if strings.Contains(content, "Hi there! How can I help you today?") {
				hiThereCount++
			}
			if content == userMsg {
				mathQuestionCount++
			}
		}

		// Each message should appear exactly once
		assert.Equal(t, 1, helloCount, "Initial user message 'Hello' should appear exactly once")
		assert.Equal(t, 1, hiThereCount, "Initial assistant message should appear exactly once")
		assert.Equal(t, 1, mathQuestionCount, "New user message should appear exactly once")

		// Total history should be: 2 initial + 1 new user + 1 assistant response = 4
		assert.Equal(t, 4, len(history), "History should contain exactly 4 messages")
	})

	// Test 2: Tool call scenario
	t.Run("WithToolCalls", func(t *testing.T) {
		// Create a chat with initial messages
		initialMessages := []chat.Message{
			chat.UserMessage("Hi, I need help with calculations"),
			chat.AssistantMessage("I can help you with calculations. What would you like me to compute?"),
		}

		chatSession := client.NewChat("You are a helpful calculator assistant.", initialMessages...)

		// Register a simple calculation tool
		calcTool := &testToolDef{
			name:        "calculate",
			description: "Perform basic arithmetic calculations",
			jsonSchema: `{
				"name": "calculate",
				"description": "Perform basic arithmetic calculations",
				"inputSchema": {
					"type": "object",
					"properties": {
						"expression": {
							"type": "string",
							"description": "The mathematical expression to evaluate"
						}
					},
					"required": ["expression"]
				}
			}`,
		}

		toolCalled := false
		err := chatSession.RegisterTool(calcTool, func(ctx context.Context, input string) string {
			toolCalled = true
			// Simple response for testing
			return `{"result": 42}`
		})
		require.NoError(t, err, "Failed to register tool")

		// Send a message that triggers tool use
		userMsg := "Please calculate 6 times 7 for me"
		response, err := chatSession.Message(context.Background(), chat.UserMessage(userMsg))
		require.NoError(t, err, "Failed to get response with tool")
		require.NotEmpty(t, response.GetText(), "Expected non-empty response")

		// Verify tool was called
		assert.True(t, toolCalled, "Tool should have been called")

		// Get the history
		_, history := chatSession.History()

		// Count occurrences of the new user message
		calcRequestCount := 0
		for _, msg := range history {
			if msg.GetText() == userMsg {
				calcRequestCount++
			}
		}

		// The user message should appear exactly once
		assert.Equal(t, 1, calcRequestCount, "User message requesting calculation should appear exactly once")

		// History should contain initial messages + new exchange
		// Note: Tool calls may add additional messages, but the user message should not be duplicated
		assert.GreaterOrEqual(t, len(history), 4, "History should have at least 4 messages")

		// Check for duplicate consecutive messages (a clear sign of the bug)
		for i := 1; i < len(history); i++ {
			if history[i].Role == history[i-1].Role &&
				history[i].GetText() == history[i-1].GetText() &&
				history[i].GetText() != "" {
				t.Errorf("Found duplicate consecutive messages at positions %d and %d: role=%s, text=%s",
					i-1, i, history[i].Role, history[i].GetText())
			}
		}
	})
}

// TestMessagePersistenceAfterRestore tests that messages are not duplicated
// when a session is restored from persistence and new messages are sent.
// This is a regression test for a bug where user messages were being pre-added
// to the store before calling Message(), causing them to be stored twice.
func TestMessagePersistenceAfterRestore(t *testing.T, client chat.Client) {
	store := persistence.NewMemoryStore()
	systemPrompt := "You are a helpful assistant."

	t.Run("SimpleMessages", func(t *testing.T) {
		// Create a new session with persistence
		session := agent.NewSession(client, systemPrompt, agent.WithStore(store))
		sessionID := session.SessionID()

		// Baseline record count before first message
		baselineRecords, err := store.GetAllRecords(sessionID)
		require.NoError(t, err)
		initialLen := len(baselineRecords)

		// Send first message
		userMsg1 := "Hello, how are you?"
		_, err = session.Message(context.Background(), chat.UserMessage(userMsg1))
		require.NoError(t, err)

		// Check that we have exactly 2 records (user + assistant)
		records1, err := store.GetAllRecords(sessionID)
		require.NoError(t, err)

		// Verify ordering for the first exchange
		require.GreaterOrEqual(t, len(records1), initialLen+2, "First exchange should add at least two records")
		exchange1 := records1[initialLen:]
		require.GreaterOrEqual(t, len(exchange1), 2)
		require.Equal(t, chat.UserRole, exchange1[0].Role, "First exchange should start with user record")
		require.Equal(t, userMsg1, exchange1[0].GetText())
		require.Equal(t, chat.AssistantRole, exchange1[1].Role, "Second record should be assistant response")

		userRecordCount1 := 0
		for _, r := range records1 {
			if r.Role == "user" && r.GetText() == userMsg1 {
				userRecordCount1++
			}
		}
		assert.Equal(t, 1, userRecordCount1, "First user message should appear exactly once in persistence")

		// Restore the session from persistence (simulating browser reload)
		restoredSession := agent.NewSession(client, systemPrompt,
			agent.WithStore(store),
			agent.WithRestoreSession(sessionID))

		// Send second message on restored session
		userMsg2 := "What is the capital of France?"
		_, err = restoredSession.Message(context.Background(), chat.UserMessage(userMsg2))
		require.NoError(t, err)

		// Get all records after second message
		records2, err := store.GetAllRecords(sessionID)
		require.NoError(t, err)

		// Newly added records should begin with the user message
		newRecords := records2[len(records1):]
		require.GreaterOrEqual(t, len(newRecords), 2, "Second exchange should add at least two records")
		require.Equal(t, chat.UserRole, newRecords[0].Role, "Second exchange should start with user record")
		require.Equal(t, userMsg2, newRecords[0].GetText())

		// Count occurrences of each user message in persistence
		userRecord1Count := 0
		userRecord2Count := 0
		for _, r := range records2 {
			if r.Role == "user" {
				if r.GetText() == userMsg1 {
					userRecord1Count++
				}
				if r.GetText() == userMsg2 {
					userRecord2Count++
				}
			}
		}

		// Each user message should appear exactly once in persistence
		assert.Equal(t, 1, userRecord1Count, "First user message should appear exactly once")
		assert.Equal(t, 1, userRecord2Count, "Second user message should appear exactly once")

		// Should have at least 4 records: user1, assistant1, user2, assistant2
		// (May have more if tool calls are involved, but should have at least these)
		assert.GreaterOrEqual(t, len(records2), 4, "Should have at least 4 records after 2 exchanges")
	})

	t.Run("WithToolCalls", func(t *testing.T) {
		// Create a new session with persistence and a tool
		session := agent.NewSession(client, systemPrompt, agent.WithStore(store))
		sessionID := session.SessionID()

		// Register a tool
		calcTool := &testToolDef{
			name:        "calculate",
			description: "Perform basic arithmetic calculations",
			jsonSchema: `{
				"name": "calculate",
				"description": "Perform basic arithmetic calculations",
				"inputSchema": {
					"type": "object",
					"properties": {
						"expression": {
							"type": "string",
							"description": "The mathematical expression to evaluate"
						}
					},
					"required": ["expression"]
				}
			}`,
		}

		err := session.RegisterTool(calcTool, func(ctx context.Context, input string) string {
			return `{"result": 42}`
		})
		require.NoError(t, err)

		baselineRecords, err := store.GetAllRecords(sessionID)
		require.NoError(t, err)
		initialLen := len(baselineRecords)

		// Send message that triggers tool use
		userMsg1 := "Calculate 6 times 7"
		_, err = session.Message(context.Background(), chat.UserMessage(userMsg1))
		require.NoError(t, err)

		// Get initial record count
		records1, err := store.GetAllRecords(sessionID)
		require.NoError(t, err)

		// Verify ordering: user message should precede any tool metadata
		require.GreaterOrEqual(t, len(records1), initialLen+2, "Tool exchange should add at least user and assistant records")
		exchange1 := records1[initialLen:]
		require.GreaterOrEqual(t, len(exchange1), 2)
		require.Equal(t, chat.UserRole, exchange1[0].Role, "Tool exchange should start with user record")
		require.Equal(t, userMsg1, exchange1[0].GetText())

		userRecordCount1 := 0
		for _, r := range records1 {
			if r.Role == "user" && r.GetText() == userMsg1 {
				userRecordCount1++
			}
		}
		assert.Equal(t, 1, userRecordCount1, "First user message should appear exactly once")

		// Restore session
		restoredSession := agent.NewSession(client, systemPrompt,
			agent.WithStore(store),
			agent.WithRestoreSession(sessionID))

		// Re-register tool on restored session
		err = restoredSession.RegisterTool(calcTool, func(ctx context.Context, input string) string {
			return `{"result": 84}`
		})
		require.NoError(t, err)

		// Send another message
		userMsg2 := "Now calculate 12 times 7"
		_, err = restoredSession.Message(context.Background(), chat.UserMessage(userMsg2))
		require.NoError(t, err)

		// Get all records
		records2, err := store.GetAllRecords(sessionID)
		require.NoError(t, err)

		// Newly persisted records for second exchange must start with the user input
		newRecords := records2[len(records1):]
		require.GreaterOrEqual(t, len(newRecords), 2, "Second tool exchange should add user and response records")
		require.Equal(t, chat.UserRole, newRecords[0].Role, "Second exchange should start with user record")
		require.Equal(t, userMsg2, newRecords[0].GetText())

		// Count user messages
		userRecord1Count := 0
		userRecord2Count := 0
		for _, r := range records2 {
			if r.Role == "user" {
				if r.GetText() == userMsg1 {
					userRecord1Count++
				}
				if r.GetText() == userMsg2 {
					userRecord2Count++
				}
			}
		}

		assert.Equal(t, 1, userRecord1Count, "First user message should appear exactly once")
		assert.Equal(t, 1, userRecord2Count, "Second user message should appear exactly once")
	})

	t.Run("EmptyTextMessage", func(t *testing.T) {
		// Test with a message that has no text (edge case)
		session := agent.NewSession(client, systemPrompt, agent.WithStore(store))
		sessionID := session.SessionID()

		// Send a normal message first
		userMsg1 := "Hello"
		_, err := session.Message(context.Background(), chat.UserMessage(userMsg1))
		require.NoError(t, err)

		records1, err := store.GetAllRecords(sessionID)
		require.NoError(t, err)

		userRecordCount1 := 0
		for _, r := range records1 {
			if r.Role == "user" && r.GetText() == userMsg1 {
				userRecordCount1++
			}
		}
		assert.Equal(t, 1, userRecordCount1, "User message should appear exactly once")

		// Restore and send another message
		restoredSession := agent.NewSession(client, systemPrompt,
			agent.WithStore(store),
			agent.WithRestoreSession(sessionID))

		userMsg2 := "Goodbye"
		_, err = restoredSession.Message(context.Background(), chat.UserMessage(userMsg2))
		require.NoError(t, err)

		records2, err := store.GetAllRecords(sessionID)
		require.NoError(t, err)

		userRecord1Count := 0
		userRecord2Count := 0
		for _, r := range records2 {
			if r.Role == "user" {
				if r.GetText() == userMsg1 {
					userRecord1Count++
				}
				if r.GetText() == userMsg2 {
					userRecord2Count++
				}
			}
		}

		assert.Equal(t, 1, userRecord1Count, "First user message should appear exactly once")
		assert.Equal(t, 1, userRecord2Count, "Second user message should appear exactly once")
	})
}

// TestThinkingPreservedInHistory tests that thinking content is preserved in message history
// for models that support thinking/reasoning. This test only applies to thinking-capable models
// like Claude Opus 4 and Sonnet 4.
func TestThinkingPreservedInHistory(t *testing.T, client chat.Client) {
	chatSession := client.NewChat("You are a helpful assistant. Think carefully before responding.")

	// Ask a question that should trigger thinking
	response, err := chatSession.Message(
		context.Background(),
		chat.UserMessage("What is the capital of France? Think carefully before answering."),
	)
	require.NoError(t, err, "Failed to get response")
	require.NotEmpty(t, response.GetText(), "Expected non-empty response")

	// Get the history
	_, history := chatSession.History()
	require.GreaterOrEqual(t, len(history), 2, "History should contain at least user message and assistant response")

	// Find the assistant's response in history
	var assistantMsg *chat.Message
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == chat.AssistantRole {
			assistantMsg = &history[i]
			break
		}
	}
	require.NotNil(t, assistantMsg, "Should have assistant message in history")

	// Check if the message contains thinking content
	hasThinking := false
	for _, content := range assistantMsg.Contents {
		if content.Thinking != nil && content.Thinking.Text != "" {
			hasThinking = true
			t.Logf("Found thinking content in history (length: %d chars)", len(content.Thinking.Text))
			break
		}
	}

	// For thinking models, we expect thinking content to be present
	// For non-thinking models, this test will still pass but won't find thinking
	if hasThinking {
		t.Log("Thinking content successfully preserved in message history")
	} else {
		t.Log("No thinking content found - model may not support thinking or didn't use it for this prompt")
	}
}

// TestThinkingPreservedWithToolCalls tests that thinking content is preserved alongside tool calls
// This ensures that when a model thinks and then calls tools, the thinking is retained in history
func TestThinkingPreservedWithToolCalls(t *testing.T, client chat.Client) {
	chatSession := client.NewChat("You are a helpful assistant with access to tools. Think carefully before using tools.")

	// Register a simple calculation tool
	calcTool := &testToolDef{
		name:        "calculate",
		description: "Perform arithmetic calculations",
		jsonSchema: `{
			"name": "calculate",
			"description": "Perform arithmetic calculations",
			"inputSchema": {
				"type": "object",
				"properties": {
					"a": {
						"type": "number",
						"description": "First number"
					},
					"b": {
						"type": "number",
						"description": "Second number"
					},
					"operation": {
						"type": "string",
						"description": "Operation to perform: add, subtract, multiply, divide"
					}
				},
				"required": ["a", "b", "operation"]
			}
		}`,
	}

	toolCalled := false
	err := chatSession.RegisterTool(calcTool, func(ctx context.Context, input string) string {
		toolCalled = true
		var params struct {
			A         float64 `json:"a"`
			B         float64 `json:"b"`
			Operation string  `json:"operation"`
		}
		if err := json.Unmarshal([]byte(input), &params); err != nil {
			return fmt.Sprintf(`{"error": "Failed to parse input: %v"}`, err)
		}

		var result float64
		switch params.Operation {
		case "add":
			result = params.A + params.B
		case "subtract":
			result = params.A - params.B
		case "multiply":
			result = params.A * params.B
		case "divide":
			if params.B != 0 {
				result = params.A / params.B
			} else {
				return `{"error": "Division by zero"}`
			}
		default:
			return `{"error": "Unknown operation"}`
		}
		return fmt.Sprintf(`{"result": %v}`, result)
	})
	require.NoError(t, err, "Failed to register tool")

	// Ask a question that should trigger both thinking and tool use
	response, err := chatSession.Message(
		context.Background(),
		chat.UserMessage("Please calculate 15 multiplied by 23. Think about it first, then use the calculator."),
	)
	require.NoError(t, err, "Failed to get response")
	require.NotEmpty(t, response.GetText(), "Expected non-empty response")
	require.True(t, toolCalled, "Tool should have been called")

	// Get the history
	_, history := chatSession.History()

	// Find messages with tool calls in history
	foundMessageWithToolCalls := false
	foundThinkingWithToolCalls := false

	for _, msg := range history {
		if msg.Role == chat.AssistantRole {
			hasToolCalls := msg.HasToolCalls()
			hasThinking := false
			for _, content := range msg.Contents {
				if content.Thinking != nil && content.Thinking.Text != "" {
					hasThinking = true
					break
				}
			}

			if hasToolCalls {
				foundMessageWithToolCalls = true
				if hasThinking {
					foundThinkingWithToolCalls = true
					t.Logf("Found assistant message with both thinking and tool calls")
				}
			}
		}
	}

	require.True(t, foundMessageWithToolCalls, "Should have found assistant message with tool calls")

	if foundThinkingWithToolCalls {
		t.Log("Thinking content successfully preserved alongside tool calls in message history")
	} else {
		t.Log("No thinking content found with tool calls - model may not support thinking or didn't use it for this prompt")
	}
}
