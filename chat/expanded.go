package chat

import (
	"context"
	"encoding/json"
)

// Tool represents a function that can be called by the LLM.
type Tool struct {
	// Name is the unique identifier for this tool.
	Name string `json:"name"`
	// Description helps the LLM understand when to use this tool.
	Description string `json:"description"`
	// Parameters defines the JSON schema for the tool's input parameters.
	Parameters json.RawMessage `json:"parameters,omitempty"`
}

// ToolCall represents a request from the LLM to invoke a tool.
type ToolCall struct {
	// ID is a unique identifier for this tool call.
	ID string `json:"id"`
	// Name is the name of the tool to invoke.
	Name string `json:"name"`
	// Arguments contains the JSON-encoded arguments for the tool.
	Arguments json.RawMessage `json:"arguments"`
}

// ToolResult represents the result of executing a tool.
type ToolResult struct {
	// ToolCallID matches the ID from the corresponding ToolCall.
	ToolCallID string `json:"tool_call_id"`
	// Content is the result of the tool execution.
	Content string `json:"content"`
	// Error indicates if the tool execution failed.
	Error string `json:"error,omitempty"`
}

// StreamEventType represents the type of content in a streaming event.
type StreamEventType string

const (
	// StreamEventTypeContent indicates normal text content.
	StreamEventTypeContent StreamEventType = "content"
	// StreamEventTypeThinking indicates the model is thinking/reasoning.
	StreamEventTypeThinking StreamEventType = "thinking"
	// StreamEventTypeThinkingSummary provides a summary of the thinking process.
	StreamEventTypeThinkingSummary StreamEventType = "thinking_summary"
	// StreamEventTypeToolCall indicates a tool is being invoked.
	StreamEventTypeToolCall StreamEventType = "tool_call"
	// StreamEventTypeDone indicates the stream has completed.
	StreamEventTypeDone StreamEventType = "done"
)

// StreamEvent represents a chunk of data in a streaming response.
type StreamEvent struct {
	// Type indicates what kind of event this is.
	Type StreamEventType `json:"type"`
	// Content contains the text content for content events (was Delta).
	Content string `json:"content,omitzero"`
	// Delta contains the incremental text content (deprecated, use Content).
	Delta string `json:"delta,omitempty"`
	// ThinkingStatus contains status information for thinking events.
	ThinkingStatus *ThinkingStatus `json:"thinking_status,omitzero"`
	// ToolCalls contains any tool calls in this chunk.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// FinishReason indicates why the stream ended (if applicable).
	FinishReason string `json:"finish_reason,omitempty"`
}

// ThinkingStatus represents the status of model reasoning/thinking.
type ThinkingStatus struct {
	// IsThinking indicates if the model is currently thinking.
	IsThinking bool `json:"is_thinking"`
	// Summary provides a summary of what the model thought about.
	Summary string `json:"summary,omitzero"`
	// Duration indicates how long the thinking took (in milliseconds).
	Duration int64 `json:"duration,omitzero"`
}

// StreamCallback is called for each streaming event.
// If it returns an error, streaming will be stopped.
type StreamCallback func(event StreamEvent) error

// StreamHandler is a callback function for processing stream events.
// This is an alias for StreamCallback for backward compatibility.
type StreamHandler func(event StreamEvent) error

// AdvancedMessage extends Message with tool-related fields.
// This type is primarily used for advanced scenarios involving tool calls
// and their results. Most users should use the standard Message type.
type AdvancedMessage struct {
	Message
	// ToolCalls contains any tool invocations requested by the assistant.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolResults contains results from tool executions.
	ToolResults []ToolResult `json:"tool_results,omitempty"`
}

// AdvancedChat extends the Chat interface with explicit tool support methods.
// Note: The standard Chat interface already includes tool support via RegisterTool.
// This interface provides lower-level control for advanced use cases.
type AdvancedChat interface {
	Chat
	// MessageWithTools sends a message with available tools and returns the response.
	MessageWithTools(ctx context.Context, msg Message, tools []Tool, opts ...Option) (AdvancedMessage, error)
	// MessageStreamWithTools sends a message with tools and streams the response.
	// Note: Basic MessageStream is already in the Chat interface
	MessageStreamWithTools(ctx context.Context, msg Message, tools []Tool, handler StreamHandler, opts ...Option) error
	// SendToolResults sends the results of tool executions back to the LLM.
	SendToolResults(ctx context.Context, results []ToolResult, opts ...Option) (AdvancedMessage, error)
}

// AdvancedClient extends Client to create AdvancedChat instances.
// This interface is for providers that need explicit tool management
// beyond the standard RegisterTool mechanism.
type AdvancedClient interface {
	Client
	// NewAdvancedChat returns an AdvancedChat instance with enhanced capabilities.
	NewAdvancedChat(systemPrompt string, initialMsgs ...AdvancedMessage) AdvancedChat
}

// Additional options for advanced features

// WithTools specifies available tools for the LLM to use.
// Note: This is a placeholder for future API expansion.
// Currently, tools should be registered using Chat.RegisterTool().
func WithTools(tools []Tool) Option {
	return func(opts *requestOpts) {
		// This would need to be added to requestOpts
		// For now, this is a placeholder showing the API design
	}
}

// WithStreamHandler specifies a handler for streaming responses.
// Note: This is a placeholder for future API expansion.
// Currently, use MessageStream with a callback parameter.
func WithStreamHandler(handler StreamHandler) Option {
	return func(opts *requestOpts) {
		// This would need to be added to requestOpts
		// For now, this is a placeholder showing the API design
	}
}

// WithToolChoice specifies how the model should use tools.
// Values can be "auto", "none", or a specific tool name.
// Note: This is a placeholder for future API expansion.
func WithToolChoice(choice string) Option {
	return func(opts *requestOpts) {
		// This would need to be added to requestOpts
		// For now, this is a placeholder showing the API design
	}
}

// Helper functions for tool management

// MarshalToolArguments is a helper to marshal tool arguments.
func MarshalToolArguments(v interface{}) (json.RawMessage, error) {
	return json.Marshal(v)
}

// UnmarshalToolArguments is a helper to unmarshal tool arguments.
func UnmarshalToolArguments(data json.RawMessage, v interface{}) error {
	return json.Unmarshal(data, v)
}

// MakeToolParameters creates a JSON schema for tool parameters.
func MakeToolParameters(properties map[string]interface{}, required []string) json.RawMessage {
	schema := map[string]interface{}{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	data, _ := json.Marshal(schema)
	return data
}
