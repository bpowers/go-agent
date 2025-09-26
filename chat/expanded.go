package chat

import (
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
	// StreamEventTypeToolResult indicates the result of a tool execution.
	StreamEventTypeToolResult StreamEventType = "tool_result"
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
