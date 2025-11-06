package chat

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/bpowers/go-agent/schema"
)

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
	// Name is the tool name associated with this result.
	Name string `json:"name"`
	// Content is the result of the tool execution.
	Content string `json:"content"`
	// Error indicates if the tool execution failed.
	Error string `json:"error,omitzero"`
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
	// StreamEventTypeRedactedThinking indicates safety-flagged thinking content.
	StreamEventTypeRedactedThinking StreamEventType = "redacted_thinking"
	// StreamEventTypeToolCall indicates a tool is being invoked.
	StreamEventTypeToolCall StreamEventType = "tool_call"
	// StreamEventTypeToolResult indicates the result of a tool execution.
	StreamEventTypeToolResult StreamEventType = "tool_result"
	// StreamEventTypeServerToolUse indicates a server-side tool invocation.
	StreamEventTypeServerToolUse StreamEventType = "server_tool_use"
	// StreamEventTypeWebSearchResult indicates web search results from server-side search.
	StreamEventTypeWebSearchResult StreamEventType = "web_search_result"
	// StreamEventTypeCitation indicates citation updates.
	StreamEventTypeCitation StreamEventType = "citation"
	// StreamEventTypeDone indicates the stream has completed.
	StreamEventTypeDone StreamEventType = "done"
)

// StreamEvent represents a chunk of data in a streaming response.
type StreamEvent struct {
	// Type indicates what kind of event this is.
	Type StreamEventType `json:"type"`
	// Content contains the text content for content events.
	Content string `json:"content,omitzero"`
	// ThinkingStatus contains status information for thinking events.
	ThinkingStatus *ThinkingStatus `json:"thinking_status,omitzero"`
	// ToolCalls contains any tool calls in this chunk.
	ToolCalls []ToolCall `json:"tool_calls,omitzero"`
	// ToolResults contains tool execution outputs in this chunk.
	ToolResults []ToolResult `json:"tool_results,omitzero"`
	// FinishReason indicates why the stream ended (if applicable).
	FinishReason string `json:"finish_reason,omitzero"`
}

// ThinkingStatus represents the status of model reasoning/thinking.
type ThinkingStatus struct {
	// Summary provides a summary of what the model thought about.
	Summary string `json:"summary,omitzero"`
	// Signature contains the encrypted signature for thinking block verification.
	Signature string `json:"signature,omitzero"`
	// RedactedData contains encrypted thinking content when flagged by safety.
	RedactedData string `json:"redacted_data,omitzero"`
}

// ThinkingContent represents thinking/reasoning content in a message.
type ThinkingContent struct {
	// Text contains the thinking content.
	Text string `json:"text,omitzero"`
	// Signature contains the encrypted signature for thinking block verification.
	Signature string `json:"signature,omitzero"`
}

// StreamCallback is called for each streaming event.
// If it returns an error, streaming will be stopped.
type StreamCallback func(event StreamEvent) error

// StreamHandler is a callback function for processing stream events.
// This is an alias for StreamCallback for backward compatibility.
type StreamHandler func(event StreamEvent) error

// Role represents who a message came from.
type Role string

const (
	// UserRole identifies messages from the user.
	UserRole Role = "user"
	// AssistantRole identifies messages from the LLM.
	AssistantRole Role = "assistant"
	// ToolRole identifies messages originating from tool executions.
	ToolRole Role = "tool"
)

// TokenUsageDetails represents detailed token usage information
type TokenUsageDetails struct {
	// InputTokens is the number of tokens in the input/prompt
	InputTokens int `json:"input_tokens"`
	// OutputTokens is the number of tokens in the output/completion
	OutputTokens int `json:"output_tokens"`
	// TotalTokens is the total tokens used (input + output)
	TotalTokens int `json:"total_tokens"`
	// CachedTokens is the number of cached tokens used (if applicable)
	CachedTokens int `json:"cached_tokens,omitzero"`
}

// TokenUsage represents token usage for both the last message and cumulative session
type TokenUsage struct {
	// LastMessage contains token counts for the most recent message exchange
	LastMessage TokenUsageDetails `json:"last_message"`
	// Cumulative contains total token counts for the entire conversation
	Cumulative TokenUsageDetails `json:"cumulative"`
}

// TokenLimits represents the token limits for a given model
type TokenLimits struct {
	Context int `json:"context"`
	Output  int `json:"output"`
}

// ModelTokenLimits represents the token limits for a given named model
type ModelTokenLimits struct {
	Model string `json:"model"`
	TokenLimits
}

// ToolDef represents a tool definition that can be registered with an LLM.
type ToolDef interface {
	// MCPJsonSchema returns the MCP JSON schema for the tool as a compact JSON string
	MCPJsonSchema() string
	// Name returns the tool's name
	Name() string
	// Description returns the tool's description
	Description() string
}

// Tool represents a callable tool that can be registered with an LLM.
// It extends ToolDef with the ability to execute the tool.
type Tool interface {
	ToolDef
	// Call executes the tool with the given context and JSON input, returning JSON output
	Call(ctx context.Context, input string) string
}

// Chat is the stateful interface used to interact with an LLM in a turn-based way (including single-turn use).
type Chat interface {
	// Message sends a new message, as well as all previous messages, to an LLM returning the result.
	// Use WithStreamingCb option to receive streaming events during the call.
	Message(ctx context.Context, msg Message, opts ...Option) (Message, error)
	// History extracts the system prompt and history up to this point for a chat for storage and later Chat object re-initialization.
	History() (systemPrompt string, msgs []Message)

	// TokenUsage returns token usage for both the last message and cumulative session
	TokenUsage() (TokenUsage, error)
	// MaxTokens returns the maximum token limit for the model
	MaxTokens() int

	// RegisterTool registers a tool that can be called by the LLM during conversation.
	// Tools enable LLMs to perform actions by executing registered implementations.
	// The tool parameter provides the tool's name, description, MCP JSON schema, and execution handler.
	//
	// All LLM providers (OpenAI, Claude, Gemini) support multi-round tool calling, where the model
	// can request multiple tools in sequence, using outputs from earlier tools as inputs to later ones.
	// The implementation handles up to 10 rounds of tool calls to prevent infinite loops.
	// Tool execution is synchronous within each round but multiple tools requested in the same round
	// may be executed concurrently depending on the provider implementation.
	//
	// Context passing: The context provided to Message/MessageStream is propagated to tool handlers,
	// allowing tools to access request-scoped resources like filesystems or databases.
	//
	// Note: OpenAI's Responses API doesn't support tools yet. When tools are registered,
	// the OpenAI implementation automatically uses the ChatCompletions API instead.
	RegisterTool(tool Tool) error
	// DeregisterTool removes a tool by name
	DeregisterTool(name string)
	// ListTools returns the names of all registered tools
	ListTools() []string
}

// Client is used to create new chats that talk to a specific LLM hosted on a particular service (like Ollama, Anthropic, OpenAI, etc).
type Client interface {
	// NewChat returns a Chat instance configured for the current LLM with a given system prompt and initial messages.
	// It itself does not do API calls to the LLM, that happens when additional messages are added to the chat with the
	// Chat's Message method.
	NewChat(systemPrompt string, initialMsgs ...Message) Chat
}

// Content represents a single piece of content within a message.
// It uses a union-like structure where only one field should be set.
type Content struct {
	// Text content (most common case)
	Text string `json:"text,omitzero"`

	// Tool-related content
	ToolCall   *ToolCall   `json:"tool_call,omitzero"`
	ToolResult *ToolResult `json:"tool_result,omitzero"`

	// Thinking/reasoning content
	Thinking *ThinkingContent `json:"thinking,omitzero"`

	// System reminder content (ephemeral context added by tooling, filtered when replaying history)
	SystemReminder string `json:"system_reminder,omitzero"`
}

// Message represents a message to or from an LLM.
type Message struct {
	Role     Role      `json:"role,omitzero"`
	Contents []Content `json:"contents,omitzero"`
}

// requestOpts is private so that Option can only be implemented by _this_ package.
type requestOpts struct {
	temperature     *float64
	maxTokens       int
	reasoningEffort string
	responseFormat  *JsonSchema
	streamingCb     StreamCallback
}

// Options shouldn't be used directly, but is public so that LLM implementations can reference it.
type Options struct {
	Temperature     *float64
	MaxTokens       int
	ReasoningEffort string
	ResponseFormat  *JsonSchema
	StreamingCb     StreamCallback
}

// JsonSchema represents a requested schema that an LLM's response should conform to.
type JsonSchema struct {
	Name   string       `json:"name"`
	Strict bool         `json:"strict,omitzero"`
	Schema *schema.JSON `json:"schema,omitzero"`
}

// Option is a tunable parameter for an LLM interaction.
type Option func(*requestOpts)

// WithTemperature allows the user to change the randomness of the response - closer to 0
// for analytic or multiple choice responses, or closer to 1 for creative responses is a
// good mental model.
func WithTemperature(t float64) Option {
	return func(opts *requestOpts) {
		opts.temperature = &t
	}
}

// WithMaxTokens specifies the maximum number of tokens used to generate the response.
func WithMaxTokens(tokens int) Option {
	return func(opts *requestOpts) {
		opts.maxTokens = tokens
	}
}

// WithReasoningEffort specifies how hard a model should think - not all models support it.
func WithReasoningEffort(effort string) Option {
	return func(opts *requestOpts) {
		opts.reasoningEffort = effort
	}
}

// WithResponseFormat specifies the JSON schema to use to constrain the response.
func WithResponseFormat(name string, strict bool, schema *schema.JSON) Option {
	return func(opts *requestOpts) {
		opts.responseFormat = &JsonSchema{
			Name:   name,
			Strict: strict,
			Schema: schema,
		}
	}
}

// WithStreamingCb specifies a callback to receive streaming events during message processing.
func WithStreamingCb(callback StreamCallback) Option {
	return func(opts *requestOpts) {
		opts.streamingCb = callback
	}
}

// ApplyOptions is for use by LLM implementations, not users of the library.
func ApplyOptions(opts ...Option) Options {
	var options requestOpts
	for _, opt := range opts {
		opt(&options)
	}

	return Options{
		Temperature:     options.temperature,
		MaxTokens:       options.maxTokens,
		ReasoningEffort: options.reasoningEffort,
		ResponseFormat:  options.responseFormat,
		StreamingCb:     options.streamingCb,
	}
}

type debugDirContextKey struct{}

// WithDebugDir specifies a directory within which to store requests and response bodies for debugging purposes.
func WithDebugDir(ctx context.Context, dir string) context.Context {
	return context.WithValue(ctx, debugDirContextKey{}, dir)
}

// DebugDir returns the specified directory request and response bodies should be written to, if any.
func DebugDir(ctx context.Context) string {
	return ctx.Value(debugDirContextKey{}).(string)
}

// Helper functions for creating messages

// TextMessage creates a message with text content.
func TextMessage(role Role, text string) Message {
	return Message{
		Role: role,
		Contents: []Content{
			{Text: text},
		},
	}
}

// UserMessage creates a user message with text content.
func UserMessage(text string) Message {
	return TextMessage(UserRole, text)
}

// AssistantMessage creates an assistant message with text content.
func AssistantMessage(text string) Message {
	return TextMessage(AssistantRole, text)
}

// SystemMessage creates a system message with text content.
func SystemMessage(text string) Message {
	return Message{
		Role: "system",
		Contents: []Content{
			{Text: text},
		},
	}
}

// Builder pattern methods for complex messages

// AddText adds text content to the message.
func (m *Message) AddText(text string) *Message {
	m.Contents = append(m.Contents, Content{Text: text})
	return m
}

// AddToolCall adds a tool call to the message.
func (m *Message) AddToolCall(tc ToolCall) *Message {
	m.Contents = append(m.Contents, Content{ToolCall: &tc})
	return m
}

// AddToolResult adds a tool result to the message.
func (m *Message) AddToolResult(tr ToolResult) *Message {
	m.Contents = append(m.Contents, Content{ToolResult: &tr})
	return m
}

// AddThinking adds thinking/reasoning content to the message.
func (m *Message) AddThinking(text, signature string) *Message {
	m.Contents = append(m.Contents, Content{
		Thinking: &ThinkingContent{
			Text:      text,
			Signature: signature,
		},
	})
	return m
}

// GetText returns all text content concatenated with newlines.
// This is a convenience method for accessing text content.
func (m Message) GetText() string {
	var texts []string
	for _, c := range m.Contents {
		if c.Text != "" {
			texts = append(texts, c.Text)
		}
	}
	if len(texts) == 0 {
		return ""
	}
	if len(texts) == 1 {
		return texts[0]
	}
	return strings.Join(texts, "\n")
}

// GetToolCalls returns all tool calls in the message.
func (m Message) GetToolCalls() []ToolCall {
	var calls []ToolCall
	for _, c := range m.Contents {
		if c.ToolCall != nil {
			calls = append(calls, *c.ToolCall)
		}
	}
	return calls
}

// GetToolResults returns all tool results in the message.
func (m Message) GetToolResults() []ToolResult {
	var results []ToolResult
	for _, c := range m.Contents {
		if c.ToolResult != nil {
			results = append(results, *c.ToolResult)
		}
	}
	return results
}

// IsEmpty returns true if the message has no content.
func (m Message) IsEmpty() bool {
	return len(m.Contents) == 0
}

// HasText returns true if the message contains any text content.
func (m Message) HasText() bool {
	for _, c := range m.Contents {
		if c.Text != "" {
			return true
		}
	}
	return false
}

// HasToolCalls returns true if the message contains any tool calls.
func (m Message) HasToolCalls() bool {
	for _, c := range m.Contents {
		if c.ToolCall != nil {
			return true
		}
	}
	return false
}

// HasToolResults returns true if the message contains any tool results.
func (m Message) HasToolResults() bool {
	for _, c := range m.Contents {
		if c.ToolResult != nil {
			return true
		}
	}
	return false
}
