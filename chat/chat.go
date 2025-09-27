package chat

import (
	"context"

	"github.com/bpowers/go-agent/schema"
)

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

	// RegisterTool registers a tool with its MCP definition and handler function.
	// Tools enable LLMs to perform actions by calling registered functions during conversation.
	// The def parameter provides the tool's name, description, and MCP JSON schema.
	// The handler function receives a context (which may contain request-specific state)
	// and the tool arguments as a JSON string, and returns a JSON string response.
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
	RegisterTool(def ToolDef, fn func(context.Context, string) string) error
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

// Message represents a message to or from an LLM.
type Message struct {
	Role        Role         `json:"role,omitzero"`
	Content     string       `json:"content,omitzero"`
	ToolCalls   []ToolCall   `json:"tool_calls,omitzero"`
	ToolResults []ToolResult `json:"tool_results,omitzero"`
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
