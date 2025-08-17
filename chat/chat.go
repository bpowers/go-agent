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
)

// TokenUsage represents token usage information for a chat session
type TokenUsage struct {
	// InputTokens is the number of tokens in the input/prompt
	InputTokens int `json:"input_tokens"`
	// OutputTokens is the number of tokens in the output/completion
	OutputTokens int `json:"output_tokens"`
	// TotalTokens is the total tokens used (input + output)
	TotalTokens int `json:"total_tokens"`
	// CachedTokens is the number of cached tokens used (if applicable)
	CachedTokens int `json:"cached_tokens,omitzero"`
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

// Chat is the stateful interface used to interact with an LLM in a turn-based way (including single-turn use).
type Chat interface {
	// Message sends a new message, as well as all previous messages, to an LLM returning the result.
	Message(ctx context.Context, msg Message, opts ...Option) (Message, error)
	// MessageStream sends a message and streams the response through the callback.
	// The callback is called with StreamEvent objects containing content chunks,
	// thinking status updates, and other streaming events.
	// The complete message is also returned after streaming completes.
	MessageStream(ctx context.Context, msg Message, callback StreamCallback, opts ...Option) (Message, error)
	// History extracts the system prompt and history up to this point for a chat for storage and later Chat object re-initialization.
	History() (systemPrompt string, msgs []Message)

	// TokenUsage returns the token usage for the last message exchange
	TokenUsage() (TokenUsage, error)
	// MaxTokens returns the maximum token limit for the model
	MaxTokens() int

	// RegisterTool registers a tool with its MCP definition and handler function.
	// Tools enable LLMs to perform actions by calling registered functions during conversation.
	// The def parameter should be a JSON string containing name, description, and inputSchema fields
	// following the Model Context Protocol (MCP) specification. The handler function receives
	// a context (which may contain request-specific state) and the tool arguments as a JSON string,
	// and returns a JSON string response.
	//
	// All LLM providers (OpenAI, Claude, Gemini) support multi-round tool calling, where the model
	// can request multiple tools in sequence, using outputs from earlier tools as inputs to later ones.
	// The implementation handles up to 10 rounds of tool calls to prevent infinite loops.
	// Tool execution is synchronous within each round but multiple tools requested in the same round
	// may be executed concurrently depending on the provider implementation.
	//
	// Context passing: The context provided to Message/MessageStream is propagated to tool handlers,
	// allowing tools to access request-scoped resources like filesystems or databases.
	// OpenAI's Responses API does not support tools and will automatically fall back to ChatCompletions
	// when tools are registered.
	RegisterTool(def string, fn func(context.Context, string) string) error
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
	Role    Role   `json:"role,omitzero"`
	Content string `json:"content,omitzero"`
}

// requestOpts is private so that Option can only be implemented by _this_ package.
type requestOpts struct {
	temperature     *float64
	maxTokens       int
	reasoningEffort string
	responseFormat  *JsonSchema
}

// Options shouldn't be used directly, but is public so that LLM implementations can reference it.
type Options struct {
	Temperature     *float64
	MaxTokens       int
	ReasoningEffort string
	ResponseFormat  *JsonSchema
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
