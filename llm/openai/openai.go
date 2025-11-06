package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/internal/logging"
	"github.com/bpowers/go-agent/llm/internal/common"
)

// logger is the package-level structured logger with provider context.
// Log levels used in this package:
//   - Info: Client creation, API selection, model warnings
//   - Debug: Stream events, tool calls, token updates, raw API data
//   - Warn: Missing token usage, unknown models, fallback behavior
//   - Error: Should never occur (indicates bugs)
var logger = logging.Logger().With("provider", "openai")

// logUnhandledEvent logs unhandled events at debug level
func logUnhandledEvent(logger *slog.Logger, apiName, eventType string, rawData interface{}) {
	if rawData != nil {
		if jsonBytes, err := json.Marshal(rawData); err == nil {
			logger.Debug("unhandled event type", "api", apiName, "type", eventType, "data", string(jsonBytes))
		} else {
			logger.Debug("unhandled event type", "api", apiName, "type", eventType, "data_raw", rawData)
		}
	} else {
		logger.Debug("unhandled event type", "api", apiName, "type", eventType)
	}
}

const (
	OpenAIURL = "https://api.openai.com/v1"
	OllamaURL = "http://localhost:11434/v1"
	GeminiURL = "https://generativelanguage.googleapis.com/v1beta/openai"
)

type API int

const (
	ChatCompletions API = iota
	Responses
)

// set is a simple set type for tracking unique values
type set[T comparable] map[T]struct{}

func (s set[T]) Add(v T) {
	s[v] = struct{}{}
}

func (s set[T]) Contains(v T) bool {
	_, ok := s[v]
	return ok
}

type client struct {
	openaiClient openai.Client
	modelName    string
	api          API
	apiSet       bool              // true if WithAPI was explicitly provided
	baseURL      string            // Store base URL for testing
	headers      map[string]string // Custom HTTP headers
	logger       *slog.Logger
}

var _ chat.Client = &client{}

type Option func(*client)

func WithModel(modelName string) Option {
	return func(c *client) {
		c.modelName = strings.TrimSpace(modelName)
	}
}

func WithAPI(api API) Option {
	return func(c *client) {
		c.api = api
		c.apiSet = true
	}
}

func WithHeaders(headers map[string]string) Option {
	return func(c *client) {
		c.headers = headers
	}
}

// NewClient returns a chat client that can begin chat sessions with an LLM service that speaks
// the OpenAI chat completion API.
func NewClient(apiBase string, apiKey string, opts ...Option) (chat.Client, error) {
	c := &client{
		api:     ChatCompletions, // default to chat completions
		baseURL: apiBase,         // Store for testing
		logger:  logger,
	}

	for _, opt := range opts {
		opt(c)
	}

	if c.modelName == "" {
		return nil, fmt.Errorf("WithModel is a required option")
	}

	// Auto-select Responses API for reasoning-first models unless explicitly overridden
	if !c.apiSet && isNoTemperatureModel(c.modelName) {
		c.api = Responses
	}

	// Build OpenAI client options
	clientOpts := []option.RequestOption{
		option.WithBaseURL(apiBase),
	}

	if apiKey != "" {
		clientOpts = append(clientOpts, option.WithAPIKey(apiKey))
	}

	// Add custom headers if provided
	for key, value := range c.headers {
		clientOpts = append(clientOpts, option.WithHeader(key, value))
	}

	c.openaiClient = openai.NewClient(clientOpts...)

	return c, nil
}

// BaseURL returns the base URL for testing purposes.
// This is exported for integration testing only.
func (c *client) BaseURL() string {
	return c.baseURL
}

// Headers returns the custom headers for testing purposes.
// This is exported for integration testing only.
func (c *client) Headers() map[string]string {
	return c.headers
}

// NewChat returns a chat instance.
func (c client) NewChat(systemPrompt string, initialMsgs ...chat.Message) chat.Chat {
	// Determine max tokens based on model
	maxTokens := getModelMaxTokens(c.modelName)

	return &chatClient{
		client:    c,
		state:     common.NewState(systemPrompt, initialMsgs),
		tools:     common.NewTools(),
		maxTokens: maxTokens,
	}
}

// modelLimits are in a particular order, so that longer, more specific strings come first
var modelLimits = []chat.ModelTokenLimits{
	{Model: "gpt-5-mini", TokenLimits: chat.TokenLimits{Context: 400000, Output: 128000}},
	{Model: "gpt-5-nano", TokenLimits: chat.TokenLimits{Context: 400000, Output: 128000}},
	{Model: "gpt-5", TokenLimits: chat.TokenLimits{Context: 400000, Output: 128000}},
	{Model: "gpt-4.5-preview", TokenLimits: chat.TokenLimits{Context: 128000, Output: 16384}},
	{Model: "gpt-4.1-mini", TokenLimits: chat.TokenLimits{Context: 1000000, Output: 32768}},
	{Model: "gpt-4.1", TokenLimits: chat.TokenLimits{Context: 1000000, Output: 32768}},
	{Model: "gpt-4o-mini", TokenLimits: chat.TokenLimits{Context: 128000, Output: 16384}},
	{Model: "gpt-4o", TokenLimits: chat.TokenLimits{Context: 128000, Output: 16384}},
	{Model: "gpt-4-turbo", TokenLimits: chat.TokenLimits{Context: 128000, Output: 4096}},
	{Model: "gpt-4", TokenLimits: chat.TokenLimits{Context: 8192, Output: 8192}},
	{Model: "o4-mini", TokenLimits: chat.TokenLimits{Context: 200000, Output: 100000}},
	{Model: "o3", TokenLimits: chat.TokenLimits{Context: 200000, Output: 100000}},
	{Model: "o3-mini", TokenLimits: chat.TokenLimits{Context: 200000, Output: 100000}},
	{Model: "gpt-3.5-turbo", TokenLimits: chat.TokenLimits{Context: 16385, Output: 4096}},
}

// getModelMaxTokens returns the maximum token limit for known models
func getModelMaxTokens(model string) int {
	modelLower := strings.ToLower(model)

	for _, m := range modelLimits {
		if strings.HasPrefix(modelLower, m.Model) {
			return m.TokenLimits.Output
		}
	}

	// Conservative default for unknown models instead of panic
	logger.Warn("unknown model, using conservative default output token limit", "model", model, "default_limit", 4096)
	return 4096
}

// isNoTemperatureModel checks if a model doesn't support custom temperature
func isNoTemperatureModel(model string) bool {
	modelLower := strings.ToLower(model)
	// gpt-5, o1, and o3 models don't support custom temperature
	return strings.HasPrefix(modelLower, "gpt-5") ||
		strings.HasPrefix(modelLower, "o1-") ||
		strings.HasPrefix(modelLower, "o3")
}

// withPrependedSystemReminder returns a new message with system reminder prepended as first content block
func withPrependedSystemReminder(ctx context.Context, msg chat.Message) chat.Message {
	if reminderFunc := chat.GetSystemReminder(ctx); reminderFunc != nil {
		if reminder := reminderFunc(); reminder != "" {
			newContents := make([]chat.Content, 0, len(msg.Contents)+1)
			newContents = append(newContents, chat.Content{SystemReminder: reminder})
			newContents = append(newContents, msg.Contents...)
			return chat.Message{Role: msg.Role, Contents: newContents}
		}
	}
	return msg
}

type chatClient struct {
	client
	state     *common.State
	tools     *common.Tools
	maxTokens int
}

// snapshotState returns a copy of the system prompt and message history.
// This allows streaming operations to work with a consistent view of the state
// without holding locks during long-running operations.
func (c *chatClient) snapshotState() (systemPrompt string, history []chat.Message) {
	return c.state.Snapshot()
}

// updateHistoryAndUsage appends messages to history and updates token usage.
// It properly manages locks using defer to ensure they're always released.
func (c *chatClient) updateHistoryAndUsage(msgs []chat.Message, usage chat.TokenUsageDetails) {
	c.state.AppendMessages(msgs, &usage)
}

func (c *chatClient) Message(ctx context.Context, msg chat.Message, opts ...chat.Option) (chat.Message, error) {
	// Apply options to get callback if provided
	appliedOpts := chat.ApplyOptions(opts...)
	callback := appliedOpts.StreamingCb

	// Determine route to appropriate API based on model type and whether tools are registered
	nTools := c.tools.Count()
	// Note: The Responses API doesn't support tools yet, so we fall back to ChatCompletions when tools are registered
	if c.api == Responses && nTools == 0 {
		return c.messageStreamResponses(ctx, msg, callback, opts...)
	}
	return c.messageStreamChatCompletions(ctx, msg, callback, opts...)
}

// messageStreamResponses uses the Responses API for reasoning models (gpt-5, o1, o3)
func (c *chatClient) messageStreamResponses(ctx context.Context, msg chat.Message, callback chat.StreamCallback, opts ...chat.Option) (chat.Message, error) {
	reqOpts := chat.ApplyOptions(opts...)

	// Snapshot state without holding lock during streaming
	systemPrompt, history := c.snapshotState()

	// Build input items for Responses API
	var inputItems []responses.ResponseInputItemUnionParam

	// Add system prompt as a system message if present
	if systemPrompt != "" {
		inputItems = append(inputItems, responses.ResponseInputItemUnionParam{
			OfMessage: &responses.EasyInputMessageParam{
				Role:    responses.EasyInputMessageRoleDeveloper, // Use developer role for system prompts
				Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(systemPrompt)},
			},
		})
	}

	// Add history messages using direct Contents access
	for _, m := range history {
		var role responses.EasyInputMessageRole
		switch m.Role {
		case chat.UserRole:
			role = responses.EasyInputMessageRoleUser
		case chat.AssistantRole:
			role = responses.EasyInputMessageRoleAssistant
		default:
			role = responses.EasyInputMessageRoleDeveloper
		}

		// Extract text content directly from Contents array
		text := extractText(m)
		if text == "" {
			continue // Skip messages without text content
		}

		inputItems = append(inputItems, responses.ResponseInputItemUnionParam{
			OfMessage: &responses.EasyInputMessageParam{
				Role:    role,
				Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(text)},
			},
		})
	}

	// Add current message with system reminder prepended if present
	// This message (with system reminder) will be persisted for audit trail
	msgWithReminder := withPrependedSystemReminder(ctx, msg)

	var currentRole responses.EasyInputMessageRole
	switch msgWithReminder.Role {
	case chat.UserRole:
		currentRole = responses.EasyInputMessageRoleUser
	case chat.AssistantRole:
		currentRole = responses.EasyInputMessageRoleAssistant
	default:
		currentRole = responses.EasyInputMessageRoleDeveloper
	}

	text := extractText(msgWithReminder)
	if text == "" {
		return chat.Message{}, fmt.Errorf("current message has no text content")
	}

	inputItems = append(inputItems, responses.ResponseInputItemUnionParam{
		OfMessage: &responses.EasyInputMessageParam{
			Role:    currentRole,
			Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(text)},
		},
	})

	// Build request parameters for Responses API
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(c.modelName),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam(inputItems),
		},
	}

	// Set temperature if provided (Responses API doesn't restrict temperature for reasoning models)
	if reqOpts.Temperature != nil {
		params.Temperature = param.NewOpt(*reqOpts.Temperature)
	}

	if reqOpts.MaxTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(reqOpts.MaxTokens))
	}

	c.logger.Debug("starting stream", "api", "responses", "model", c.modelName)

	// Create streaming response
	stream := c.openaiClient.Responses.NewStreaming(ctx, params)

	var respContent strings.Builder
	var reasoningContent strings.Builder
	var inReasoning bool
	eventCount := 0
	var lastUsage chat.TokenUsageDetails
	// For tracking tool calls in Responses API
	var toolCalls []responses.ResponseFunctionToolCall
	var currentToolCall *responses.ResponseFunctionToolCall
	_ = currentToolCall // Used in response.output_item.added
	var toolCallArgs strings.Builder
	_ = toolCallArgs // Will be used when we fully implement tool call argument streaming

	for stream.Next() {
		event := stream.Current()
		eventCount++

		c.logger.Debug("event received", "api", "responses", "event_num", eventCount, "type", event.Type)

		// Handle different event types
		switch event.Type {
		case "response.reasoning.delta", "response.thinking.delta", "response.reasoning_summary.delta", "response.reasoning_summary_text.delta":
			// Reasoning content is being streamed
			if !inReasoning && callback != nil {
				inReasoning = true
				// Emit thinking started event
				thinkingEvent := chat.StreamEvent{
					Type:           chat.StreamEventTypeThinking,
					ThinkingStatus: &chat.ThinkingStatus{},
				}
				if err := callback(thinkingEvent); err != nil {
					return chat.Message{}, err
				}
			}

			// Extract reasoning delta content
			if deltaStr := event.Delta.OfString; deltaStr != "" {
				reasoningContent.WriteString(deltaStr)
				if callback != nil {
					// Stream reasoning content
					thinkingEvent := chat.StreamEvent{
						Type:           chat.StreamEventTypeThinking,
						Content:        deltaStr,
						ThinkingStatus: &chat.ThinkingStatus{},
					}
					if err := callback(thinkingEvent); err != nil {
						return chat.Message{}, err
					}
				}
			}

		case "response.reasoning.done", "response.thinking.done", "response.reasoning_summary.done", "response.reasoning_summary_text.done":
			// Reasoning is complete
			if inReasoning && callback != nil {
				inReasoning = false
				// Send thinking summary event
				thinkingSummaryEvent := chat.StreamEvent{
					Type: chat.StreamEventTypeThinkingSummary,
					ThinkingStatus: &chat.ThinkingStatus{
						Summary: reasoningContent.String(),
					},
				}
				if err := callback(thinkingSummaryEvent); err != nil {
					return chat.Message{}, err
				}
			}

		case "response.output_text.delta":
			// Regular output text
			if inReasoning && callback != nil {
				// End reasoning if we're getting output text
				inReasoning = false
				if reasoningContent.Len() > 0 {
					thinkingSummaryEvent := chat.StreamEvent{
						Type: chat.StreamEventTypeThinkingSummary,
						ThinkingStatus: &chat.ThinkingStatus{
							Summary: reasoningContent.String(),
						},
					}
					if err := callback(thinkingSummaryEvent); err != nil {
						return chat.Message{}, err
					}
				}
			}

			if deltaStr := event.Delta.OfString; deltaStr != "" {
				respContent.WriteString(deltaStr)
				if callback != nil {
					event := chat.StreamEvent{
						Type:    chat.StreamEventTypeContent,
						Content: deltaStr,
					}
					if err := callback(event); err != nil {
						return chat.Message{}, err
					}
				}
			}

		case "response.completed":
			// Response is complete - extract usage information
			if event.JSON.Response.Valid() && event.Response.JSON.Usage.Valid() {
				usage := chat.TokenUsageDetails{
					InputTokens:  int(event.Response.Usage.InputTokens),
					OutputTokens: int(event.Response.Usage.OutputTokens),
					TotalTokens:  int(event.Response.Usage.TotalTokens),
				}
				lastUsage = usage
				c.logger.Debug("usage from completed event", "api", "responses", "input", usage.InputTokens, "output", usage.OutputTokens, "total", usage.TotalTokens)
			}
			c.logger.Debug("stream completed", "api", "responses")

		case "response.output_text.done":
			// Text output is complete
			c.logger.Debug("output text done", "api", "responses")

		case "response.created", "response.in_progress":
			// Status events - just log at debug level
			c.logger.Debug("status event", "api", "responses", "type", event.Type)

		case "response.output_item.added":
			// Check if this is a function call item
			if event.Item.Type == "function_call" && event.Item.ID != "" && event.Item.Name != "" {
				currentToolCall = &responses.ResponseFunctionToolCall{
					ID:   event.Item.ID,
					Name: event.Item.Name,
				}
				c.logger.Debug("tool call started", "api", "responses", "id", event.Item.ID, "name", event.Item.Name)
			} else {
				// Non-function item added (reasoning, message, etc.)
				c.logger.Debug("output item added", "api", "responses", "type", event.Item.Type)
			}

		case "response.output_item.done", "response.content_part.added", "response.content_part.done":
			// Informational events about content structure
			c.logger.Debug("content structure event", "api", "responses", "type", event.Type)

		case "error":
			// Handle error events
			c.logger.Debug("error event received", "api", "responses")

		default:
			// Log unhandled event types at debug level
			logUnhandledEvent(c.logger, "Responses API", event.Type, event)
		}
	}

	if err := stream.Err(); err != nil {
		return chat.Message{}, fmt.Errorf("responses API streaming error: %w", err)
	}

	// Note: Tool calls in Responses API would need different handling than ChatCompletions
	// The Responses API handles tools differently - it doesn't use the multi-round pattern
	// For now, we log if tools were detected but not fully implemented
	if len(toolCalls) > 0 {
		c.logger.Warn("tool calls detected but not yet fully implemented for Responses API", "api", "responses", "tool_count", len(toolCalls))
		for _, tc := range toolCalls {
			c.logger.Debug("tool detected", "api", "responses", "name", tc.Name, "id", tc.ID)
		}
		// For now, just include any tool call info in the response
		// TODO: Implement proper tool handling for Responses API
	}

	respMsg := chat.AssistantMessage(respContent.String())

	// Update history and usage under lock
	// Persist the message WITH system reminder for complete audit trail
	c.updateHistoryAndUsage([]chat.Message{msgWithReminder, respMsg}, lastUsage)

	return respMsg, nil
}

// messageStreamChatCompletions uses the standard Chat Completions API
func (c *chatClient) messageStreamChatCompletions(ctx context.Context, msg chat.Message, callback chat.StreamCallback, opts ...chat.Option) (chat.Message, error) {
	reqOpts := chat.ApplyOptions(opts...)

	// Snapshot state without holding lock during streaming
	systemPrompt, history := c.snapshotState()

	// Build message list
	var messages []openai.ChatCompletionMessageParamUnion

	// Add system prompt if present
	if systemPrompt != "" {
		messages = append(messages, openai.SystemMessage(systemPrompt))
	}

	// Convert history messages using the new converter
	historyMsgs, err := messagesToOpenAI(history)
	if err != nil {
		return chat.Message{}, fmt.Errorf("converting history messages: %w", err)
	}
	messages = append(messages, historyMsgs...)

	// Convert current message using the new converter, prepending system reminder if present
	// This message (with system reminder) will be persisted for audit trail
	msgWithReminder := withPrependedSystemReminder(ctx, msg)
	currentMsgs, err := messageToOpenAI(msgWithReminder)
	if err != nil {
		return chat.Message{}, fmt.Errorf("converting current message: %w", err)
	}
	messages = append(messages, currentMsgs...)

	// Build request parameters
	params := openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    c.modelName,
	}

	// Add tools if registered
	allTools := c.tools.GetAll()
	if len(allTools) > 0 {
		tools := make([]openai.ChatCompletionToolParam, 0, len(allTools))
		for _, tool := range allTools {
			toolParam, err := c.mcpToOpenAITool(tool)
			if err != nil {
				return chat.Message{}, fmt.Errorf("failed to convert tool: %w", err)
			}
			tools = append(tools, toolParam)
		}
		params.Tools = tools
	}

	// Track if temperature was set for error retry logic
	temperatureSet := false
	// Only set temperature for models that support it
	if reqOpts.Temperature != nil && !isNoTemperatureModel(c.modelName) {
		params.Temperature = openai.Float(*reqOpts.Temperature)
		temperatureSet = true
	}

	if reqOpts.MaxTokens > 0 {
		params.MaxCompletionTokens = openai.Int(int64(reqOpts.MaxTokens))
	}

	if reqOpts.ResponseFormat != nil && reqOpts.ResponseFormat.Schema != nil {
		// Response format configuration would go here if supported by the SDK
		// Currently skipping as the exact API may differ
	}

	// Handle reasoning effort for o1 models
	if reqOpts.ReasoningEffort != "" && c.api == Responses {
		// Reasoning effort is supported through the Responses API
		// It can be configured in the ResponseNewParams if needed
		c.logger.Debug("reasoning effort set", "api", "responses", "effort", reqOpts.ReasoningEffort)
	}

	// Add stream options to include usage information
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: param.NewOpt(true),
	}

	// Streaming implementation
	stream := c.openaiClient.Chat.Completions.NewStreaming(ctx, params)

	var respContent strings.Builder
	var thinkingContent strings.Builder
	var inThinking bool
	chunkCount := 0
	var toolCalls []openai.ChatCompletionMessageToolCall
	var toolCallArgs map[int]strings.Builder = make(map[int]strings.Builder)
	toolCallEmitted := make(set[int])
	var lastUsage chat.TokenUsageDetails

	for stream.Next() {
		chunk := stream.Current()
		chunkCount++

		// Check for usage information (provided in the final chunk when stream_options.include_usage is true)
		if chunk.JSON.Usage.Valid() && chunk.Usage.PromptTokens > 0 {
			// This is the final usage chunk
			usage := chat.TokenUsageDetails{
				InputTokens:  int(chunk.Usage.PromptTokens),
				OutputTokens: int(chunk.Usage.CompletionTokens),
				TotalTokens:  int(chunk.Usage.TotalTokens),
			}
			lastUsage = usage
			c.logger.Debug("usage chunk received", "api", "chat_completions", "input", usage.InputTokens, "output", usage.OutputTokens, "total", usage.TotalTokens)
		}

		// Debug logging for SSE responses
		rawJSON := chunk.RawJSON()
		c.logger.Debug("chunk received", "api", "chat_completions", "chunk_num", chunkCount, "model", c.modelName, "raw", string(rawJSON))

		// Log structured information about the chunk
		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			c.logger.Debug("chunk choice", "api", "chat_completions", "chunk_num", chunkCount, "index", choice.Index, "finish_reason", choice.FinishReason, "role", choice.Delta.Role, "content", choice.Delta.Content)

			// Check for extra fields that might contain reasoning content
			if len(choice.Delta.JSON.ExtraFields) > 0 {
				extraFieldsJSON, _ := json.Marshal(choice.Delta.JSON.ExtraFields)
				c.logger.Debug("delta extra fields", "api", "chat_completions", "chunk_num", chunkCount, "fields", string(extraFieldsJSON))
			}
			if len(choice.JSON.ExtraFields) > 0 {
				extraFieldsJSON, _ := json.Marshal(choice.JSON.ExtraFields)
				c.logger.Debug("choice extra fields", "api", "chat_completions", "chunk_num", chunkCount, "fields", string(extraFieldsJSON))
			}
		}

		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]

			// Check for refusal content
			if choice.Delta.Refusal != "" {
				refusalContent := choice.Delta.Refusal
				respContent.WriteString(refusalContent)

				if callback != nil {
					event := chat.StreamEvent{
						Type:    chat.StreamEventTypeContent,
						Content: refusalContent,
					}
					if err := callback(event); err != nil {
						return chat.Message{}, err
					}
				}

				c.logger.Debug("refusal content", "api", "chat_completions", "content", refusalContent)
			}

			// Check for tool calls
			if len(choice.Delta.ToolCalls) > 0 {
				for _, tc := range choice.Delta.ToolCalls {
					// New tool call or continuation
					idx := int(tc.Index)

					// Ensure we have enough space in toolCalls slice
					for len(toolCalls) <= idx {
						toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCall{})
					}

					if tc.ID != "" {
						// Starting a new tool call
						toolCalls[idx].ID = tc.ID
						toolCallArgs[idx] = strings.Builder{}
					}

					if tc.Function.Name != "" {
						toolCalls[idx].Function.Name = tc.Function.Name
					}
					if tc.Function.Arguments != "" {
						builder := toolCallArgs[idx]
						builder.WriteString(tc.Function.Arguments)
						toolCallArgs[idx] = builder
						toolCalls[idx].Function.Arguments = builder.String()
					}

					// Emit tool call event only once per index when arguments are valid JSON
					if callback != nil && !toolCallEmitted.Contains(idx) && toolCalls[idx].ID != "" && toolCalls[idx].Function.Name != "" && isValidJSON(toolCalls[idx].Function.Arguments) {
						toolCallEvent := chat.StreamEvent{
							Type: chat.StreamEventTypeToolCall,
							ToolCalls: []chat.ToolCall{
								{
									ID:        toolCalls[idx].ID,
									Name:      toolCalls[idx].Function.Name,
									Arguments: json.RawMessage(toolCalls[idx].Function.Arguments),
								},
							},
						}
						toolCallEmitted.Add(idx)
						if err := callback(toolCallEvent); err != nil {
							return chat.Message{}, err
						}
					}
				}
			}

			// Check for regular content
			if choice.Delta.Content != "" {
				content := choice.Delta.Content

				// If we were in thinking mode and now getting regular content, end thinking
				if inThinking && callback != nil {
					inThinking = false
					// Send thinking summary
					if thinkingContent.Len() > 0 {
						event := chat.StreamEvent{
							Type: chat.StreamEventTypeThinkingSummary,
							ThinkingStatus: &chat.ThinkingStatus{
								Summary: thinkingContent.String(),
							},
						}
						if err := callback(event); err != nil {
							return chat.Message{}, err
						}
					}
				}

				respContent.WriteString(content)

				// Call the callback with the content event
				if callback != nil {
					event := chat.StreamEvent{
						Type:    chat.StreamEventTypeContent,
						Content: content,
					}
					if err := callback(event); err != nil {
						// User requested to stop streaming
						return chat.Message{}, err
					}
				}
			}

			// Check if stream is done
			if choice.FinishReason != "" {
				c.logger.Debug("stream finished", "api", "chat_completions", "reason", choice.FinishReason)
			}

			// Log any unhandled extra fields
			if len(choice.Delta.JSON.ExtraFields) > 0 {
				for fieldName, field := range choice.Delta.JSON.ExtraFields {
					if field.Valid() {
						c.logger.Debug("unhandled extra field", "api", "chat_completions", "field", fieldName, "value", field.Raw())
					}
				}
			}
		}
	}

	c.logger.Debug("stream completed", "api", "chat_completions", "total_chunks", chunkCount)

	if err := stream.Err(); err != nil {
		// Check if the error is about unsupported temperature
		errStr := err.Error()
		if strings.Contains(errStr, "temperature") && strings.Contains(errStr, "does not support") && temperatureSet {
			// Retry without temperature
			c.logger.Info("retrying without temperature", "model", c.modelName, "reason", "temperature not supported")
			// Create new params without temperature
			paramsNoTemp := openai.ChatCompletionNewParams{
				Messages: messages,
				Model:    c.modelName,
			}
			if reqOpts.MaxTokens > 0 {
				paramsNoTemp.MaxCompletionTokens = openai.Int(int64(reqOpts.MaxTokens))
			}
			// Add tools if registered (for retry)
			allTools := c.tools.GetAll()
			if len(allTools) > 0 {
				tools := make([]openai.ChatCompletionToolParam, 0, len(allTools))
				for _, tool := range allTools {
					toolParam, err := c.mcpToOpenAITool(tool)
					if err != nil {
						// Skip this tool on error
						continue
					}
					tools = append(tools, toolParam)
				}
				paramsNoTemp.Tools = tools
			}
			// Add stream options to include usage information
			paramsNoTemp.StreamOptions = openai.ChatCompletionStreamOptionsParam{
				IncludeUsage: param.NewOpt(true),
			}
			stream = c.openaiClient.Chat.Completions.NewStreaming(ctx, paramsNoTemp)

			respContent.Reset()
			thinkingContent.Reset()
			inThinking = false
			chunkCount = 0
			lastUsage = chat.TokenUsageDetails{}

			for stream.Next() {
				chunk := stream.Current()
				chunkCount++

				// Check for usage information in retry path
				if chunk.JSON.Usage.Valid() && chunk.Usage.PromptTokens > 0 {
					usage := chat.TokenUsageDetails{
						InputTokens:  int(chunk.Usage.PromptTokens),
						OutputTokens: int(chunk.Usage.CompletionTokens),
						TotalTokens:  int(chunk.Usage.TotalTokens),
					}
					lastUsage = usage
					c.logger.Debug("retry usage chunk received", "api", "chat_completions", "input", usage.InputTokens, "output", usage.OutputTokens, "total", usage.TotalTokens)
				}

				c.logger.Debug("retry chunk received", "api", "chat_completions", "chunk_num", chunkCount, "model", c.modelName)

				if len(chunk.Choices) > 0 {
					choice := chunk.Choices[0]

					// Check for reasoning content in retry
					reasoningFieldNames := []string{"reasoning_content", "reasoning", "thinking_content", "thinking"}
					var reasoningFieldRaw string

					for _, fieldName := range reasoningFieldNames {
						if field, exists := choice.Delta.JSON.ExtraFields[fieldName]; exists && field.Valid() {
							reasoningFieldRaw = field.Raw()
							break
						}
					}

					if reasoningFieldRaw != "" {
						var reasoningContent string
						if err := json.Unmarshal([]byte(reasoningFieldRaw), &reasoningContent); err == nil && reasoningContent != "" {
							if !inThinking && callback != nil {
								inThinking = true
								event := chat.StreamEvent{
									Type:           chat.StreamEventTypeThinking,
									ThinkingStatus: &chat.ThinkingStatus{},
								}
								if err := callback(event); err != nil {
									return chat.Message{}, err
								}
							}

							thinkingContent.WriteString(reasoningContent)
							if callback != nil {
								event := chat.StreamEvent{
									Type:           chat.StreamEventTypeThinking,
									Content:        reasoningContent,
									ThinkingStatus: &chat.ThinkingStatus{},
								}
								if err := callback(event); err != nil {
									return chat.Message{}, err
								}
							}
						}
					}

					if choice.Delta.Content != "" {
						content := choice.Delta.Content

						if inThinking && callback != nil {
							inThinking = false
							if thinkingContent.Len() > 0 {
								event := chat.StreamEvent{
									Type: chat.StreamEventTypeThinkingSummary,
									ThinkingStatus: &chat.ThinkingStatus{
										Summary: thinkingContent.String(),
									},
								}
								if err := callback(event); err != nil {
									return chat.Message{}, err
								}
							}
						}

						respContent.WriteString(content)

						if callback != nil {
							event := chat.StreamEvent{
								Type:    chat.StreamEventTypeContent,
								Content: content,
							}
							if err := callback(event); err != nil {
								return chat.Message{}, err
							}
						}
					}
				}
			}

			if err := stream.Err(); err != nil {
				return chat.Message{}, fmt.Errorf("streaming error after temperature retry: %w", err)
			}
		} else {
			return chat.Message{}, fmt.Errorf("streaming error: %w", err)
		}
	}

	// Handle tool calls with multiple rounds if needed
	if len(toolCalls) > 0 {
		return c.handleToolCallRounds(ctx, msgWithReminder, respContent.String(), toolCalls, reqOpts, callback)
	}

	respMsg := chat.AssistantMessage(respContent.String())

	// Update history and usage under lock
	// Persist the message WITH system reminder for complete audit trail
	c.updateHistoryAndUsage([]chat.Message{msgWithReminder, respMsg}, lastUsage)

	// Update last usage
	if lastUsage.TotalTokens == 0 {
		c.logger.Warn("no token usage information received", "api", "chat_completions")
	}

	return respMsg, nil
}

// handleToolCallRounds handles potentially multiple rounds of tool calls
func (c *chatClient) handleToolCallRounds(ctx context.Context, initialMsg chat.Message, initialContent string, initialToolCalls []openai.ChatCompletionMessageToolCall, reqOpts chat.Options, callback chat.StreamCallback) (chat.Message, error) {
	// Keep track of all messages for the conversation
	var msgs []openai.ChatCompletionMessageParamUnion

	// Build conversation messages and update history
	systemPrompt, history := c.state.Snapshot()
	if systemPrompt != "" {
		msgs = append(msgs, openai.SystemMessage(systemPrompt))
	}
	historyMsgs, err := messagesToOpenAI(history)
	if err != nil {
		return chat.Message{}, fmt.Errorf("converting history messages: %w", err)
	}
	msgs = append(msgs, historyMsgs...)

	// Add the initial user message to history
	c.state.AppendMessages([]chat.Message{initialMsg}, nil)

	// Convert initial message
	initialMsgs, err := messageToOpenAI(initialMsg)
	if err != nil {
		return chat.Message{}, fmt.Errorf("converting initial message: %w", err)
	}
	msgs = append(msgs, initialMsgs...)

	// Process tool calls in a loop until we get a final response
	toolCalls := initialToolCalls
	isFirstIteration := true

	for len(toolCalls) > 0 {
		c.logger.Debug("processing tool calls", "count", len(toolCalls))

		// Execute tool calls
		chatToolResults, err := c.handleToolCalls(ctx, toolCalls, callback)
		if err != nil {
			return chat.Message{}, fmt.Errorf("failed to execute tool calls: %w", err)
		}

		// Persist assistant tool call message and tool results in chat state
		chatToolCalls := make([]chat.ToolCall, len(toolCalls))
		for i, tc := range toolCalls {
			chatToolCalls[i] = openaiToolCallToChat(tc)
		}
		assistantMsg := chat.Message{Role: chat.AssistantRole}
		// Add initial text content to the first assistant message if present
		if isFirstIteration && initialContent != "" {
			assistantMsg.AddText(initialContent)
		}
		for _, tc := range chatToolCalls {
			assistantMsg.AddToolCall(tc)
		}
		toolMessages := []chat.Message{assistantMsg}
		if len(chatToolResults) > 0 {
			toolMsg := chat.Message{Role: chat.ToolRole}
			for _, tr := range chatToolResults {
				toolMsg.AddToolResult(tr)
			}
			toolMessages = append(toolMessages, toolMsg)
		}
		c.state.AppendMessages(toolMessages, nil)

		// Convert assistant message with tool calls using the new converter
		assistantMsgs, err := messageToOpenAI(assistantMsg)
		if err != nil {
			return chat.Message{}, fmt.Errorf("converting assistant message with tool calls: %w", err)
		}
		msgs = append(msgs, assistantMsgs...)

		// Add tool results to messages (only if non-empty to avoid potential API issues)
		if len(chatToolResults) > 0 {
			// Prepend system reminder before converting tool messages
			toolMsg := toolMessages[len(toolMessages)-1]
			toolMsgWithReminder := withPrependedSystemReminder(ctx, toolMsg)
			toolResultMsgs, err := messageToOpenAI(toolMsgWithReminder)
			if err != nil {
				return chat.Message{}, fmt.Errorf("converting tool result messages: %w", err)
			}
			msgs = append(msgs, toolResultMsgs...)
		}

		// Make another API call with tool results
		followUpParams := openai.ChatCompletionNewParams{
			Messages: msgs,
			Model:    c.modelName,
		}
		if reqOpts.Temperature != nil {
			followUpParams.Temperature = openai.Float(*reqOpts.Temperature)
		}
		if reqOpts.MaxTokens > 0 {
			followUpParams.MaxCompletionTokens = openai.Int(int64(reqOpts.MaxTokens))
		}
		// Add tools if registered (for follow-up after tool execution)
		allTools := c.tools.GetAll()
		if len(allTools) > 0 {
			tools := make([]openai.ChatCompletionToolParam, 0, len(allTools))
			for _, tool := range allTools {
				toolParam, err := c.mcpToOpenAITool(tool)
				if err != nil {
					// Skip this tool on error
					continue
				}
				tools = append(tools, toolParam)
			}
			followUpParams.Tools = tools
		}
		// Add stream options to include usage information
		followUpParams.StreamOptions = openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: param.NewOpt(true),
		}

		// Create a new stream for the follow-up request
		followUpStream := c.openaiClient.Chat.Completions.NewStreaming(ctx, followUpParams)

		// Process the follow-up stream
		var respContent strings.Builder
		toolCalls = nil // Reset for next round
		var toolCallArgs map[int]strings.Builder = make(map[int]strings.Builder)
		toolCallEmitted := make(set[int])
		var lastUsage chat.TokenUsageDetails

		for followUpStream.Next() {
			chunk := followUpStream.Current()

			// Check for usage information
			if chunk.JSON.Usage.Valid() && chunk.Usage.PromptTokens > 0 {
				usage := chat.TokenUsageDetails{
					InputTokens:  int(chunk.Usage.PromptTokens),
					OutputTokens: int(chunk.Usage.CompletionTokens),
					TotalTokens:  int(chunk.Usage.TotalTokens),
				}
				lastUsage = usage
			}

			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]

				// Check for refusal content in follow-up
				if choice.Delta.Refusal != "" {
					refusalContent := choice.Delta.Refusal
					respContent.WriteString(refusalContent)

					if callback != nil {
						event := chat.StreamEvent{
							Type:    chat.StreamEventTypeContent,
							Content: refusalContent,
						}
						if err := callback(event); err != nil {
							return chat.Message{}, err
						}
					}

					c.logger.Debug("follow-up refusal content", "content", refusalContent)
				}

				// Check for tool calls
				if len(choice.Delta.ToolCalls) > 0 {
					for _, tc := range choice.Delta.ToolCalls {
						idx := int(tc.Index)

						// Ensure we have enough space in toolCalls slice
						for len(toolCalls) <= idx {
							toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCall{})
						}

						if tc.ID != "" {
							// Starting a new tool call
							toolCalls[idx].ID = tc.ID
							toolCallArgs[idx] = strings.Builder{}
						}

						if tc.Function.Name != "" {
							toolCalls[idx].Function.Name = tc.Function.Name
						}
						if tc.Function.Arguments != "" {
							builder := toolCallArgs[idx]
							builder.WriteString(tc.Function.Arguments)
							toolCallArgs[idx] = builder
							toolCalls[idx].Function.Arguments = builder.String()
						}

						// Emit tool call event only once per index when arguments are valid JSON
						if callback != nil && !toolCallEmitted.Contains(idx) && toolCalls[idx].ID != "" && toolCalls[idx].Function.Name != "" && isValidJSON(toolCalls[idx].Function.Arguments) {
							toolCallEvent := chat.StreamEvent{
								Type: chat.StreamEventTypeToolCall,
								ToolCalls: []chat.ToolCall{
									{
										ID:        toolCalls[idx].ID,
										Name:      toolCalls[idx].Function.Name,
										Arguments: json.RawMessage(toolCalls[idx].Function.Arguments),
									},
								},
							}
							toolCallEmitted.Add(idx)
							if err := callback(toolCallEvent); err != nil {
								return chat.Message{}, err
							}
						}
					}
				}

				// Check for regular content
				if choice.Delta.Content != "" {
					content := choice.Delta.Content
					respContent.WriteString(content)

					// Call the callback with the content event
					if callback != nil {
						event := chat.StreamEvent{
							Type:    chat.StreamEventTypeContent,
							Content: content,
						}
						if err := callback(event); err != nil {
							return chat.Message{}, err
						}
					}
				}
			}
		}

		if err := followUpStream.Err(); err != nil {
			return chat.Message{}, fmt.Errorf("follow-up streaming error: %w", err)
		}

		// If we got more tool calls, continue the loop
		if len(toolCalls) > 0 {
			c.logger.Debug("got more tool calls", "count", len(toolCalls))
			isFirstIteration = false
			continue
		}

		// No more tool calls, we have the final response
		finalMsg := chat.AssistantMessage(respContent.String())

		// Log if content is empty
		if finalMsg.GetText() == "" {
			c.logger.Warn("final response after tool execution has empty content")
		}

		// Update history with the final response
		c.state.AppendMessages([]chat.Message{finalMsg}, &lastUsage)

		return finalMsg, nil
	}

	// This should never be reached since the loop continues until no tool calls
	return chat.Message{}, fmt.Errorf("unexpected end of tool call processing")
}

// mcpToOpenAITool converts an MCP tool definition to OpenAI format
func (c *chatClient) mcpToOpenAITool(mcpDef chat.ToolDef) (openai.ChatCompletionToolParam, error) {
	// Parse the MCP JSON schema to extract the inputSchema
	var mcp struct {
		InputSchema json.RawMessage `json:"inputSchema"`
	}

	jsonSchema := mcpDef.MCPJsonSchema()
	if err := json.Unmarshal([]byte(jsonSchema), &mcp); err != nil {
		return openai.ChatCompletionToolParam{}, fmt.Errorf("failed to parse MCP definition: %w", err)
	}

	// Convert inputSchema to FunctionParameters
	var parameters shared.FunctionParameters
	if len(mcp.InputSchema) > 0 {
		// The inputSchema is already in JSON Schema format, which OpenAI expects
		if err := json.Unmarshal(mcp.InputSchema, &parameters); err != nil {
			return openai.ChatCompletionToolParam{}, fmt.Errorf("failed to parse input schema: %w", err)
		}
	}

	return openai.ChatCompletionToolParam{
		Function: shared.FunctionDefinitionParam{
			Name:        mcpDef.Name(),
			Description: param.NewOpt(mcpDef.Description()),
			Parameters:  parameters,
		},
	}, nil
}

// handleToolCalls processes tool calls from the model and returns tool results
func (c *chatClient) handleToolCalls(ctx context.Context, toolCalls []openai.ChatCompletionMessageToolCall, callback chat.StreamCallback) ([]chat.ToolResult, error) {
	if len(toolCalls) == 0 {
		return nil, nil
	}

	var chatResults []chat.ToolResult

	for _, toolCall := range toolCalls {
		result, err := c.tools.Execute(ctx, toolCall.Function.Name, toolCall.Function.Arguments)

		// Emit tool result event if callback is provided
		toolResult := chat.ToolResult{
			ToolCallID: toolCall.ID,
			Name:       toolCall.Function.Name,
		}

		if err != nil {
			toolResult.Error = err.Error()
		} else {
			toolResult.Content = result
		}

		if callback != nil {
			toolResultEvent := chat.StreamEvent{
				Type:        chat.StreamEventTypeToolResult,
				ToolResults: []chat.ToolResult{toolResult},
			}
			if callbackErr := callback(toolResultEvent); callbackErr != nil {
				return nil, fmt.Errorf("callback error: %w", callbackErr)
			}
		}

		chatResults = append(chatResults, toolResult)
	}

	return chatResults, nil
}

func openaiToolCallToChat(tc openai.ChatCompletionMessageToolCall) chat.ToolCall {
	var args json.RawMessage
	if tc.Function.Arguments != "" {
		args = json.RawMessage(tc.Function.Arguments)
	}
	return chat.ToolCall{
		ID:        tc.ID,
		Name:      tc.Function.Name,
		Arguments: args,
	}
}

func (c *chatClient) History() (systemPrompt string, msgs []chat.Message) {
	return c.state.History()
}

// TokenUsage returns token usage for both the last message and cumulative session
func (c *chatClient) TokenUsage() (chat.TokenUsage, error) {
	return c.state.TokenUsage()
}

// MaxTokens returns the maximum token limit for the model
func (c *chatClient) MaxTokens() int {
	return c.maxTokens
}

// RegisterTool registers a tool that can be called by the LLM
func (c *chatClient) RegisterTool(tool chat.Tool) error {
	return c.tools.Register(tool)
}

// DeregisterTool removes a tool by name
func (c *chatClient) DeregisterTool(name string) {
	c.tools.Deregister(name)
}

// ListTools returns the names of all registered tools
func (c *chatClient) ListTools() []string {
	return c.tools.List()
}

// isValidJSON returns true if s is a complete valid JSON value
func isValidJSON(s string) bool {
	if strings.TrimSpace(s) == "" {
		return false
	}
	var v interface{}
	return json.Unmarshal([]byte(s), &v) == nil
}

// messageToOpenAI converts a chat.Message to OpenAI message parameters.
// This function handles all message types (User, Assistant, Tool) and content types
// (text, tool calls, tool results) using the unified Contents array approach.
//
// IMPORTANT INVARIANTS for OpenAI:
// - Tool calls must be in Assistant role messages
// - Tool results must be in separate Tool role messages
// - System messages are a distinct role and should not contain tool content
// - OpenAI uses "tool" role for tool results, not "user" like Claude
func messageToOpenAI(msg chat.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	if len(msg.Contents) == 0 {
		return nil, fmt.Errorf("message has no contents")
	}

	switch msg.Role {
	case chat.UserRole:
		// User messages can only contain text content
		text := extractText(msg)
		if text == "" {
			return nil, fmt.Errorf("user message has no text content")
		}
		return []openai.ChatCompletionMessageParamUnion{openai.UserMessage(text)}, nil

	case chat.AssistantRole:
		// Assistant messages can contain text and/or tool calls
		assistant := openai.ChatCompletionAssistantMessageParam{}

		// Extract text content if present
		if text := extractText(msg); text != "" {
			assistant.Content.OfString = param.NewOpt(text)
		}

		// Extract tool calls if present
		toolCalls := extractToolCalls(msg)
		if len(toolCalls) > 0 {
			assistant.ToolCalls = buildOpenAIToolCallParams(toolCalls)
		}

		// Validate that we have at least some content
		if assistant.Content.OfString.Value == "" && len(assistant.ToolCalls) == 0 {
			return nil, fmt.Errorf("assistant message has no valid content")
		}

		return []openai.ChatCompletionMessageParamUnion{{OfAssistant: &assistant}}, nil

	case chat.ToolRole:
		// Tool role messages contain tool results
		// OpenAI requires separate messages for each tool result
		toolResults := extractToolResults(msg)
		if len(toolResults) == 0 {
			// Fallback: if no structured tool results but text present
			if text := extractText(msg); text != "" {
				return []openai.ChatCompletionMessageParamUnion{
					openai.ToolMessage(text, ""),
				}, nil
			}
			return nil, fmt.Errorf("tool message has no tool results")
		}

		// Convert each tool result to a separate message
		msgs := make([]openai.ChatCompletionMessageParamUnion, 0, len(toolResults))
		for _, tr := range toolResults {
			content := tr.Content
			if tr.Error != "" {
				content = common.FormatToolErrorJSON(tr.Error)
			}
			if content == "" {
				content = "{}"
			}
			msgs = append(msgs, openai.ToolMessage(content, tr.ToolCallID))
		}
		return msgs, nil

	case "system":
		// System messages can only contain text
		text := extractText(msg)
		if text == "" {
			return nil, fmt.Errorf("system message has no text content")
		}
		return []openai.ChatCompletionMessageParamUnion{openai.SystemMessage(text)}, nil

	default:
		return nil, fmt.Errorf("unknown message role: %s", msg.Role)
	}
}

// extractText concatenates all text content from a message, including system reminders.
func extractText(msg chat.Message) string {
	var text string
	for _, content := range msg.Contents {
		// Include both Text and SystemReminder fields
		if content.Text != "" {
			if text != "" {
				text += "\n"
			}
			text += content.Text
		}
		if content.SystemReminder != "" {
			if text != "" {
				text += "\n"
			}
			text += content.SystemReminder
		}
	}
	return text
}

// extractToolCalls collects all tool calls from a message.
func extractToolCalls(msg chat.Message) []chat.ToolCall {
	var calls []chat.ToolCall
	for _, content := range msg.Contents {
		if content.ToolCall != nil {
			calls = append(calls, *content.ToolCall)
		}
	}
	return calls
}

// extractToolResults collects all tool results from a message.
func extractToolResults(msg chat.Message) []chat.ToolResult {
	var results []chat.ToolResult
	for _, content := range msg.Contents {
		if content.ToolResult != nil {
			results = append(results, *content.ToolResult)
		}
	}
	return results
}

// buildOpenAIToolCallParams converts chat.ToolCall array to OpenAI tool call params.
func buildOpenAIToolCallParams(toolCalls []chat.ToolCall) []openai.ChatCompletionMessageToolCallParam {
	params := make([]openai.ChatCompletionMessageToolCallParam, len(toolCalls))
	for i, tc := range toolCalls {
		params[i] = openai.ChatCompletionMessageToolCallParam{
			ID: tc.ID,
			Function: openai.ChatCompletionMessageToolCallFunctionParam{
				Name:      tc.Name,
				Arguments: string(tc.Arguments),
			},
		}
	}
	return params
}

// messagesToOpenAI converts a slice of chat messages to OpenAI message parameters.
// This handles the conversion of multiple messages, properly expanding tool results
// which may result in multiple OpenAI messages for a single chat message.
func messagesToOpenAI(msgs []chat.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	var result []openai.ChatCompletionMessageParamUnion

	for i, msg := range msgs {
		converted, err := messageToOpenAI(msg)
		if err != nil {
			return nil, fmt.Errorf("converting message %d: %w", i, err)
		}
		result = append(result, converted...)
	}

	return result, nil
}
