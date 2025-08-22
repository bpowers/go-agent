package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/llm/internal/common"
)

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
	debug        bool
	apiSet       bool // true if WithAPI was explicitly provided
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

func WithDebug(debug bool) Option {
	return func(c *client) {
		c.debug = debug
	}
}

// NewClient returns a chat client that can begin chat sessions with an LLM service that speaks
// the OpenAI chat completion API.
func NewClient(apiBase string, apiKey string, opts ...Option) (chat.Client, error) {
	c := &client{
		api: ChatCompletions, // default to chat completions
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

	c.openaiClient = openai.NewClient(clientOpts...)

	return c, nil
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
	log.Printf("[OpenAI] Unknown model %q; using conservative default output token limit", model)
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

	// For debug mode, wrap or create callback to stream to stderr
	if c.debug {
		origCallback := callback
		callback = func(event chat.StreamEvent) error {
			switch event.Type {
			case chat.StreamEventTypeContent:
				fmt.Fprint(os.Stderr, event.Content)
			case chat.StreamEventTypeThinking:
				if event.ThinkingStatus != nil && event.ThinkingStatus.IsThinking {
					fmt.Fprint(os.Stderr, "[Thinking...] ")
				}
			}
			if origCallback != nil {
				return origCallback(event)
			}
			return nil
		}
	}

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
	reqMsg := msg
	reqOpts := chat.ApplyOptions(opts...)

	// Snapshot state without holding lock during streaming
	systemPrompt, history := c.snapshotState()

	// Check if debug logging is enabled
	debugSSE := os.Getenv("GO_AGENT_DEBUG") == "1"

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

	// Add history messages
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

		inputItems = append(inputItems, responses.ResponseInputItemUnionParam{
			OfMessage: &responses.EasyInputMessageParam{
				Role:    role,
				Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(m.Content)},
			},
		})
	}

	// Add current message
	var currentRole responses.EasyInputMessageRole
	switch msg.Role {
	case chat.UserRole:
		currentRole = responses.EasyInputMessageRoleUser
	case chat.AssistantRole:
		currentRole = responses.EasyInputMessageRoleAssistant
	default:
		currentRole = responses.EasyInputMessageRoleDeveloper
	}

	inputItems = append(inputItems, responses.ResponseInputItemUnionParam{
		OfMessage: &responses.EasyInputMessageParam{
			Role:    currentRole,
			Content: responses.EasyInputMessageContentUnionParam{OfString: param.NewOpt(msg.Content)},
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

	if debugSSE {
		log.Printf("[OpenAI Responses API] Starting stream for model: %s\n", c.modelName)
	}

	// Create streaming response
	stream := c.openaiClient.Responses.NewStreaming(ctx, params)

	var respContent strings.Builder
	var reasoningContent strings.Builder
	var inReasoning bool
	eventCount := 0
	var lastUsage chat.TokenUsageDetails

	for stream.Next() {
		event := stream.Current()
		eventCount++

		if debugSSE {
			log.Printf("[OpenAI Responses API Event %d] Type: %s\n", eventCount, event.Type)
		}

		// Handle different event types
		switch event.Type {
		case "response.reasoning.delta", "response.thinking.delta", "response.reasoning_summary.delta", "response.reasoning_summary_text.delta":
			// Reasoning content is being streamed
			if !inReasoning && callback != nil {
				inReasoning = true
				// Emit thinking started event
				thinkingEvent := chat.StreamEvent{
					Type: chat.StreamEventTypeThinking,
					ThinkingStatus: &chat.ThinkingStatus{
						IsThinking: true,
					},
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
						Type:    chat.StreamEventTypeThinking,
						Content: deltaStr,
						ThinkingStatus: &chat.ThinkingStatus{
							IsThinking: true,
						},
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
						IsThinking: false,
						Summary:    reasoningContent.String(),
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
							IsThinking: false,
							Summary:    reasoningContent.String(),
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
				if debugSSE {
					log.Printf("[OpenAI Responses API] Usage from completed event - Input: %d, Output: %d, Total: %d\n",
						usage.InputTokens, usage.OutputTokens, usage.TotalTokens)
				}
			}
			if debugSSE {
				log.Printf("[OpenAI Responses API] Stream completed\n")
			}

		case "response.output_text.done":
			// Text output is complete
			if debugSSE {
				log.Printf("[OpenAI Responses API] Output text done\n")
			}

		case "error":
			// Handle error events
			if debugSSE {
				log.Printf("[OpenAI Responses API] Error event received\n")
			}
		}
	}

	if err := stream.Err(); err != nil {
		return chat.Message{}, fmt.Errorf("responses API streaming error: %w", err)
	}

	respMsg := chat.Message{
		Role:    chat.AssistantRole,
		Content: respContent.String(),
	}

	// Update history and usage under lock
	c.updateHistoryAndUsage([]chat.Message{reqMsg, respMsg}, lastUsage)

	return respMsg, nil
}

// messageStreamChatCompletions uses the standard Chat Completions API
func (c *chatClient) messageStreamChatCompletions(ctx context.Context, msg chat.Message, callback chat.StreamCallback, opts ...chat.Option) (chat.Message, error) {
	reqMsg := msg
	reqOpts := chat.ApplyOptions(opts...)

	// Snapshot state without holding lock during streaming
	systemPrompt, history := c.snapshotState()

	// Check if debug logging is enabled via environment variable
	debugSSE := os.Getenv("GO_AGENT_DEBUG") == "1"

	// Build message list
	var messages []openai.ChatCompletionMessageParamUnion

	// Add system prompt if present
	if systemPrompt != "" {
		messages = append(messages, openai.SystemMessage(systemPrompt))
	}

	// Add history
	for _, m := range history {
		switch m.Role {
		case chat.UserRole:
			messages = append(messages, openai.UserMessage(m.Content))
		case chat.AssistantRole:
			messages = append(messages, openai.AssistantMessage(m.Content))
		default:
			messages = append(messages, openai.SystemMessage(m.Content))
		}
	}

	// Add current message
	switch msg.Role {
	case chat.UserRole:
		messages = append(messages, openai.UserMessage(msg.Content))
	case chat.AssistantRole:
		messages = append(messages, openai.AssistantMessage(msg.Content))
	default:
		messages = append(messages, openai.SystemMessage(msg.Content))
	}

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
			toolParam, err := c.mcpToOpenAITool(tool.Definition)
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
		if debugSSE {
			log.Printf("[OpenAI] Reasoning effort: %s (feature may require additional API parameters)\n", reqOpts.ReasoningEffort)
		}
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
			if debugSSE {
				log.Printf("[OpenAI SSE] Usage chunk received - Input: %d, Output: %d, Total: %d\n",
					usage.InputTokens, usage.OutputTokens, usage.TotalTokens)
			}
		}

		// Debug logging for SSE responses
		if debugSSE {
			// Log the raw JSON of the chunk
			rawJSON := chunk.RawJSON()
			log.Printf("[OpenAI SSE Chunk %d] Model: %s, Raw: %s\n", chunkCount, c.modelName, rawJSON)

			// Log structured information about the chunk
			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]
				log.Printf("[OpenAI SSE Chunk %d] Choice[0] - Index: %d, FinishReason: %s, Delta.Role: %s, Delta.Content: %q\n",
					chunkCount, choice.Index, choice.FinishReason, choice.Delta.Role, choice.Delta.Content)

				// Check for extra fields that might contain reasoning content
				if len(choice.Delta.JSON.ExtraFields) > 0 {
					extraFieldsJSON, _ := json.Marshal(choice.Delta.JSON.ExtraFields)
					log.Printf("[OpenAI SSE Chunk %d] Delta ExtraFields: %s\n", chunkCount, string(extraFieldsJSON))
				}
				if len(choice.JSON.ExtraFields) > 0 {
					extraFieldsJSON, _ := json.Marshal(choice.JSON.ExtraFields)
					log.Printf("[OpenAI SSE Chunk %d] Choice ExtraFields: %s\n", chunkCount, string(extraFieldsJSON))
				}
			}
		}

		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]

			// Check for reasoning content in ExtraFields
			// OpenAI might use different field names: reasoning_content, reasoning, thinking_content, etc.
			reasoningFieldNames := []string{"reasoning_content", "reasoning", "thinking_content", "thinking"}
			var reasoningFieldRaw string
			var foundFieldName string

			for _, fieldName := range reasoningFieldNames {
				if field, exists := choice.Delta.JSON.ExtraFields[fieldName]; exists && field.Valid() {
					reasoningFieldRaw = field.Raw()
					foundFieldName = fieldName
					break
				}
			}

			if reasoningFieldRaw != "" {
				if debugSSE {
					log.Printf("[OpenAI SSE] Found %s field! Raw: %s\n", foundFieldName, reasoningFieldRaw)
				}
				// Try to extract the reasoning content
				var reasoningContent string
				if err := json.Unmarshal([]byte(reasoningFieldRaw), &reasoningContent); err == nil && reasoningContent != "" {
					if !inThinking && callback != nil {
						// Start thinking
						inThinking = true
						event := chat.StreamEvent{
							Type: chat.StreamEventTypeThinking,
							ThinkingStatus: &chat.ThinkingStatus{
								IsThinking: true,
							},
						}
						if err := callback(event); err != nil {
							return chat.Message{}, err
						}
					}

					thinkingContent.WriteString(reasoningContent)
					if callback != nil {
						// Stream thinking content
						event := chat.StreamEvent{
							Type:    chat.StreamEventTypeThinking,
							Content: reasoningContent,
							ThinkingStatus: &chat.ThinkingStatus{
								IsThinking: true,
							},
						}
						if err := callback(event); err != nil {
							return chat.Message{}, err
						}
					}
				}
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
								IsThinking: false,
								Summary:    thinkingContent.String(),
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

			// Check if stream is done and we were still thinking
			if choice.FinishReason != "" && inThinking && callback != nil {
				inThinking = false
				if thinkingContent.Len() > 0 {
					event := chat.StreamEvent{
						Type: chat.StreamEventTypeThinkingSummary,
						ThinkingStatus: &chat.ThinkingStatus{
							IsThinking: false,
							Summary:    thinkingContent.String(),
						},
					}
					if err := callback(event); err != nil {
						return chat.Message{}, err
					}
				}
			}
		}
	}

	if debugSSE {
		log.Printf("[OpenAI SSE] Stream completed. Total chunks: %d\n", chunkCount)
	}

	if err := stream.Err(); err != nil {
		// Check if the error is about unsupported temperature
		errStr := err.Error()
		if strings.Contains(errStr, "temperature") && strings.Contains(errStr, "does not support") && temperatureSet {
			// Retry without temperature
			if c.debug {
				fmt.Fprintf(os.Stderr, "\nTemperature not supported for model %s, retrying without temperature\n", c.modelName)
			}
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
					toolParam, err := c.mcpToOpenAITool(tool.Definition)
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
					if debugSSE {
						log.Printf("[OpenAI SSE Retry] Usage chunk received - Input: %d, Output: %d, Total: %d\n",
							usage.InputTokens, usage.OutputTokens, usage.TotalTokens)
					}
				}

				if debugSSE {
					log.Printf("[OpenAI SSE Retry Chunk %d] Model: %s\n", chunkCount, c.modelName)
				}

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
									Type: chat.StreamEventTypeThinking,
									ThinkingStatus: &chat.ThinkingStatus{
										IsThinking: true,
									},
								}
								if err := callback(event); err != nil {
									return chat.Message{}, err
								}
							}

							thinkingContent.WriteString(reasoningContent)
							if callback != nil {
								event := chat.StreamEvent{
									Type:    chat.StreamEventTypeThinking,
									Content: reasoningContent,
									ThinkingStatus: &chat.ThinkingStatus{
										IsThinking: true,
									},
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
										IsThinking: false,
										Summary:    thinkingContent.String(),
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
		return c.handleToolCallRounds(ctx, reqMsg, toolCalls, reqOpts, callback)
	}

	respMsg := chat.Message{
		Role:    chat.AssistantRole,
		Content: respContent.String(),
	}

	// Update history and usage under lock
	c.updateHistoryAndUsage([]chat.Message{reqMsg, respMsg}, lastUsage)

	// Update last usage
	if lastUsage.TotalTokens == 0 && debugSSE {
		log.Printf("[OpenAI] Warning: No token usage information received")
	}

	return respMsg, nil
}

// handleToolCallRounds handles potentially multiple rounds of tool calls
func (c *chatClient) handleToolCallRounds(ctx context.Context, initialMsg chat.Message, initialToolCalls []openai.ChatCompletionMessageToolCall, reqOpts chat.Options, callback chat.StreamCallback) (chat.Message, error) {
	// Check if debug logging is enabled
	debugSSE := os.Getenv("GO_AGENT_DEBUG") == "1"

	// Keep track of all messages for the conversation
	var msgs []openai.ChatCompletionMessageParamUnion

	// Build conversation messages and update history
	systemPrompt, history := c.state.Snapshot()
	if systemPrompt != "" {
		msgs = append(msgs, openai.SystemMessage(systemPrompt))
	}
	for _, m := range history {
		switch m.Role {
		case chat.UserRole:
			msgs = append(msgs, openai.UserMessage(m.Content))
		case chat.AssistantRole:
			msgs = append(msgs, openai.AssistantMessage(m.Content))
		default:
			msgs = append(msgs, openai.SystemMessage(m.Content))
		}
	}
	// Add the initial user message to history
	c.state.AppendMessages([]chat.Message{initialMsg}, nil)

	msgs = append(msgs, openai.UserMessage(initialMsg.Content))

	// Process tool calls in a loop until we get a final response
	toolCalls := initialToolCalls

	for len(toolCalls) > 0 {
		if debugSSE {
			log.Printf("[OpenAI] Processing %d tool calls", len(toolCalls))
		}

		// Execute tool calls
		toolResults, err := c.handleToolCalls(ctx, toolCalls)
		if err != nil {
			return chat.Message{}, fmt.Errorf("failed to execute tool calls: %w", err)
		}

		// Add assistant message with tool calls
		// Convert tool calls to the proper format
		toolCallParams := make([]openai.ChatCompletionMessageToolCallParam, len(toolCalls))
		for i, tc := range toolCalls {
			toolCallParams[i] = openai.ChatCompletionMessageToolCallParam{
				ID: tc.ID,
				Function: openai.ChatCompletionMessageToolCallFunctionParam{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}
		}

		// Create assistant message with tool calls
		assistantWithTools := openai.ChatCompletionAssistantMessageParam{
			ToolCalls: toolCallParams,
		}
		msgs = append(msgs, openai.ChatCompletionMessageParamUnion{
			OfAssistant: &assistantWithTools,
		})

		// Add tool results to messages
		msgs = append(msgs, toolResults...)

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
				toolParam, err := c.mcpToOpenAITool(tool.Definition)
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
			if debugSSE {
				log.Printf("[OpenAI] Got %d more tool calls", len(toolCalls))
			}
			continue
		}

		// No more tool calls, we have the final response
		finalMsg := chat.Message{
			Role:    chat.AssistantRole,
			Content: respContent.String(),
		}

		// Debug log if content is empty
		if debugSSE && finalMsg.Content == "" {
			log.Printf("[OpenAI] Warning: Final response after tool execution has empty content")
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
func (c *chatClient) handleToolCalls(ctx context.Context, toolCalls []openai.ChatCompletionMessageToolCall) ([]openai.ChatCompletionMessageParamUnion, error) {
	if len(toolCalls) == 0 {
		return nil, nil
	}

	var toolResults []openai.ChatCompletionMessageParamUnion

	for _, toolCall := range toolCalls {
		result, err := c.tools.Execute(ctx, toolCall.Function.Name, toolCall.Function.Arguments)
		if err != nil {
			// Tool not found or execution error, return error message
			toolResults = append(toolResults, openai.ToolMessage(
				fmt.Sprintf(`{"error": "%s"}`, err.Error()),
				toolCall.ID,
			))
			continue
		}

		// Add tool result message
		toolResults = append(toolResults, openai.ToolMessage(
			result,
			toolCall.ID,
		))
	}

	return toolResults, nil
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

// RegisterTool registers a tool with its MCP definition and handler function
func (c *chatClient) RegisterTool(def chat.ToolDef, fn func(context.Context, string) string) error {
	return c.tools.Register(def, fn)
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
