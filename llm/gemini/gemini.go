package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"google.golang.org/genai"

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
var logger = logging.Logger().With("provider", "gemini")

type client struct {
	genaiClient *genai.Client
	modelName   string
	baseURL     string
	headers     map[string]string // Custom HTTP headers
	logger      *slog.Logger
}

var _ chat.Client = &client{}

// generateFunctionCallID generates a unique ID for function calls
func generateFunctionCallID() string {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	return fmt.Sprintf("gemini_%d_%d", time.Now().Unix(), rng.Intn(1000000))
}

type Option func(*client)

func WithModel(modelName string) Option {
	return func(c *client) {
		c.modelName = strings.TrimSpace(modelName)
	}
}

func WithBaseURL(baseURL string) Option {
	return func(c *client) {
		c.baseURL = baseURL
	}
}

func WithHeaders(headers map[string]string) Option {
	return func(c *client) {
		c.headers = headers
	}
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

// NewClient returns a chat client that can begin chat sessions with Google's Gemini API.
func NewClient(apiKey string, opts ...Option) (chat.Client, error) {
	c := &client{
		logger: logger,
	}

	for _, opt := range opts {
		opt(c)
	}

	if c.modelName == "" {
		return nil, fmt.Errorf("WithModel is a required option")
	}

	if apiKey == "" {
		return nil, fmt.Errorf("API key is required for Gemini API")
	}

	ctx := context.Background()

	// Build client config
	config := &genai.ClientConfig{
		APIKey: apiKey,
	}

	// Add custom headers if provided
	if len(c.headers) > 0 {
		httpHeaders := make(http.Header)
		for key, value := range c.headers {
			httpHeaders.Set(key, value)
		}
		config.HTTPOptions.Headers = httpHeaders
	}

	genaiClient, err := genai.NewClient(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}

	c.genaiClient = genaiClient

	return c, nil
}

// getSystemReminderText retrieves and executes system reminder function if present
func getSystemReminderText(ctx context.Context) string {
	if reminderFunc := chat.GetSystemReminder(ctx); reminderFunc != nil {
		return reminderFunc()
	}
	return ""
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

var modelLimits = []chat.ModelTokenLimits{
	{Model: "gemini-2.5-pro", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 65536}},
	{Model: "gemini-2.5-flash", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 65536}},
	{Model: "gemini-2.5-flash-lite", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 65536}},
	{Model: "gemini-2.0-flash", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 8192}},
	{Model: "gemini-2.0-flash-lite", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 8192}},
	{Model: "gemini-1.5-pro", TokenLimits: chat.TokenLimits{Context: 2097152, Output: 8192}},
	{Model: "gemini-1.5-flash", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 8192}},
	{Model: "gemini-1.5-flash-8b", TokenLimits: chat.TokenLimits{Context: 1048576, Output: 8192}},
}

// getModelMaxTokens returns the maximum token limit for known models
func getModelMaxTokens(model string) int {
	modelLower := strings.ToLower(model)

	for _, m := range modelLimits {
		if strings.HasPrefix(modelLower, m.Model) {
			return m.TokenLimits.Output
		}
	}

	logger.Warn("unknown model, using conservative default output token limit", "model", model, "default_limit", 128000)
	return 128000
}

type chatClient struct {
	client
	state     *common.State
	tools     *common.Tools
	maxTokens int
}

func (c *chatClient) Message(ctx context.Context, msg chat.Message, opts ...chat.Option) (chat.Message, error) {
	// Apply options to get callback if provided
	appliedOpts := chat.ApplyOptions(opts...)
	callback := appliedOpts.StreamingCb
	reqOpts := chat.ApplyOptions(opts...)

	// Build content for all messages
	var contents []*genai.Content

	// Snapshot history with minimal lock
	systemPrompt, history := c.state.Snapshot()

	// Add system instruction as first content if present
	if systemPrompt != "" {
		systemText := systemPrompt
		// Handle response format if provided
		if reqOpts.ResponseFormat != nil && reqOpts.ResponseFormat.Schema != nil {
			systemText += fmt.Sprintf("\n\nYou must respond with valid JSON that conforms to the schema named: %s", reqOpts.ResponseFormat.Name)
		}
		contents = append(contents, &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{Text: systemText},
			},
		})
	}

	// Add history messages using the new converter
	for _, m := range history {
		converted, err := messageToGemini(m)
		if err != nil {
			// Skip messages that can't be converted (e.g., system messages are handled separately)
			continue
		}
		contents = append(contents, converted...)
	}

	// Add current message with system reminder prepended if present
	// This message (with system reminder) will be persisted for audit trail
	msgWithReminder := withPrependedSystemReminder(ctx, msg)
	converted, err := messageToGemini(msgWithReminder)
	if err != nil {
		return chat.Message{}, fmt.Errorf("converting current message: %w", err)
	}
	contents = append(contents, converted...)

	// Configure generation settings
	config := &genai.GenerateContentConfig{}

	// Apply base URL if configured
	if c.baseURL != "" {
		config.HTTPOptions = &genai.HTTPOptions{
			BaseURL: c.baseURL,
		}
	}

	if reqOpts.Temperature != nil {
		temp := float32(*reqOpts.Temperature)
		config.Temperature = &temp
	}

	if reqOpts.MaxTokens > 0 {
		config.MaxOutputTokens = int32(reqOpts.MaxTokens)
	}

	// Add tools if registered
	allTools := c.tools.GetAll()
	if len(allTools) > 0 {
		tools := make([]*genai.Tool, 0, 1)
		functionDeclarations := make([]*genai.FunctionDeclaration, 0, len(allTools))
		for _, tool := range allTools {
			funcDecl, err := c.mcpToGeminiFunctionDeclaration(tool.Definition)
			if err != nil {
				return chat.Message{}, fmt.Errorf("failed to convert tool: %w", err)
			}
			functionDeclarations = append(functionDeclarations, funcDecl)
		}
		// Create a single Tool with all function declarations
		tools = append(tools, &genai.Tool{
			FunctionDeclarations: functionDeclarations,
		})
		config.Tools = tools
	}

	// Stream content
	c.logger.Debug("starting stream", "model", c.modelName, "has_tools", len(allTools) > 0)
	stream := c.genaiClient.Models.GenerateContentStream(ctx, c.modelName, contents, config)

	var respContent strings.Builder
	var functionCalls []*genai.FunctionCall
	chunkCount := 0
	for chunk, err := range stream {
		if err != nil {
			return chat.Message{}, fmt.Errorf("streaming error: %w", err)
		}
		if chunk == nil {
			continue
		}
		chunkCount++
		c.logger.Debug("chunk received", "chunk_num", chunkCount, "candidates", len(chunk.Candidates))

		// Extract text and function calls from chunk
		for _, candidate := range chunk.Candidates {
			if candidate.Content != nil {
				for _, part := range candidate.Content.Parts {
					if part.Text != "" {
						content := part.Text
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
					// Check for function calls
					if part.FunctionCall != nil {
						// Generate ID if not present
						if part.FunctionCall.ID == "" {
							part.FunctionCall.ID = generateFunctionCallID()
						}
						functionCalls = append(functionCalls, part.FunctionCall)

						// Log function call detection
						argsJSON, _ := json.Marshal(part.FunctionCall.Args)
						c.logger.Debug("function call detected", "id", part.FunctionCall.ID, "name", part.FunctionCall.Name, "args", string(argsJSON))

						// Emit tool call event
						if callback != nil {
							// Convert arguments to JSON (already done above)
							toolCallEvent := chat.StreamEvent{
								Type: chat.StreamEventTypeToolCall,
								ToolCalls: []chat.ToolCall{
									{
										ID:        part.FunctionCall.ID,
										Name:      part.FunctionCall.Name,
										Arguments: json.RawMessage(argsJSON),
									},
								},
							}
							if err := callback(toolCallEvent); err != nil {
								return chat.Message{}, err
							}
						}
					}
				}
			}
			// Extract token usage if available
			if chunk.UsageMetadata != nil {
				usage := chat.TokenUsageDetails{
					InputTokens:  int(chunk.UsageMetadata.PromptTokenCount),
					OutputTokens: int(chunk.UsageMetadata.CandidatesTokenCount),
					TotalTokens:  int(chunk.UsageMetadata.TotalTokenCount),
					CachedTokens: int(chunk.UsageMetadata.CachedContentTokenCount),
				}

				// Update usage
				c.state.UpdateUsage(usage)

				// Log token usage
				totalUsage, _ := c.state.TokenUsage()
				c.logger.Debug("usage metadata", "input", usage.InputTokens, "output", usage.OutputTokens, "total", usage.TotalTokens, "cached", usage.CachedTokens,
					"cumulative_input", totalUsage.Cumulative.InputTokens, "cumulative_output", totalUsage.Cumulative.OutputTokens, "cumulative_total", totalUsage.Cumulative.TotalTokens)
			}
		}
	}

	// Log stream completion
	c.logger.Debug("stream completed", "has_function_calls", len(functionCalls) > 0, "content_length", respContent.Len())

	// Handle tool calls with multiple rounds if needed
	if len(functionCalls) > 0 {
		return c.handleToolCallRounds(ctx, msgWithReminder, functionCalls, reqOpts, callback)
	}

	respMsg := chat.AssistantMessage(respContent.String())

	// Update history
	// Persist the message WITH system reminder for complete audit trail
	c.state.AppendMessages([]chat.Message{msgWithReminder, respMsg}, nil)

	return respMsg, nil
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

// jsonSchemaToGeminiSchema recursively converts a JSON Schema object to Gemini Schema format.
// It handles all basic types, arrays with items, objects with properties and required fields,
// and schema attributes like description and enum.
func jsonSchemaToGeminiSchema(schemaMap map[string]interface{}) (*genai.Schema, error) {
	schema := &genai.Schema{}

	if typeStr, ok := schemaMap["type"].(string); ok {
		switch typeStr {
		case "string":
			schema.Type = genai.TypeString
		case "integer":
			schema.Type = genai.TypeInteger
		case "number":
			schema.Type = genai.TypeNumber
		case "boolean":
			schema.Type = genai.TypeBoolean
		case "array":
			schema.Type = genai.TypeArray
			if items, ok := schemaMap["items"].(map[string]interface{}); ok {
				itemSchema, err := jsonSchemaToGeminiSchema(items)
				if err != nil {
					return nil, fmt.Errorf("failed to convert array items schema: %w", err)
				}
				schema.Items = itemSchema
			}
		case "object":
			schema.Type = genai.TypeObject
			if props, ok := schemaMap["properties"].(map[string]interface{}); ok {
				schema.Properties = make(map[string]*genai.Schema)
				for propName, propValue := range props {
					if propMap, ok := propValue.(map[string]interface{}); ok {
						propSchema, err := jsonSchemaToGeminiSchema(propMap)
						if err != nil {
							return nil, fmt.Errorf("failed to convert property %q: %w", propName, err)
						}
						schema.Properties[propName] = propSchema
					}
				}
			}
			if required, ok := schemaMap["required"].([]interface{}); ok {
				requiredFields := make([]string, 0, len(required))
				for _, field := range required {
					if fieldName, ok := field.(string); ok {
						requiredFields = append(requiredFields, fieldName)
					}
				}
				schema.Required = requiredFields
			}
		}
	}

	if desc, ok := schemaMap["description"].(string); ok {
		schema.Description = desc
	}

	if enum, ok := schemaMap["enum"].([]interface{}); ok {
		enumStrs := make([]string, 0, len(enum))
		for _, e := range enum {
			if eStr, ok := e.(string); ok {
				enumStrs = append(enumStrs, eStr)
			}
		}
		schema.Enum = enumStrs
	}

	return schema, nil
}

// mcpToGeminiFunctionDeclaration converts an MCP tool definition to Gemini FunctionDeclaration format
func (c *chatClient) mcpToGeminiFunctionDeclaration(mcpDef chat.ToolDef) (*genai.FunctionDeclaration, error) {
	// Parse the MCP JSON schema to extract the inputSchema
	var mcp struct {
		InputSchema json.RawMessage `json:"inputSchema"`
	}

	jsonSchema := mcpDef.MCPJsonSchema()
	if err := json.Unmarshal([]byte(jsonSchema), &mcp); err != nil {
		return nil, fmt.Errorf("failed to parse MCP definition: %w", err)
	}

	// Convert inputSchema to Gemini Schema format using recursive converter
	var parameters *genai.Schema
	if len(mcp.InputSchema) > 0 {
		var schemaMap map[string]interface{}
		if err := json.Unmarshal(mcp.InputSchema, &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to parse input schema: %w", err)
		}

		var err error
		parameters, err = jsonSchemaToGeminiSchema(schemaMap)
		if err != nil {
			return nil, fmt.Errorf("failed to convert input schema: %w", err)
		}
	}

	return &genai.FunctionDeclaration{
		Name:        mcpDef.Name(),
		Description: mcpDef.Description(),
		Parameters:  parameters,
	}, nil
}

// handleToolCallRounds handles potentially multiple rounds of tool calls
func (c *chatClient) handleToolCallRounds(ctx context.Context, initialMsg chat.Message, initialFunctionCalls []*genai.FunctionCall, reqOpts chat.Options, callback chat.StreamCallback) (chat.Message, error) {
	c.logger.Debug("starting tool call rounds", "initial_function_count", len(initialFunctionCalls))

	// Keep track of all messages for the conversation
	var msgs []*genai.Content
	var systemPrompt string

	// Build initial conversation with system prompt and history
	// Snapshot history with minimal lock
	systemPrompt, history := c.state.Snapshot()

	if systemPrompt != "" {
		msgs = append(msgs, &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{Text: systemPrompt},
			},
		})
	}

	// Add history messages using the new converter
	for _, m := range history {
		converted, err := messageToGemini(m)
		if err != nil {
			// Skip messages that can't be converted
			continue
		}
		// Filter out nil results and empty contents
		for _, content := range converted {
			if content != nil && len(content.Parts) > 0 {
				msgs = append(msgs, content)
			}
		}
	}

	// Add the initial user message using the converter
	converted, err := messageToGemini(initialMsg)
	if err != nil {
		return chat.Message{}, fmt.Errorf("converting initial message: %w", err)
	}
	msgs = append(msgs, converted...)

	// Persist the user message before tool execution to maintain chronological ordering
	c.state.AppendMessages([]chat.Message{initialMsg}, nil)

	// Process tool calls in a loop until we get a final response
	functionCalls := initialFunctionCalls

	for len(functionCalls) > 0 {
		c.logger.Debug("processing function calls", "count", len(functionCalls))
		for i, fc := range functionCalls {
			argsJSON, _ := json.Marshal(fc.Args)
			c.logger.Debug("function call", "index", i+1, "id", fc.ID, "name", fc.Name, "args", string(argsJSON))
		}

		// Execute tool calls
		functionResults, _, err := c.handleFunctionCalls(ctx, functionCalls, callback)
		if err != nil {
			return chat.Message{}, fmt.Errorf("failed to execute function calls: %w", err)
		}

		assistantParts := make([]*genai.Part, len(functionCalls))
		chatToolCalls := make([]chat.ToolCall, len(functionCalls))
		for i, fc := range functionCalls {
			assistantParts[i] = &genai.Part{
				FunctionCall: fc,
			}
			chatToolCalls[i] = geminiFunctionCallToChat(fc)
		}

		msgs = append(msgs, &genai.Content{
			Role:  "model",
			Parts: assistantParts,
		})

		// Note: We don't store intermediate tool-calling messages in conversation history
		// The final conversation will only include the initial user message and final assistant response

		// Add function results to messages (only if we have actual results)
		if len(functionResults) > 0 {
			// Build parts with system reminder first, then function results
			resultParts := []*genai.Part{}
			if reminder := getSystemReminderText(ctx); reminder != "" {
				resultParts = append(resultParts, &genai.Part{Text: reminder})
			}
			for _, fr := range functionResults {
				resultParts = append(resultParts, &genai.Part{
					FunctionResponse: fr,
				})
			}

			msgs = append(msgs, &genai.Content{
				Role:  "user",
				Parts: resultParts,
			})
		}

		// Make another API call with tool results
		followUpConfig := &genai.GenerateContentConfig{}

		// Apply base URL if configured
		if c.baseURL != "" {
			followUpConfig.HTTPOptions = &genai.HTTPOptions{
				BaseURL: c.baseURL,
			}
		}

		if reqOpts.Temperature != nil {
			temp := float32(*reqOpts.Temperature)
			followUpConfig.Temperature = &temp
		}
		if reqOpts.MaxTokens > 0 {
			followUpConfig.MaxOutputTokens = int32(reqOpts.MaxTokens)
		}

		// Add tools again for follow-up after tool execution
		allTools := c.tools.GetAll()
		if len(allTools) > 0 {
			tools := make([]*genai.Tool, 0, 1)
			functionDeclarations := make([]*genai.FunctionDeclaration, 0, len(allTools))
			for _, tool := range allTools {
				funcDecl, err := c.mcpToGeminiFunctionDeclaration(tool.Definition)
				if err != nil {
					// Skip this tool on error
					continue
				}
				functionDeclarations = append(functionDeclarations, funcDecl)
			}
			// Create a single Tool with all function declarations
			tools = append(tools, &genai.Tool{
				FunctionDeclarations: functionDeclarations,
			})
			followUpConfig.Tools = tools
		}

		// Create a new stream for the follow-up request
		followUpStream := c.genaiClient.Models.GenerateContentStream(ctx, c.modelName, msgs, followUpConfig)

		// Process the follow-up stream
		var respContent strings.Builder
		functionCalls = nil // Reset for next round
		followUpChunkCount := 0

		for chunk, err := range followUpStream {
			if err != nil {
				return chat.Message{}, fmt.Errorf("follow-up streaming error: %w", err)
			}
			if chunk == nil {
				continue
			}
			followUpChunkCount++
			c.logger.Debug("follow-up chunk received", "chunk_num", followUpChunkCount, "candidates", len(chunk.Candidates))

			for _, candidate := range chunk.Candidates {
				if candidate.Content != nil {
					for _, part := range candidate.Content.Parts {
						// Check for function calls
						if part.FunctionCall != nil {
							// Generate ID if not present
							if part.FunctionCall.ID == "" {
								part.FunctionCall.ID = generateFunctionCallID()
							}
							functionCalls = append(functionCalls, part.FunctionCall)

							// Emit tool call event
							if callback != nil {
								// Convert arguments to JSON
								argsJSON, _ := json.Marshal(part.FunctionCall.Args)
								toolCallEvent := chat.StreamEvent{
									Type: chat.StreamEventTypeToolCall,
									ToolCalls: []chat.ToolCall{
										{
											ID:        part.FunctionCall.ID,
											Name:      part.FunctionCall.Name,
											Arguments: json.RawMessage(argsJSON),
										},
									},
								}
								if err := callback(toolCallEvent); err != nil {
									return chat.Message{}, err
								}
							}
						}

						// Check for regular content
						if part.Text != "" {
							content := part.Text
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
				// Extract token usage if available
				if chunk.UsageMetadata != nil {
					usage := chat.TokenUsageDetails{
						InputTokens:  int(chunk.UsageMetadata.PromptTokenCount),
						OutputTokens: int(chunk.UsageMetadata.CandidatesTokenCount),
						TotalTokens:  int(chunk.UsageMetadata.TotalTokenCount),
						CachedTokens: int(chunk.UsageMetadata.CachedContentTokenCount),
					}

					// Update usage
					c.state.UpdateUsage(usage)

					// Log token usage
					totalUsage, _ := c.state.TokenUsage()
					c.logger.Debug("follow-up usage metadata", "input", usage.InputTokens, "output", usage.OutputTokens, "total", usage.TotalTokens,
						"cumulative_input", totalUsage.Cumulative.InputTokens, "cumulative_output", totalUsage.Cumulative.OutputTokens, "cumulative_total", totalUsage.Cumulative.TotalTokens)
				}
			}
		}

		// If we got more function calls, continue the loop
		if len(functionCalls) > 0 {
			c.logger.Debug("got more function calls, continuing", "count", len(functionCalls))
			continue
		}

		// No more function calls, we have the final response
		c.logger.Debug("no more function calls, returning final response", "content_length", len(respContent.String()))

		finalMsg := chat.AssistantMessage(respContent.String())

		// Warn if final response is empty
		if respContent.Len() == 0 {
			c.logger.Warn("final response has no content")
		}

		// Update history with final assistant response (user message already persisted)
		c.state.AppendMessages([]chat.Message{finalMsg}, nil)

		return finalMsg, nil
	}

	// This should never be reached since the loop continues until no function calls
	c.logger.Error("unexpected end of function call processing", "initial_function_count", len(initialFunctionCalls))
	return chat.Message{}, fmt.Errorf("unexpected end of function call processing")
}

func geminiFunctionCallToChat(fc *genai.FunctionCall) chat.ToolCall {
	var args json.RawMessage
	if fc != nil && fc.Args != nil {
		if data, err := json.Marshal(fc.Args); err == nil {
			args = data
		}
	}
	return chat.ToolCall{
		ID:        fc.ID,
		Name:      fc.Name,
		Arguments: args,
	}
}

// handleFunctionCalls processes function calls from the model and returns function results
func (c *chatClient) handleFunctionCalls(ctx context.Context, functionCalls []*genai.FunctionCall, callback chat.StreamCallback) ([]*genai.FunctionResponse, []chat.ToolResult, error) {
	if len(functionCalls) == 0 {
		return nil, nil, nil
	}

	var functionResults []*genai.FunctionResponse
	var chatResults []chat.ToolResult

	for _, fc := range functionCalls {
		argsJSON, err := json.Marshal(fc.Args)
		if err != nil {
			errorResponse := map[string]interface{}{
				"error": fmt.Sprintf("Failed to marshal function arguments: %v", err),
			}
			functionResults = append(functionResults, &genai.FunctionResponse{
				ID:       fc.ID,
				Name:     fc.Name,
				Response: errorResponse,
			})
			chatResults = append(chatResults, chat.ToolResult{
				ToolCallID: fc.ID,
				Name:       fc.Name,
				Error:      fmt.Sprintf("Failed to marshal function arguments: %v", err),
			})
			continue
		}

		resultStr, err := c.tools.Execute(ctx, fc.Name, string(argsJSON))

		toolResult := chat.ToolResult{
			ToolCallID: fc.ID,
			Name:       fc.Name,
		}

		if err != nil {
			toolResult.Error = err.Error()
			c.logger.Debug("tool execution failed", "name", fc.Name, "args", string(argsJSON), "error", err.Error())
		} else {
			toolResult.Content = resultStr
			c.logger.Debug("tool executed successfully", "name", fc.Name, "args", string(argsJSON), "result", resultStr)
		}

		if callback != nil {
			toolResultEvent := chat.StreamEvent{
				Type:        chat.StreamEventTypeToolResult,
				ToolResults: []chat.ToolResult{toolResult},
			}
			if callbackErr := callback(toolResultEvent); callbackErr != nil {
				return nil, nil, fmt.Errorf("callback error: %w", callbackErr)
			}
		}

		if err != nil {
			errorResponse := map[string]interface{}{
				"error": err.Error(),
			}
			functionResults = append(functionResults, &genai.FunctionResponse{
				ID:       fc.ID,
				Name:     fc.Name,
				Response: errorResponse,
			})
			chatResults = append(chatResults, toolResult)
			continue
		}

		var resultMap map[string]interface{}
		// Handle empty results specially to ensure valid response structure
		if resultStr == "" {
			resultMap = map[string]interface{}{
				"result": "success", // Provide a non-empty result for empty tool responses
			}
		} else if err := json.Unmarshal([]byte(resultStr), &resultMap); err != nil {
			// If not valid JSON, wrap as string result
			resultMap = map[string]interface{}{
				"result": resultStr,
			}
		}

		functionResults = append(functionResults, &genai.FunctionResponse{
			ID:       fc.ID,
			Name:     fc.Name,
			Response: resultMap,
		})
		chatResults = append(chatResults, toolResult)
	}

	return functionResults, chatResults, nil
}

// messageToGemini converts a chat.Message to Gemini Content format.
// This function handles all message types (User, Assistant, Tool) and content types
// (text, tool calls, tool results) using the unified Contents array approach.
//
// IMPORTANT INVARIANTS for Gemini:
// - Tool calls are FunctionCall parts within a Content
// - Tool results are FunctionResponse parts with "function" role
// - Assistant role maps to "model", User role maps to "user", Tool role maps to "function"
// - Multiple content types can be mixed within a single message's Parts array
// - Empty messages should return nil rather than empty Content objects
func messageToGemini(msg chat.Message) ([]*genai.Content, error) {
	if len(msg.Contents) == 0 {
		return nil, fmt.Errorf("message has no contents")
	}

	switch msg.Role {
	case chat.UserRole, "system":
		// User and system messages
		text := extractText(msg)
		if text == "" {
			return nil, fmt.Errorf("user/system message has no text content")
		}
		return []*genai.Content{{
			Role:  "user",
			Parts: []*genai.Part{{Text: text}},
		}}, nil

	case chat.AssistantRole:
		// Assistant messages can contain text and/or tool calls
		var parts []*genai.Part

		// Add text content if present
		if text := extractText(msg); text != "" {
			parts = append(parts, &genai.Part{Text: text})
		}

		// Add tool calls if present
		toolCalls := extractToolCalls(msg)
		for _, tc := range toolCalls {
			// Convert arguments from JSON to map
			var args map[string]any
			if len(tc.Arguments) > 0 {
				if err := json.Unmarshal(tc.Arguments, &args); err != nil {
					// If unmarshal fails, wrap as raw argument
					args = map[string]any{"raw": string(tc.Arguments)}
				}
			}

			// Ensure ID is set
			id := tc.ID
			if id == "" {
				id = generateFunctionCallID()
			}

			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   id,
					Name: tc.Name,
					Args: args,
				},
			})
		}

		// Skip empty assistant messages
		if len(parts) == 0 {
			return nil, nil
		}

		return []*genai.Content{{
			Role:  "model",
			Parts: parts,
		}}, nil

	case chat.ToolRole:
		// Tool role messages contain tool results
		// Gemini uses "function" role for these
		toolResults := extractToolResults(msg)
		if len(toolResults) == 0 {
			return nil, fmt.Errorf("tool message has no tool results")
		}

		// Convert tool results to function response parts
		parts := make([]*genai.Part, 0, len(toolResults))
		for _, tr := range toolResults {
			response := make(map[string]any)

			if tr.Error != "" {
				response["error"] = tr.Error
			} else if tr.Content != "" {
				// Try to unmarshal as JSON first
				if err := json.Unmarshal([]byte(tr.Content), &response); err != nil {
					// If not valid JSON, wrap as result
					response["result"] = tr.Content
				}
			} else {
				// Empty result - provide a non-empty response for Gemini
				response["result"] = "success"
			}

			parts = append(parts, &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					ID:       tr.ToolCallID,
					Name:     tr.Name,
					Response: response,
				},
			})
		}

		return []*genai.Content{{
			Role:  "function",
			Parts: parts,
		}}, nil

	default:
		// Unknown role, treat as user message
		text := extractText(msg)
		if text == "" {
			return nil, fmt.Errorf("message with unknown role has no text content")
		}
		return []*genai.Content{{
			Role:  "user",
			Parts: []*genai.Part{{Text: text}},
		}}, nil
	}
}

// extractText concatenates all text content from a message.
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

// messagesToGemini converts a slice of chat messages to Gemini Content format.
// This handles the conversion of multiple messages, filtering out any nil results.
func messagesToGemini(msgs []chat.Message) ([]*genai.Content, error) {
	var result []*genai.Content

	for i, msg := range msgs {
		converted, err := messageToGemini(msg)
		if err != nil {
			return nil, fmt.Errorf("converting message %d: %w", i, err)
		}
		// Filter out nil results (e.g., empty assistant messages)
		for _, content := range converted {
			if content != nil && len(content.Parts) > 0 {
				result = append(result, content)
			}
		}
	}

	return result, nil
}
