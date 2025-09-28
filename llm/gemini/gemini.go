package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"google.golang.org/genai"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/llm/internal/common"
)

type client struct {
	genaiClient *genai.Client
	modelName   string
	baseURL     string
	debug       bool
	headers     map[string]string // Custom HTTP headers
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

func WithDebug(debug bool) Option {
	return func(c *client) {
		c.debug = debug
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
	c := &client{}

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
	reqMsg := msg
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

	// Add current message using the new converter
	converted, err := messageToGemini(msg)
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
	stream := c.genaiClient.Models.GenerateContentStream(ctx, c.modelName, contents, config)

	var respContent strings.Builder
	var functionCalls []*genai.FunctionCall
	for chunk, err := range stream {
		if err != nil {
			return chat.Message{}, fmt.Errorf("streaming error: %w", err)
		}

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
			}
		}
	}

	// Handle tool calls with multiple rounds if needed
	if len(functionCalls) > 0 {
		return c.handleToolCallRounds(ctx, reqMsg, functionCalls, reqOpts, callback)
	}

	respMsg := chat.AssistantMessage(respContent.String())

	// Update history
	c.state.AppendMessages([]chat.Message{reqMsg, respMsg}, nil)

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

	// Convert inputSchema to Gemini Schema format
	var parameters *genai.Schema
	if len(mcp.InputSchema) > 0 {
		// Parse the JSON schema into a map first
		var schemaMap map[string]interface{}
		if err := json.Unmarshal(mcp.InputSchema, &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to parse input schema: %w", err)
		}

		// Create a Gemini Schema from the parsed JSON schema
		parameters = &genai.Schema{
			Type:       genai.TypeObject, // MCP tools typically use object schemas
			Properties: make(map[string]*genai.Schema),
		}

		// Extract properties if they exist
		if props, ok := schemaMap["properties"].(map[string]interface{}); ok {
			for propName, propSchema := range props {
				if propMap, ok := propSchema.(map[string]interface{}); ok {
					geminiProp := &genai.Schema{}

					// Convert basic type
					if typeStr, ok := propMap["type"].(string); ok {
						switch typeStr {
						case "string":
							geminiProp.Type = genai.TypeString
						case "integer":
							geminiProp.Type = genai.TypeInteger
						case "number":
							geminiProp.Type = genai.TypeNumber
						case "boolean":
							geminiProp.Type = genai.TypeBoolean
						case "array":
							geminiProp.Type = genai.TypeArray
						case "object":
							geminiProp.Type = genai.TypeObject
						}
					}

					// Add description if available
					if desc, ok := propMap["description"].(string); ok {
						geminiProp.Description = desc
					}

					parameters.Properties[propName] = geminiProp
				}
			}
		}

		// Extract required fields if they exist
		if required, ok := schemaMap["required"].([]interface{}); ok {
			requiredFields := make([]string, 0, len(required))
			for _, field := range required {
				if fieldName, ok := field.(string); ok {
					requiredFields = append(requiredFields, fieldName)
				}
			}
			parameters.Required = requiredFields
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

	// Process tool calls in a loop until we get a final response
	functionCalls := initialFunctionCalls

	for len(functionCalls) > 0 {
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
			resultParts := make([]*genai.Part, len(functionResults))
			for i, fr := range functionResults {
				resultParts[i] = &genai.Part{
					FunctionResponse: fr,
				}
			}

			msgs = append(msgs, &genai.Content{
				Role:  "function",
				Parts: resultParts,
			})
		}

		// Check for system reminder after tool execution
		if reminderFunc := chat.GetSystemReminder(ctx); reminderFunc != nil {
			if reminder := reminderFunc(); reminder != "" {
				msgs = append(msgs, &genai.Content{
					Role: "user",
					Parts: []*genai.Part{
						{Text: reminder},
					},
				})
			}
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

		for chunk, err := range followUpStream {
			if err != nil {
				return chat.Message{}, fmt.Errorf("follow-up streaming error: %w", err)
			}

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
				}
			}
		}

		// If we got more function calls, continue the loop
		if len(functionCalls) > 0 {
			continue
		}

		// No more function calls, we have the final response
		finalMsg := chat.AssistantMessage(respContent.String())

		// Update history - update both messages at once
		c.state.AppendMessages([]chat.Message{initialMsg, finalMsg}, nil)

		return finalMsg, nil
	}

	// This should never be reached since the loop continues until no function calls
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
		} else {
			toolResult.Content = resultStr
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
