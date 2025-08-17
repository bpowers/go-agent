package main

import (
	"bytes"
	"context"
	"flag"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/bpowers/go-agent/chat"
)

// Tests for isResponsesModel and detectProvider were moved to llm/client_test.go

func TestParseFlags(t *testing.T) {
	// Save original command line args
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()

	// Reset flags for testing
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)

	tests := []struct {
		name     string
		args     []string
		expected *Config
	}{
		{
			name: "Default values",
			args: []string{"agent-cli"},
			expected: &Config{
				Model:        "gpt-4o-mini",
				APIKey:       "",
				Temperature:  0.7,
				MaxTokens:    0,
				SystemPrompt: "You are a helpful assistant.",
				Debug:        false,
			},
		},
		{
			name: "Custom model",
			args: []string{"agent-cli", "-model", "claude-3-opus"},
			expected: &Config{
				Model:        "claude-3-opus",
				APIKey:       "",
				Temperature:  0.7,
				MaxTokens:    0,
				SystemPrompt: "You are a helpful assistant.",
				Debug:        false,
			},
		},
		{
			name: "With API key",
			args: []string{"agent-cli", "-api-key", "test-key-123"},
			expected: &Config{
				Model:        "gpt-4o-mini",
				APIKey:       "test-key-123",
				Temperature:  0.7,
				MaxTokens:    0,
				SystemPrompt: "You are a helpful assistant.",
				Debug:        false,
			},
		},
		{
			name: "Custom temperature",
			args: []string{"agent-cli", "-temperature", "0.2"},
			expected: &Config{
				Model:        "gpt-4o-mini",
				APIKey:       "",
				Temperature:  0.2,
				MaxTokens:    0,
				SystemPrompt: "You are a helpful assistant.",
				Debug:        false,
			},
		},
		{
			name: "Custom max tokens",
			args: []string{"agent-cli", "-max-tokens", "2048"},
			expected: &Config{
				Model:        "gpt-4o-mini",
				APIKey:       "",
				Temperature:  0.7,
				MaxTokens:    2048,
				SystemPrompt: "You are a helpful assistant.",
				Debug:        false,
			},
		},
		{
			name: "Custom system prompt",
			args: []string{"agent-cli", "-system", "You are a coding assistant."},
			expected: &Config{
				Model:        "gpt-4o-mini",
				APIKey:       "",
				Temperature:  0.7,
				MaxTokens:    0,
				SystemPrompt: "You are a coding assistant.",
				Debug:        false,
			},
		},
		{
			name: "Debug mode",
			args: []string{"agent-cli", "-debug"},
			expected: &Config{
				Model:        "gpt-4o-mini",
				APIKey:       "",
				Temperature:  0.7,
				MaxTokens:    0,
				SystemPrompt: "You are a helpful assistant.",
				Debug:        true,
			},
		},
		{
			name: "All options",
			args: []string{
				"agent-cli",
				"-model", "gemini-1.5-pro",
				"-api-key", "secret-key",
				"-temperature", "0.9",
				"-max-tokens", "4096",
				"-system", "Custom prompt",
				"-debug",
			},
			expected: &Config{
				Model:        "gemini-1.5-pro",
				APIKey:       "secret-key",
				Temperature:  0.9,
				MaxTokens:    4096,
				SystemPrompt: "Custom prompt",
				Debug:        true,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset flags for each test
			flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
			os.Args = tt.args

			result := parseFlags()
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestCreateClient was moved to llm/client_test.go as TestNewClient

// MockChat implements the chat.Chat interface for testing
type MockChat struct {
	systemPrompt string
	messages     []chat.Message
	responses    []chat.Message
	responseIdx  int
	err          error
	streamChunks []string           // For testing streaming
	streamEvents []chat.StreamEvent // For testing thinking and other events
}

func (m *MockChat) Message(ctx context.Context, msg chat.Message, opts ...chat.Option) (chat.Message, error) {
	// Just call MessageStream with nil callback
	return m.MessageStream(ctx, msg, nil, opts...)
}

func (m *MockChat) MessageStream(ctx context.Context, msg chat.Message, callback chat.StreamCallback, opts ...chat.Option) (chat.Message, error) {
	if m.err != nil {
		return chat.Message{}, m.err
	}

	m.messages = append(m.messages, msg)

	var resp chat.Message
	if m.responseIdx < len(m.responses) {
		resp = m.responses[m.responseIdx]
		m.responseIdx++
	} else {
		resp = chat.Message{
			Role:    chat.AssistantRole,
			Content: "Mock response",
		}
	}

	// If we have specific stream events, send those
	if callback != nil && len(m.streamEvents) > 0 {
		for _, event := range m.streamEvents {
			if err := callback(event); err != nil {
				return chat.Message{}, err
			}
		}
	} else if callback != nil && len(m.streamChunks) > 0 {
		// If we have stream chunks and a callback, simulate streaming
		for _, chunk := range m.streamChunks {
			event := chat.StreamEvent{
				Type:    chat.StreamEventTypeContent,
				Content: chunk,
			}
			if err := callback(event); err != nil {
				return chat.Message{}, err
			}
		}
	} else if callback != nil {
		// Stream the full response as chunks of words
		words := strings.Fields(resp.Content)
		for i, word := range words {
			chunk := word
			if i < len(words)-1 {
				chunk += " "
			}
			event := chat.StreamEvent{
				Type:    chat.StreamEventTypeContent,
				Content: chunk,
			}
			if err := callback(event); err != nil {
				return chat.Message{}, err
			}
		}
	}

	return resp, nil
}

func (m *MockChat) History() (systemPrompt string, msgs []chat.Message) {
	return m.systemPrompt, m.messages
}

// TokenUsage returns mock token usage
func (m *MockChat) TokenUsage() (chat.TokenUsage, error) {
	return chat.TokenUsage{}, nil
}

// MaxTokens returns mock max tokens
func (m *MockChat) MaxTokens() int {
	return 4096
}

// RegisterTool registers a mock tool
func (m *MockChat) RegisterTool(def chat.ToolDef, fn func(context.Context, string) string) error {
	return nil
}

// DeregisterTool removes a mock tool
func (m *MockChat) DeregisterTool(name string) {
	// No-op for mock
}

// ListTools returns mock tool list
func (m *MockChat) ListTools() []string {
	return []string{}
}

// MockClient implements the chat.Client interface for testing
type MockClient struct {
	chat *MockChat
}

func (m *MockClient) NewChat(systemPrompt string, initialMsgs ...chat.Message) chat.Chat {
	if m.chat == nil {
		m.chat = &MockChat{
			systemPrompt: systemPrompt,
			messages:     initialMsgs,
		}
	} else {
		m.chat.systemPrompt = systemPrompt
		m.chat.messages = initialMsgs
	}
	return m.chat
}

func TestRun(t *testing.T) {
	tests := []struct {
		name      string
		config    *Config
		input     string
		responses []chat.Message
		wantOut   []string
		wantErr   bool
	}{
		{
			name: "Single message and exit",
			config: &Config{
				Model:        "gpt-4",
				APIKey:       "test-key",
				SystemPrompt: "Test assistant",
			},
			input: "Hello\n\nexit\n",
			responses: []chat.Message{
				{Role: chat.AssistantRole, Content: "Hello! How can I help you?"},
			},
			wantOut: []string{
				"Chat started",
				"You:",
				"Assistant: Hello! How can I help you?",
				"Goodbye!",
			},
			wantErr: false,
		},
		{
			name: "Multiple messages",
			config: &Config{
				Model:        "gpt-4",
				APIKey:       "test-key",
				SystemPrompt: "Test assistant",
			},
			input: "First message\n\nSecond message\n\nexit\n",
			responses: []chat.Message{
				{Role: chat.AssistantRole, Content: "First response"},
				{Role: chat.AssistantRole, Content: "Second response"},
			},
			wantOut: []string{
				"Chat started",
				"First response",
				"Second response",
				"Goodbye!",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock client
			oldCreateClient := createClientFunc
			defer func() { createClientFunc = oldCreateClient }()

			mockClient := &MockClient{
				chat: &MockChat{
					responses: tt.responses,
				},
			}

			createClientFunc = func(config *Config) (chat.Client, error) {
				return mockClient, nil
			}

			// Set up IO
			input := strings.NewReader(tt.input)
			var output bytes.Buffer
			var errOutput bytes.Buffer

			// Run the function
			err := run(tt.config, input, &output, &errOutput)

			// Check error
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}

			// Check output contains expected strings
			outputStr := output.String()
			for _, want := range tt.wantOut {
				assert.Contains(t, outputStr, want)
			}
		})
	}
}
