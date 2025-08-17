package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/examples/fstools"
	"github.com/bpowers/go-agent/llm"
)

func main() {
	if err := run(parseFlags(), os.Stdin, os.Stdout, os.Stderr); err != nil {
		log.Fatal(err)
	}
}

// Config holds the application configuration
type Config struct {
	Model        string
	APIKey       string
	Temperature  float64
	MaxTokens    int
	SystemPrompt string
	Debug        bool
}

func parseFlags() *Config {
	return parseFlagsArgs(os.Args[1:])
}

func parseFlagsArgs(args []string) *Config {
	var config Config
	fs := flag.NewFlagSet("agent-cli", flag.ContinueOnError)

	fs.StringVar(&config.Model, "model", "gpt-4o-mini", "Model to use (e.g., gpt-4o, claude-3-5-sonnet-20241022, gemini-1.5-flash)")
	fs.StringVar(&config.APIKey, "api-key", "", "API key (defaults to environment variable based on provider)")
	fs.Float64Var(&config.Temperature, "temperature", -1, "Temperature for response generation (0.0-1.0)")
	fs.IntVar(&config.MaxTokens, "max-tokens", 0, "Maximum tokens in response (0 for default)")
	fs.StringVar(&config.SystemPrompt, "system", "You are a helpful assistant.", "System prompt")
	fs.BoolVar(&config.Debug, "debug", false, "Enable debug output")
	_ = fs.Parse(args)

	return &config
}

// createClientFunc is a variable to allow mocking in tests
var createClientFunc = func(config *Config) (chat.Client, error) {
	llmConfig := &llm.Config{
		Model:        config.Model,
		APIKey:       config.APIKey,
		Temperature:  config.Temperature,
		MaxTokens:    config.MaxTokens,
		SystemPrompt: config.SystemPrompt,
		Debug:        config.Debug,
	}
	return llm.NewClient(llmConfig)
}

func run(config *Config, input io.Reader, output io.Writer, errOutput io.Writer) error {
	// Create the appropriate client based on the model
	client, err := createClientFunc(config)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}

	// Create a new chat session
	chatSession := client.NewChat(config.SystemPrompt)

	root, err := os.OpenRoot(".")
	if err != nil {
		return fmt.Errorf("failed to open root directory: %w", err)
	}
	defer root.Close()

	ctx := fstools.WithTestFS(context.Background(), root.FS())

	if err := chatSession.RegisterTool(fstools.ReadDirToolDef, fstools.ReadDirTool); err != nil {
		return fmt.Errorf("failed to register ReadDirTool: %w", err)
	}

	if err = chatSession.RegisterTool(fstools.ReadFileToolDef, fstools.ReadFileTool); err != nil {
		return fmt.Errorf("failed to register ReadFileTool: %w", err)
	}

	if err = chatSession.RegisterTool(fstools.WriteFileToolDef, fstools.WriteFileTool); err != nil {
		return fmt.Errorf("failed to register WriteFileTool: %w", err)
	}

	// Create a reader for user input
	reader := bufio.NewReader(input)

	_, _ = fmt.Fprintln(output, "Chat started. Type 'exit' or 'quit' to end the conversation.")
	_, _ = fmt.Fprintln(output, "Type your message and press Enter twice to send (or Ctrl+D on a new line).")
	_, _ = fmt.Fprintln(output, "---")

	for {
		_, _ = fmt.Fprint(output, "\nYou: ")

		// Read multi-line input until double newline or EOF
		var lines []string
		emptyLineCount := 0

		for {
			line, err := reader.ReadString('\n')
			if err == io.EOF {
				if len(lines) > 0 {
					break
				}
				_, _ = fmt.Fprintln(output, "\nGoodbye!")

				if usage, err := chatSession.TokenUsage(); err == nil {
					_, _ = fmt.Fprintf(output, "usage: %d input tokens, %d output tokens, %d cached tokens", usage.InputTokens, usage.OutputTokens, usage.CachedTokens)
				} else {
					_, _ = fmt.Fprintf(output, "error getting usage info: %v", err)
				}

				return nil
			}
			if err != nil {
				return fmt.Errorf("error reading input: %w", err)
			}

			line = strings.TrimRight(line, "\n\r")

			// Check for exit commands
			if len(lines) == 0 && (line == "exit" || line == "quit") {
				_, _ = fmt.Fprintln(output, "\nGoodbye!")
				return nil
			}

			if line == "" {
				emptyLineCount++
				if emptyLineCount >= 1 && len(lines) > 0 {
					break
				}
			} else {
				emptyLineCount = 0
				lines = append(lines, line)
			}
		}

		userInput := strings.Join(lines, "\n")
		if strings.TrimSpace(userInput) == "" {
			continue
		}

		// Create user message
		userMsg := chat.Message{
			Role:    chat.UserRole,
			Content: userInput,
		}

		// Send message and get response
		_, _ = fmt.Fprint(output, "\nAssistant: ")

		// Build options
		var opts []chat.Option
		if config.Temperature >= 0 {
			opts = append(opts, chat.WithTemperature(config.Temperature))
		}
		if config.MaxTokens > 0 {
			opts = append(opts, chat.WithMaxTokens(config.MaxTokens))
		}

		// Track thinking state for better formatting
		var isThinking bool
		var hasShownThinkingHeader bool

		// Use streaming to show output as it arrives
		callback := func(event chat.StreamEvent) error {
			switch event.Type {
			case chat.StreamEventTypeThinking:
				if event.ThinkingStatus != nil && event.ThinkingStatus.IsThinking {
					if !hasShownThinkingHeader {
						// Show thinking header with clear visual indicator
						_, _ = fmt.Fprint(output, "\n💭 Thinking...\n")
						hasShownThinkingHeader = true
						isThinking = true
					}
					// Stream the thinking content if available
					if event.Content != "" {
						_, _ = fmt.Fprint(output, event.Content)
					}
				}
			case chat.StreamEventTypeThinkingSummary:
				if event.ThinkingStatus != nil {
					isThinking = false
					if event.ThinkingStatus.Summary != "" {
						// Show end of thinking with clear delineation
						_, _ = fmt.Fprint(output, "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n📝 Response:\n")
					} else if hasShownThinkingHeader {
						// Just show delineation if we had thinking but no summary
						_, _ = fmt.Fprint(output, "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n📝 Response:\n")
					}
				}
			case chat.StreamEventTypeToolCall:
				// Display tool invocation information
				if len(event.ToolCalls) > 0 {
					for _, tc := range event.ToolCalls {
						_, _ = fmt.Fprintf(output, "\n🔧 Invoking tool: %s\n", tc.Name)
						if config.Debug && len(tc.Arguments) > 0 {
							_, _ = fmt.Fprintf(output, "   Arguments: %s\n", string(tc.Arguments))
						}
					}
				}
			case chat.StreamEventTypeContent:
				// If we were thinking and now getting content without a summary event,
				// add delineation
				if isThinking {
					isThinking = false
					_, _ = fmt.Fprint(output, "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n📝 Response:\n")
				}
				_, _ = fmt.Fprint(output, event.Content)
			case chat.StreamEventTypeDone:
				// Stream is complete, nothing to display
			}

			return nil
		}

		_, err := chatSession.MessageStream(ctx, userMsg, callback, opts...)
		if err != nil {
			_, _ = fmt.Fprintf(errOutput, "\nError: %v\n", err)
			continue
		}

		// Add newline after streaming completes
		_, _ = fmt.Fprintln(output)
		_, _ = fmt.Fprintln(output, "---")
	}
}
