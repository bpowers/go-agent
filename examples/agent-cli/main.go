package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	agent "github.com/bpowers/go-agent"
	"github.com/bpowers/go-agent/chat"
	"github.com/bpowers/go-agent/examples/fstools"
	"github.com/bpowers/go-agent/llm"
	"github.com/bpowers/go-agent/persistence/sqlitestore"
)

const defaultModel = "claude-opus-4-1"

func main() {
	if err := run(parseFlags(), os.Stdin, os.Stdout, os.Stderr); err != nil {
		log.Fatal(err)
	}
}

// Config holds the application configuration
type Config struct {
	Model            string
	Provider         string
	APIKey           string
	Temperature      float64
	MaxTokens        int
	SystemPrompt     string
	Debug            bool
	PersistenceFile  string
	CompactThreshold float64
	SystemReminder   bool
}

func parseFlags() *Config {
	return parseFlagsArgs(os.Args[1:])
}

func parseFlagsArgs(args []string) *Config {
	var config Config
	fs := flag.NewFlagSet("agent-cli", flag.ContinueOnError)

	fs.StringVar(&config.Model, "model", defaultModel, "Model to use (e.g., gpt-5, claude-sonnet-4, gemini-2.5-flash)")
	fs.StringVar(&config.APIKey, "api-key", "", "API key (defaults to environment variable based on provider)")
	fs.Float64Var(&config.Temperature, "temperature", -1, "Temperature for response generation (0.0-1.0)")
	fs.IntVar(&config.MaxTokens, "max-tokens", 0, "Maximum tokens in response (0 for default)")
	fs.StringVar(&config.SystemPrompt, "system", "You are a helpful assistant.", "System prompt")
	fs.StringVar(&config.Provider, "provider", "", "Explicit provider to use (override auto-detecting)")
	fs.BoolVar(&config.Debug, "debug", false, "Enable debug output")
	fs.StringVar(&config.PersistenceFile, "persist", "", "SQLite file for conversation persistence (empty for memory-only)")
	fs.Float64Var(&config.CompactThreshold, "compact", 0.8, "Threshold for automatic context compaction (0.0-1.0)")
	fs.BoolVar(&config.SystemReminder, "system-reminder", false, "Enable system reminders that track tool usage and context")
	_ = fs.Parse(args)

	return &config
}

// createClientFunc is a variable to allow mocking in tests
var createClientFunc = func(config *Config) (chat.Client, error) {
	llmConfig := &llm.Config{
		Model:        config.Model,
		Provider:     config.Provider,
		APIKey:       config.APIKey,
		Temperature:  config.Temperature,
		MaxTokens:    config.MaxTokens,
		SystemPrompt: config.SystemPrompt,
		LogLevel:     -1, // Don't change log level from environment default
	}
	return llm.NewClient(llmConfig)
}

func run(config *Config, input io.Reader, output io.Writer, errOutput io.Writer) error {
	// Create the appropriate client based on the model
	client, err := createClientFunc(config)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}

	// Set up session options
	var sessionOpts []agent.SessionOption

	// Set up persistence if requested
	if config.PersistenceFile != "" {
		// Ensure the directory exists
		dir := filepath.Dir(config.PersistenceFile)
		if dir != "." && dir != "" {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return fmt.Errorf("failed to create persistence directory: %w", err)
			}
		}

		store, err := sqlitestore.New(config.PersistenceFile)
		if err != nil {
			return fmt.Errorf("failed to create persistence store: %w", err)
		}
		defer store.Close()
		sessionOpts = append(sessionOpts, agent.WithStore(store))

		_, _ = fmt.Fprintf(output, "Using persistent session: %s\n", config.PersistenceFile)
	}

	// Create a session with automatic context management
	session := agent.NewSession(client, config.SystemPrompt, sessionOpts...)
	session.SetCompactionThreshold(config.CompactThreshold)

	root, err := os.OpenRoot(".")
	if err != nil {
		return fmt.Errorf("failed to open root directory: %w", err)
	}
	defer root.Close()

	ctx := fstools.WithTestFS(context.Background(), root.FS())

	// Track tool usage if system reminders are enabled
	var (
		toolCallCount  int
		filesRead      int
		filesWritten   int
		dirsListed     int
		lastToolCalled string
	)

	// Wrap tool handlers to track usage
	readDirHandler := fstools.ReadDirTool
	readFileHandler := fstools.ReadFileTool
	writeFileHandler := fstools.WriteFileTool

	if config.SystemReminder {
		readDirHandler = func(ctx context.Context, input string) string {
			toolCallCount++
			dirsListed++
			lastToolCalled = "read_dir"
			return fstools.ReadDirTool(ctx, input)
		}

		readFileHandler = func(ctx context.Context, input string) string {
			toolCallCount++
			filesRead++
			lastToolCalled = "read_file"
			return fstools.ReadFileTool(ctx, input)
		}

		writeFileHandler = func(ctx context.Context, input string) string {
			toolCallCount++
			filesWritten++
			lastToolCalled = "write_file"
			return fstools.WriteFileTool(ctx, input)
		}
	}

	if err := session.RegisterTool(fstools.ReadDirToolDef, readDirHandler); err != nil {
		return fmt.Errorf("failed to register ReadDirTool: %w", err)
	}

	if err = session.RegisterTool(fstools.ReadFileToolDef, readFileHandler); err != nil {
		return fmt.Errorf("failed to register ReadFileTool: %w", err)
	}

	if err = session.RegisterTool(fstools.WriteFileToolDef, writeFileHandler); err != nil {
		return fmt.Errorf("failed to register WriteFileTool: %w", err)
	}

	// Create a reader for user input
	reader := bufio.NewReader(input)

	_, _ = fmt.Fprintln(output, "Chat started. Type 'exit' or 'quit' to end the conversation.")
	_, _ = fmt.Fprintln(output, "Type your message and press Enter twice to send (or Ctrl+D on a new line).")
	_, _ = fmt.Fprintln(output, "Commands: /status (show metrics), /help (show help)")
	if config.SystemReminder {
		_, _ = fmt.Fprintln(output, "System reminders: ENABLED (tracking tool usage and context)")
	}
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

				// Show session metrics
				metrics := session.Metrics()
				_, _ = fmt.Fprintf(output, "\nSession Stats:\n")
				_, _ = fmt.Fprintf(output, "  Total tokens used: %d\n", metrics.CumulativeTokens)
				_, _ = fmt.Fprintf(output, "  Live context: %d/%d tokens (%.1f%% full)\n",
					metrics.LiveTokens, metrics.MaxTokens, metrics.PercentFull*100)
				_, _ = fmt.Fprintf(output, "  Records: %d live, %d total\n", metrics.RecordsLive, metrics.RecordsTotal)
				if metrics.CompactionCount > 0 {
					_, _ = fmt.Fprintf(output, "  Compactions: %d (last: %s)\n",
						metrics.CompactionCount, metrics.LastCompaction.Format("15:04:05"))
				}

				return nil
			}
			if err != nil {
				return fmt.Errorf("error reading input: %w", err)
			}

			line = strings.TrimRight(line, "\n\r")

			// Check for commands
			if len(lines) == 0 {
				if line == "exit" || line == "quit" {
					_, _ = fmt.Fprintln(output, "\nGoodbye!")
					return nil
				} else if line == "/status" {
					// Show session status
					metrics := session.Metrics()
					_, _ = fmt.Fprintf(output, "\n📊 Session Status:\n")
					_, _ = fmt.Fprintf(output, "  Context: %d/%d tokens (%.1f%% full)\n",
						metrics.LiveTokens, metrics.MaxTokens, metrics.PercentFull*100)
					_, _ = fmt.Fprintf(output, "  Records: %d live, %d total\n", metrics.RecordsLive, metrics.RecordsTotal)
					_, _ = fmt.Fprintf(output, "  Total tokens used: %d\n", metrics.CumulativeTokens)
					if metrics.CompactionCount > 0 {
						_, _ = fmt.Fprintf(output, "  Compactions: %d (last: %s)\n",
							metrics.CompactionCount, metrics.LastCompaction.Format("15:04:05"))
					}
					_, _ = fmt.Fprintln(output, "---")
					continue
				} else if line == "/help" {
					_, _ = fmt.Fprintln(output, "\nCommands:")
					_, _ = fmt.Fprintln(output, "  /status  - Show session metrics")
					_, _ = fmt.Fprintln(output, "  /help    - Show this help")
					_, _ = fmt.Fprintln(output, "  exit/quit - Exit the program")
					_, _ = fmt.Fprintln(output, "---")
					continue
				}
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
		userMsg := chat.UserMessage(userInput)

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
			case chat.StreamEventTypeToolResult:
				// Display tool result information
				if len(event.ToolResults) > 0 {
					for _, tr := range event.ToolResults {
						name := tr.Name
						if name == "" {
							name = tr.ToolCallID
						}
						_, _ = fmt.Fprintf(output, "✅ Tool result for %s:\n", name)
						if tr.Error != "" {
							_, _ = fmt.Fprintf(output, "   Error: %s\n", tr.Error)
						} else if config.Debug && tr.Content != "" {
							_, _ = fmt.Fprintf(output, "   Result: %s\n", tr.Content)
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

		// Add streaming callback to the options
		opts = append(opts, chat.WithStreamingCb(callback))

		// Add system reminder if enabled
		messageCtx := ctx
		if config.SystemReminder {
			// Reset tool counts for this message
			prevToolCount := toolCallCount
			prevFilesRead := filesRead
			prevFilesWritten := filesWritten
			prevDirsListed := dirsListed

			messageCtx = chat.WithSystemReminder(ctx, func() string {
				// This function executes AFTER tools are called
				if toolCallCount > prevToolCount {
					var actions []string
					if filesRead > prevFilesRead {
						actions = append(actions, fmt.Sprintf("read %d file(s)", filesRead-prevFilesRead))
					}
					if filesWritten > prevFilesWritten {
						actions = append(actions, fmt.Sprintf("wrote %d file(s)", filesWritten-prevFilesWritten))
					}
					if dirsListed > prevDirsListed {
						actions = append(actions, fmt.Sprintf("listed %d director(ies)", dirsListed-prevDirsListed))
					}

					// Check context usage
					metrics := session.Metrics()
					contextInfo := fmt.Sprintf("Context: %.1f%% full", metrics.PercentFull*100)

					if len(actions) > 0 {
						return fmt.Sprintf("<system-reminder>Tools executed: %s. Last tool: %s. %s</system-reminder>",
							strings.Join(actions, ", "), lastToolCalled, contextInfo)
					}
					return fmt.Sprintf("<system-reminder>Tool '%s' was called. %s</system-reminder>",
						lastToolCalled, contextInfo)
				}
				return ""
			})
		}

		_, err := session.Message(messageCtx, userMsg, opts...)
		if err != nil {
			_, _ = fmt.Fprintf(errOutput, "\nError: %v\n", err)
			continue
		}

		// Add newline after streaming completes
		_, _ = fmt.Fprintln(output)
		_, _ = fmt.Fprintln(output, "---")
	}
}
