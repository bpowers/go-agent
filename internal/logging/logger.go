// Package logging provides centralized structured logging for the go-agent library.
//
// Log Level Semantics:
//   - Error: Unrecoverable errors and unexpected states indicating bugs
//   - Warn: Recoverable issues, missing data, fallbacks (e.g., unknown model, no token usage)
//   - Info: High-level operations (client creation, API selection, model info)
//   - Debug: Detailed execution trace (stream events, tool calls, token updates, raw data)
//
// The log level can be controlled via:
//  1. GO_AGENT_DEBUG environment variable (0=Error, 1=Warn, 2=Info, 3=Debug)
//  2. llm.SetLogLevel() function for programmatic control
//
// All logging is global and affects all LLM providers in the process.
package logging

import (
	"log/slog"
	"os"
)

var (
	logLevel = new(slog.LevelVar)
	logger   *slog.Logger
)

func init() {
	level := parseLogLevel(os.Getenv("GO_AGENT_DEBUG"))
	logLevel.Set(level)

	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: logLevel,
	})
	logger = slog.New(handler)
}

// Logger returns the global logger instance.
func Logger() *slog.Logger {
	return logger
}

// SetLogLevel sets the global log level for the entire go-agent library.
// This is a process-wide setting that affects all LLM providers (OpenAI, Claude, Gemini).
//
// Changes take effect immediately for all future log calls.
func SetLogLevel(level slog.Level) {
	logLevel.Set(level)
}

// parseLogLevel converts GO_AGENT_DEBUG environment variable values to slog levels.
// Mapping: 0=Error, 1=Warn, 2=Info, 3=Debug
// Default: Warn if not set or invalid
func parseLogLevel(envVal string) slog.Level {
	switch envVal {
	case "0":
		return slog.LevelError
	case "1":
		return slog.LevelWarn
	case "2":
		return slog.LevelInfo
	case "3":
		return slog.LevelDebug
	default:
		return slog.LevelWarn
	}
}
