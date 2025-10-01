package llm

import (
	"log/slog"

	"github.com/bpowers/go-agent/internal/logging"
)

// SetLogLevel sets the log level for the entire go-agent library.
// This affects all LLM providers (OpenAI, Claude, Gemini) and is global to the process.
//
// Log Level Guide:
//   - slog.LevelError: Only errors (minimal logging, production default for quiet systems)
//   - slog.LevelWarn: Warnings and errors (default - shows issues without overwhelming)
//   - slog.LevelInfo: Informational messages (shows high-level operations like "using OpenAI client")
//   - slog.LevelDebug: Verbose debugging (shows every stream event, tool call, token update - very verbose)
//
// Example - Enable debug logging for development:
//
//	import "log/slog"
//	llm.SetLogLevel(slog.LevelDebug)
//
// Example - Quiet production logging:
//
//	llm.SetLogLevel(slog.LevelError)
//
// This can also be controlled via the GO_AGENT_DEBUG environment variable:
//
//	GO_AGENT_DEBUG=0  # Error level
//	GO_AGENT_DEBUG=1  # Warn level (default)
//	GO_AGENT_DEBUG=2  # Info level
//	GO_AGENT_DEBUG=3  # Debug level
//
// Note: This is a global setting. You cannot set different log levels for different
// providers or clients. All logging in the process shares the same level.
func SetLogLevel(level slog.Level) {
	logging.SetLogLevel(level)
}
