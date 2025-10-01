package llm

import (
	"log/slog"

	"github.com/bpowers/go-agent/internal/logging"
)

// SetLogLevel sets the log level for the entire go-agent library.
// This affects all LLM providers (OpenAI, Claude, Gemini) and is global to the process.
//
// Common levels:
//   - slog.LevelError: Only errors
//   - slog.LevelWarn: Warnings and errors (default)
//   - slog.LevelInfo: Informational messages, warnings, and errors
//   - slog.LevelDebug: Verbose debugging output
//
// This can also be controlled via the GO_AGENT_DEBUG environment variable:
//   - 0 = Error
//   - 1 = Warn (default)
//   - 2 = Info
//   - 3 = Debug
func SetLogLevel(level slog.Level) {
	logging.SetLogLevel(level)
}
