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

// SetLogLevel sets the global log level for the entire library.
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
