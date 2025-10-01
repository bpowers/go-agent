package logging

import (
	"log/slog"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseLogLevel(t *testing.T) {
	tests := []struct {
		input    string
		expected slog.Level
	}{
		{"0", slog.LevelError},
		{"1", slog.LevelWarn},
		{"2", slog.LevelInfo},
		{"3", slog.LevelDebug},
		{"", slog.LevelWarn},        // Default
		{"invalid", slog.LevelWarn}, // Default
		{"99", slog.LevelWarn},      // Default
		{"-1", slog.LevelWarn},      // Default
		{"four", slog.LevelWarn},    // Default
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := parseLogLevel(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestSetLogLevel(t *testing.T) {
	// Save original level
	originalLevel := logLevel.Level()
	defer logLevel.Set(originalLevel)

	// Test setting to Debug
	SetLogLevel(slog.LevelDebug)
	assert.Equal(t, slog.LevelDebug, logLevel.Level())

	// Test setting to Error
	SetLogLevel(slog.LevelError)
	assert.Equal(t, slog.LevelError, logLevel.Level())

	// Test setting to Info
	SetLogLevel(slog.LevelInfo)
	assert.Equal(t, slog.LevelInfo, logLevel.Level())

	// Test setting to Warn
	SetLogLevel(slog.LevelWarn)
	assert.Equal(t, slog.LevelWarn, logLevel.Level())
}

func TestLogger(t *testing.T) {
	// Logger should return a non-nil logger
	require.NotNil(t, Logger())

	// Logger should return the same instance
	logger1 := Logger()
	logger2 := Logger()
	assert.Equal(t, logger1, logger2)
}
