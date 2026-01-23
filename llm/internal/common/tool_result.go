package common

import (
	"encoding/json"
	"strings"

	"github.com/bpowers/go-agent/chat"
)

const (
	displaySummaryFallback      = "Display result successfully shown to user."
	displaySummaryFailureMarker = "display summary failed"
)

type displaySummaryPayload struct {
	Status         string           `json:"status"`
	Summary        string           `json:"summary"`
	Error          *string          `json:"error"`
	ExecutionError *json.RawMessage `json:"executionError"`
}

// BuildToolResult returns a ToolResult with context-safe content and optional display content.
func BuildToolResult(toolName, toolCallID, raw string, execErr error) chat.ToolResult {
	result := chat.ToolResult{
		ToolCallID: toolCallID,
		Name:       toolName,
	}

	if execErr != nil {
		result.Error = execErr.Error()
		return result
	}

	if toolName != "Display" {
		result.Content = raw
		return result
	}

	result.DisplayContent = raw
	summary, ok, isError := extractDisplaySummary(raw)
	if !ok || isError {
		result.Content = raw
		return result
	}

	if summary == "" {
		result.Content = displaySummaryFallback
		return result
	}

	result.Content = summary
	return result
}

func extractDisplaySummary(raw string) (summary string, ok bool, isError bool) {
	var payload displaySummaryPayload
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return "", false, false
	}

	status := strings.TrimSpace(strings.ToLower(payload.Status))
	hasExecErr := payloadHasExecutionError(payload)
	if hasExecErr {
		return "", true, true
	}

	if payload.Error != nil {
		if status == "success" && isSummaryFailure(*payload.Error) {
			return "", true, false
		}
		return "", true, true
	}

	if status == "" || status == "error" {
		return "", true, true
	}

	return strings.TrimSpace(payload.Summary), true, false
}

func isSummaryFailure(err string) bool {
	return strings.Contains(strings.ToLower(err), displaySummaryFailureMarker)
}

func payloadHasExecutionError(payload displaySummaryPayload) bool {
	if payload.ExecutionError == nil {
		return false
	}
	return strings.TrimSpace(string(*payload.ExecutionError)) != "null"
}
