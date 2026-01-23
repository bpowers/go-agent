package common

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildToolResult_DisplaySuccessUsesSummary(t *testing.T) {
	raw := `{"status":"success","summary":"Summary here.","executionTimeMs":10,"executionError":null,"html":"<html>ok</html>"}`

	result := BuildToolResult("Display", "tool-1", raw, nil)

	assert.Equal(t, "Display", result.Name)
	assert.Equal(t, "tool-1", result.ToolCallID)
	assert.Equal(t, "Summary here.", result.Content)
	assert.Equal(t, raw, result.DisplayContent)
}

func TestBuildToolResult_DisplaySuccessFallback(t *testing.T) {
	raw := `{"status":"success","summary":"","executionTimeMs":10,"executionError":null,"html":"<html>ok</html>"}`

	result := BuildToolResult("Display", "tool-1", raw, nil)

	assert.Equal(t, displaySummaryFallback, result.Content)
	assert.Equal(t, raw, result.DisplayContent)
}

func TestBuildToolResult_DisplayErrorKeepsFullContent(t *testing.T) {
	raw := `{"status":"error","summary":"Execution failed","executionTimeMs":10,"executionError":{"message":"boom"},"html":""}`

	result := BuildToolResult("Display", "tool-1", raw, nil)

	assert.Equal(t, raw, result.Content)
	assert.Equal(t, raw, result.DisplayContent)
}

func TestBuildToolResult_DisplaySummaryFailureFallsBack(t *testing.T) {
	raw := `{"status":"success","summary":"Executed 1 code cell(s), produced 1 output(s)","executionTimeMs":10,"executionError":null,"html":"<html>ok</html>","error":"Display: display summary failed: no cheap client available"}`

	result := BuildToolResult("Display", "tool-1", raw, nil)

	assert.Equal(t, displaySummaryFallback, result.Content)
	assert.Equal(t, raw, result.DisplayContent)
}

func TestBuildToolResult_DisplayInvalidJSONKeepsFullContent(t *testing.T) {
	raw := "not json"

	result := BuildToolResult("Display", "tool-1", raw, nil)

	assert.Equal(t, raw, result.Content)
	assert.Equal(t, raw, result.DisplayContent)
}

func TestBuildToolResult_NonDisplayKeepsContent(t *testing.T) {
	raw := `{"status":"success"}`

	result := BuildToolResult("RunPython", "tool-1", raw, nil)

	assert.Equal(t, raw, result.Content)
	assert.Empty(t, result.DisplayContent)
}

func TestBuildToolResult_ErrorPropagates(t *testing.T) {
	result := BuildToolResult("Display", "tool-1", "ignored", assert.AnError)

	require.Equal(t, assert.AnError.Error(), result.Error)
	assert.Empty(t, result.Content)
	assert.Empty(t, result.DisplayContent)
}
