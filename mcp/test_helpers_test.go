package mcp

import (
	"context"

	"github.com/bpowers/go-agent/chat"
)

type stubTool struct {
	name        string
	description string
	schema      string
	result      string
	calledWith  *string
}

func (s *stubTool) MCPJsonSchema() string {
	return s.schema
}

func (s *stubTool) Name() string {
	return s.name
}

func (s *stubTool) Description() string {
	return s.description
}

func (s *stubTool) Call(ctx context.Context, input string) string {
	if s.calledWith != nil {
		*s.calledWith = input
	}
	return s.result
}

var _ chat.Tool = (*stubTool)(nil)

// panicTool is a test tool that panics when called
type panicTool struct{}

func (panicTool) MCPJsonSchema() string {
	return `{"name":"PanicTool","description":"A tool that panics for testing","inputSchema":{"type":"object","properties":{},"additionalProperties":false},"outputSchema":{"type":"object","properties":{"error":{"type":["string","null"]}},"additionalProperties":false}}`
}

func (panicTool) Name() string {
	return "PanicTool"
}

func (panicTool) Description() string {
	return "A tool that panics for testing"
}

func (panicTool) Call(_ context.Context, _ string) string {
	panic("intentional panic for testing")
}

var _ chat.Tool = (*panicTool)(nil)
