package common

import (
	"context"

	"github.com/bpowers/go-agent/chat"
)

// RegisteredTool holds a tool definition and its handler
// This is a shared data structure used by all LLM providers
type RegisteredTool struct {
	Definition chat.ToolDef
	Handler    func(context.Context, string) string
}
