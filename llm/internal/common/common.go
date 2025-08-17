package common

import (
	"context"
)

// RegisteredTool holds a tool definition and its handler
// This is a shared data structure used by all LLM providers
type RegisteredTool struct {
	Definition string
	Handler    func(context.Context, string) string
}
