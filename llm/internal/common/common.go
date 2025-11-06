package common

import (
	"context"

	"github.com/bpowers/go-agent/chat"
)

// RegisteredTool holds a callable tool
// This is a shared data structure used by all LLM providers
type RegisteredTool struct {
	Tool chat.Tool
}
