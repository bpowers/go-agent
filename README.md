# go-agent

A flexible, multi-provider agent framework for Go that provides a unified interface for building LLM-powered applications.

## Why go-agent?

Building LLM-powered applications requires dealing with multiple providers, each with their own SDKs, streaming protocols, and quirks. go-agent solves this by providing:

- **One interface, multiple providers**: Write your code once, switch between OpenAI, Claude, and Gemini without changing your application logic
- **Complexity abstracted**: Handle streaming, tool calling, and token tracking with a clean, simple API
- **Provider quirks handled**: We deal with rate limits, retry logic, and provider-specific requirements so you don't have to
- **Production-ready**: Comprehensive test coverage, concurrent-safe implementations, and battle-tested across all major providers

## Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI, Claude (Anthropic), and Gemini models
- **Unified Chat Interface**: Single API regardless of underlying provider
- **Streaming Support**: Real-time streaming responses with thinking/reasoning support
- **Tool Calling**: Comprehensive tool/function calling with multi-round support (up to 10 rounds)
- **Token Usage Tracking**: Monitor token consumption across all providers
- **Type-Safe**: Fully typed Go interfaces with compile-time safety
- **Code Generation**: Built-in tools to generate JSON schemas and MCP tool definitions from Go types

## Installation

```bash
go get github.com/bpowers/go-agent
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    
    "github.com/bpowers/go-agent/chat"
    "github.com/bpowers/go-agent/llm/openai"
    // Or: "github.com/bpowers/go-agent/llm/claude"
    // Or: "github.com/bpowers/go-agent/llm/gemini"
)

func main() {
    // Create a client for your preferred provider
    client, err := openai.NewClient(
        openai.OpenAIURL,
        os.Getenv("OPENAI_API_KEY"),
        openai.WithModel("gpt-5-mini"),
    )
    if err != nil {
        log.Fatal(err)
    }
    
    // Create a chat session
    session := client.NewChat("You are a helpful assistant.")
    
    // Send a message
    response, err := session.Message(context.Background(), chat.Message{
        Role:    chat.UserRole,
        Content: "Hello! What can you help me with?",
    })
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println(response.Content)
}
```

### Streaming Example

```go
// Stream responses with a callback
_, err := session.Message(
    context.Background(),
    chat.Message{
        Role:    chat.UserRole,
        Content: "Explain quantum computing",
    },
    chat.WithStreamingCb(func(event chat.StreamEvent) error {
        switch event.Type {
        case chat.StreamEventTypeContent:
            fmt.Print(event.Content) // Print as it arrives
        case chat.StreamEventTypeThinking:
            // Handle thinking/reasoning events (Claude, OpenAI Responses)
            if event.ThinkingStatus.IsThinking {
                fmt.Println("AI is thinking...")
            }
        }
        return nil
    }),
)
```

### Tool Calling Example

TODO

## Session Management and Persistence

The library includes a Session interface that extends Chat with automatic context window management:

```go
import (
    agent "github.com/bpowers/go-agent"
    "github.com/bpowers/go-agent/persistence/sqlitestore"
)

// Create a session with automatic context compaction
session := agent.NewSession(
    client,
    "You are a helpful assistant",
    agent.WithStore(sqlitestore.New("chat.db")),           // Optional: persist to SQLite
    agent.WithCompactionThreshold(0.8),                    // Compact at 80% full (default)
)

// Use it like a regular Chat
response, err := session.Message(ctx, chat.Message{
    Role:    chat.UserRole,
    Content: "Hello!",
})

// Access session-specific features
metrics := session.SessionMetrics()  // Token usage, compaction stats
records := session.LiveRecords()     // Current context window
session.CompactNow()                 // Manual compaction
```

When the context window approaches capacity, the Session automatically:
1. Summarizes older messages to preserve context
2. Marks old records as "dead" (kept for history but not sent to LLM)
3. Creates a summary record to maintain conversation continuity

## Examples

See the `examples/agent-cli` directory for a complete command-line chat application that demonstrates:
- Interactive chat sessions with multi-line input
- Real-time streaming responses
- Tool registration and execution
- Thinking/reasoning display for supported models
- Conversation history management with Session interface

## Architecture

```
examples/           # Example applications
  agent-cli/        # Command-line chat interface
llm/                # LLM provider implementations  
  openai/           # OpenAI (GPT models) - supports both Responses and ChatCompletions APIs
  claude/           # Anthropic Claude models
  gemini/           # Google Gemini models
  internal/common/  # Shared internal utilities (RegisteredTool)
  testing/          # Testing utilities and helpers
chat/               # Common chat interface and types
schema/             # JSON schema utilities
cmd/build/          # Code generation tools
  funcschema/       # Generate MCP tool definitions from Go functions
  jsonschema/       # Generate JSON schemas from Go types
```

## Development Guide

This section contains important information for developers and AI agents working on this codebase.

### Problem-Solving Philosophy

- **Write high-quality, general-purpose solutions**: Implement solutions that work correctly for all valid inputs, not just test cases. Do not hard-code values or create solutions that only work for specific test inputs.
- **Prioritize the right approach over the first approach**: Research the proper way to implement features rather than implementing workarounds. For example, check if an API provides token usage directly before implementing token counting.
- **Keep implementations simple and maintainable**: Start with the simplest solution that meets requirements. Only add complexity when the simple approach demonstrably fails.
- **No special casing in tests**: Tests should hold all implementations to the same standard. Never add conditional logic in tests that allows certain implementations to skip requirements.
- **Complete all aspects of a task**: When fixing bugs or implementing features, ensure the fix works for all code paths, not just the primary one.

### Development Setup

#### Required Tools

Run the setup script to install development tools and git hooks:

```bash
./scripts/install-hooks.sh
```

This will:
- Install `golangci-lint` for comprehensive Go linting
- Install `gofumpt` for code formatting
- Set up a pre-commit hook that runs all checks

#### Pre-commit Checks

The pre-commit hook automatically runs:
1. **Code formatting check** - Ensures all code is formatted with `gofumpt`
2. **Generated code check** - Verifies `go generate` output is committed
3. **Linting** - Runs `golangci-lint` with our configuration
4. **Tests** - Runs all tests with race detection enabled

You MUST NOT bypass the pre-commit hook with `--no-verify`.  Fix the root issue causing the hook to fail.  If you uncover a deep problem with an ambiguous solution, present the problem and background to the user to get their decision on how to proceed.

### Go Development Standards

#### Code Style and Safety
- Use `omitzero` instead of `omitempty` in JSON struct tags
- Run `gofumpt -w .` before committing to ensure proper formatting
- Write concurrency-safe code by default. Favor immutability; use mutexes for shared mutable state
- **ALWAYS call Unlock()/RUnlock() in a defer statement**, never synchronously (enforced by golangci-lint)
- Use `go doc` to inspect and understand third-party APIs before implementing

#### Testing
- Run tests with `./with_api_keys.sh go test -race ./...` to include integration tests
- The `with_api_keys.sh` script loads API credentials from `~/.openai_key`, `~/.claude_key`, and `~/.gemini_key`
- Use `github.com/stretchr/testify/assert` and `require` for test assertions
  - Use `require` for fatal errors that should stop the test immediately (like setup failures)
  - Use `assert` for non-fatal assertions that allow the test to continue gathering information
  - Generally avoid adding custom messages to assertions - the default error messages are usually clear
- Write table-driven tests with `t.Run()` to test all API variations comprehensively
- Integration tests should verify both feature availability and actual functionality

#### Project Conventions
- This is a monorepo with no external package dependencies on it
- When introducing a new abstraction, migrate all users to it and completely remove the old one
- Be bold in refactoring - there's no "legacy code" to preserve
- Commit messages should be single-line, descriptive but terse, starting with the package prefix (e.g., `llm/openai: add streaming support`)

### LLM Provider Implementation Patterns

When implementing features across multiple LLM providers, follow these established patterns:

#### Streaming API Differences
Each provider handles streaming differently:
- **OpenAI**: Uses Server-Sent Events with discrete chunks. Token usage arrives in a dedicated usage chunk at the end
- **Claude**: Uses a custom event stream format with delta events. Token usage comes in `message_delta` events
- **Gemini**: Uses its own streaming protocol. Token usage arrives in `UsageMetadata`

Tool arguments often arrive in fragments that must be accumulated:
- OpenAI: `delta.tool_calls` with incremental updates
- Claude: `input_json_delta` events that build up the full JSON
- Gemini: Within `FunctionCall` parts

#### Tool Calling Architecture
All providers support tool calling but with different approaches:
- **OpenAI**: Tools are top-level arrays in messages, with explicit tool call IDs (max 40 chars)
- **Claude**: Tools are content blocks within messages, mixing text and tool use blocks
- **Gemini**: Function calls are parts within content, similar to Claude but with different typing

The multi-round pattern (implemented in all providers):
1. Detect tool calls in the streaming response
2. Execute tools with the provided context (context must flow from Message to handlers)
3. Format results according to provider requirements
4. Make follow-up API calls with tool results
5. Repeat until no more tool calls (max 10 rounds to prevent infinite loops)

#### Provider-Specific Quirks

**OpenAI**:
- ChatCompletions API supports tools; Responses API has reasoning but no tool support
- Automatic retry on rate limits with exponential backoff
- Tool call IDs limited to 40 characters
- Both APIs share the same client implementation

**Claude**:
- Native thinking/reasoning support through content blocks
- Tool results must be in the same message as tool calls
- Content blocks can mix text, thinking, and tool use
- Requires careful handling of message roles in tool responses

**Gemini**:
- Function calls are "parts" within content
- Different streaming event structure than other providers
- Requires special handling for tool results formatting
- May require multiple parts for complex responses

#### Common Pitfalls to Avoid
- Don't assume streaming events arrive in a specific order or completeness
- Tool call IDs have provider-specific constraints
- Some providers require tools to be included in follow-up requests, others don't
- Message history formatting varies significantly between providers
- Not all provider features are available in all API modes
- Token counting fallbacks should only be used when the API doesn't provide usage

### Environment Variables

- `OPENAI_API_KEY`: API key for OpenAI
- `ANTHROPIC_API_KEY`: API key for Claude  
- `GEMINI_API_KEY`: API key for Gemini
- `GO_AGENT_DEBUG`: Set to "1" to enable debug logging for SSE streams

### Code Generation Tools

The project includes tools for generating JSON schemas and MCP (Model Context Protocol) tool definitions:

```bash
# Generate JSON schema from a Go type
go run ./cmd/build/jsonschema -type MyStruct -input myfile.go

# Generate MCP tool wrapper from a Go function  
go run ./cmd/build/funcschema -func MyFunction -input myfile.go
```

These tools are useful for:
- Creating tool definitions for LLM function calling
- Generating JSON schemas for API validation
- Ensuring type safety between Go code and LLM tools

### Adding a New Provider

To add support for a new LLM provider:

1. Create a new package under `llm/` (e.g., `llm/newprovider/`)
2. Implement the `chat.Client` and `chat.Chat` interfaces
3. Handle streaming with proper event types
4. Implement tool calling with the multi-round pattern
5. Add integration tests following the patterns in `llm/testing/`
6. Update `llm.NewClient()` to detect and instantiate your provider
7. Document any provider-specific quirks in this README

### Project Maintenance

- Keep provider-specific logic isolated in provider packages
- Shared code goes in `llm/internal/common/` (like `RegisteredTool`)
- Model limits and configurations stay within provider packages
- Update integration tests when adding new features
- Maintain backward compatibility for the public API

## License

[Apache 2.0](./LICENSE)
