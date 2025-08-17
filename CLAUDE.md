# go-agent

A flexible, multi-provider agent framework for Go that provides a unified interface for building LLM-powered applications.

## Why go-agent?

Building LLM-powered applications requires dealing with multiple providers, each with their own SDKs, streaming protocols, and quirks. go-agent solves this by providing:

- **One interface, multiple providers**: Write your code once, switch between OpenAI, Claude, and Gemini without changing your application logic
- **Complexity abstracted**: Handle streaming, tool calling, and token tracking with a clean, simple API
- **Provider quirks handled**: We deal with rate limits, retry logic, and provider-specific requirements so you don't have to
- **Production-ready**: Comprehensive test coverage, concurrent-safe implementations, and battle-tested across all major providers

## Repository Layout

```
examples/           # Example applications
  agent-cli/        # Command-line chat interface (displays tool invocations)
llm/                # LLM provider implementationsq
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

## Commit Message Style

- Initial-line format: component: description
- Component prefix: Use the module/directory name (e.g., llm/gemini, schema, cmd/build/funcschema)
- Description: Start with lowercase, present tense verb, no period
- Length: Keep the initial line concise, typically under 60 characters
- Examples:
  - llm: fix TokenCount() to be cumulative
  - diagram: display equations as LaTeX
  - testing: add basic visual regression tests
- Add 1 to 2 paragraphs of the "why" of the change in the body of the commit message.  Especially highlight any assumptions you made or non-obvious pieces of the change.
- DO NOT use any emoji in the commit message.


## Development Workflow for LLM Agents

When working on this codebase, follow this systematic approach:

### 0, Problem-Solving Philosophy

- **Write high-quality, general-purpose solutions**: Implement solutions that work correctly for all valid inputs, not just test cases. Do not hard-code values or create solutions that only work for specific test inputs.
- **Prioritize the right approach over the first approach**: Research the proper way to implement features rather than implementing workarounds. For example, check if an API provides token usage directly before implementing token counting.
- **Keep implementations simple and maintainable**: Start with the simplest solution that meets requirements. Only add complexity when the simple approach demonstrably fails.
- **No special casing in tests**: Tests should hold all implementations to the same standard. Never add conditional logic in tests that allows certain implementations to skip requirements.
- **Complete all aspects of a task**: When fixing bugs or implementing features, ensure the fix works for all code paths, not just the primary one.  Continue making progress until all aspects of the user's task are done, including newly discovered but necessary work and rework along the way.

### 1. Understanding Requirements
- Read relevant code and documentation (including for libraries via `go doc`) and build a plan based on the user's task.
- ultrathink: if there are important and ambiguous high-level achitecture decisions or "trapdoor" choices, stop and ask the user.
- All implementations must be consistent across providers (OpenAI, Claude, Gemini).
- Start by adding tests to validate assumptions before making changes.
- Use the existing test helpers in `llm/testing/integration.go` as patterns for new test helpers.
- Remember: we want to build the simplest interfaces and abstractions possible while fully addressing the intrinsic complexity of the problem domain.

### 2. Test-First Development
When fixing bugs or adding features:
1. **Create a test helper** in `llm/testing/integration.go` that validates the expected behavior
2. **Write provider-specific tests** that use the helper (in `llm/{openai,claude,gemini}/client_integration_test.go`)
3. **Run tests to confirm the issue** exists across all providers
4. **Fix implementations** one provider at a time
5. **Run gofumpt** after each change: `gofumpt -w <modified files>`
6. **Ensure all tests pass** before committing
7. **Commit with descriptive messages** strictly following the commit message style from above

### 3. Multi-Step Task Execution
For complex tasks with multiple components (like the TokenUsage fix + tool call events):
1. **Break down into discrete tasks** and track with a todo list
2. **Complete each task fully** including tests and formatting before moving to the next
3. **Commit each logical change separately** with clear commit messages
4. **Boldly refactor** when needed - there's no legacy code to preserve
5. **Address the root cause** if you find that a problem is due to a bad abstraction or deficiency in the current code, stop and create a plan to directly address it.  Do not work around it by skipping tests or leaving part of the user's task unaddressed.  Use the same "Problem-Solving Philosophy", "Understanding Requirements" steps and "Multi-Step Task Execution" approach when this happens - it is much more important that we are systematically improving this codebase than completing tasks as quickly as possible.  


## Development Guide

This section contains important details for how developers and AI agents working on this codebase should structure their code and solutions.

### Go Development Standards

#### Code Style and Safety
- You MUST write concurrency-safe code by default. Favor immutability; use mutexes for shared mutable state.
- Prefer sync.Mutex over sync.RWMutex unless explicitly instructed otherwise: nothing at the lavel of abstraction of this problem domain is perforamnce sensitive enough to need separate read + write locks.
- ALWAYS call Unlock()/RUnlock() in a defer statement, never synchronously
- Remember: Go mutexes are not reentrant.  If you need to call logic that requires holding a lock from callers that both already have the lock held and ones that don't, split the functionality in two: e.g. add a `GetTokens()` method that obtains a lock (and defers the unlock), and then a (non-exported) `getTokensLocked()` method with the actual logic, called both from `GetTokens` and any other package-internal callers who themselves hold the lock.  The function's godoc comment should detail _which_ lock is expected to be held, and `*Locked` variants should NEVER be visible to callers outside the package.  The package's public API should not expose the intricacies of locking to users: it should just work.
- Use `go doc` to inspect and understand third-party APIs before implementing
- Run `gofumpt -w .` before committing to ensure proper formatting
- Use `omitzero` instead of `omitempty` in JSON struct tags


#### Testing
- Run tests with `./with_api_keys.sh go test -race ./...` to include integration tests.
- The `with_api_keys.sh` script loads API credentials for all providers (OpenAI, Anthropic/Claude, Google Gemini) into well-known. 
- Use `github.com/stretchr/testify/assert` and `require` for test assertions.
  - Use `require` for fatal errors that should stop the test immediately (like setup failures).
  - Use `assert` for non-fatal assertions that allow the test to continue gathering information.
  - The default error messages are clear enough, avoid adding custom messages to assertions.
- Write table-driven tests with `t.Run()` to test all API variations comprehensively.
- Integration tests should verify both feature availability and actual functionality.


#### Project Conventions
- This is a monorepo with no external package dependencies on it
- When introducing a new abstraction, migrate all users to it and completely remove the old one
- Be bold in refactoring - there's no "legacy code" to preserve


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
2. Execute tools with the provided context (context must flow from Message/MessageStream to handlers)
3. Format results according to provider requirements
4. Make follow-up API calls with tool results
5. Repeat until no more tool calls (max 10 rounds to prevent infinite loops)


#### Provider-Specific Quirks

**OpenAI**:
- ChatCompletions API supports tools; Responses API has reasoning but no tool support
- Automatic retry on rate limits with exponential backoff
- Tool call IDs limited to 40 characters
- Both APIs share the same client implementation
- When emitting tool call events, check for both `tc.ID` and `toolCalls[idx].Function.Name` being non-empty

**Claude**:
- Native thinking/reasoning support through content blocks
- Tool results must be in the same message as tool calls
- Content blocks can mix text, thinking, and tool use
- Requires careful handling of message roles in tool responses
- Tool use events start with `content_block_start` of type `tool_use`

**Gemini**:
- Function calls are "parts" within content
- Different streaming event structure than other providers
- Requires special handling for tool results formatting
- May require multiple parts for complex responses
- Function call arguments need to be marshaled to JSON from `part.FunctionCall.Args`


#### Common Pitfalls to Avoid
- Don't assume streaming events arrive in a specific order or completeness
- Tool call IDs have provider-specific constraints
- Some providers require tools to be included in follow-up requests, others don't
- Message history formatting varies significantly between providers
- Not all provider features are available in all API modes
- Token counting fallbacks should only be used when the API doesn't provide usage
- Remember to track cumulative token usage across multiple messages in a conversation


### Project Maintenance

- Keep provider-specific logic isolated in provider packages
- Shared code goes in `llm/internal/common/` (like `RegisteredTool`)
- Model limits and configurations stay within provider packages
- Update integration tests when adding new features
- Maintain backward compatibility for the public API
- When updating multiple providers, ensure consistency across all implementations
