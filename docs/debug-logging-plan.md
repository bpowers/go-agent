# Debug Logging Overhaul Plan

## Objectives
- Replace existing ad-hoc debug toggles in provider clients with a structured logging system powered by Go's `slog` package.
- Introduce an integer-based debug detail level (e.g., 0 = off, higher numbers increase verbosity) instead of the current binary `debug` flag and `GO_AGENT_DEBUG` environment variable.
- Ensure consistent, configurable logging behavior across OpenAI, Claude, and Gemini providers while preserving compatibility for existing users.

## Current State Assessment
1. **Binary debug flags**
   - Each provider client (`llm/openai`, `llm/claude`, `llm/gemini`) exposes `WithDebug(bool)` options or reads `GO_AGENT_DEBUG` directly.
   - Logging behavior is inconsistent (mix of environment variables, option flags, and `fmt.Println`).
2. **Scattered log statements**
   - Debug output is wired in-line across streaming handlers, making it hard to adjust verbosity or redirect output.
3. **Lack of centralized logging configuration**
   - No shared logger or configuration object; consumers cannot plug in their own logger or control log levels per component.

## Proposed Architecture
1. **Central logging configuration**
   - Introduce a package-level `Logger` interface backed by `slog.Logger`.
   - Provide a central configuration struct (e.g., `logging.Config`) that includes:
     - `Logger *slog.Logger`
     - `DetailLevel int`
     - Default handlers (e.g., JSON/Text) with sane defaults.
   - Expose helper constructors for common setups (stderr text handler, JSON handler, no-op handler).

2. **Detail level semantics**
   - Define canonical detail tiers:
     - `0`: logging disabled (use a `slog.New` with `slog.LevelError + 1` or a no-op handler).
     - `1`: high-level lifecycle events (request start/finish, provider selection).
     - `2`: streaming metadata (chunk types, tool invocation summaries).
     - `3`: verbose payloads (raw SSE events, deltas, tool call arguments/responses).
   - Ensure log records include structured keys (e.g., `provider`, `model`, `event_type`, `detail_level`).

3. **Configuration propagation**
   - Extend provider client constructors/options to accept a `logging.Config` (or pointer) instead of `WithDebug` bools.
   - Update context helpers (`chat.WithDebugDir`) to coordinate with the new logging config when writing payloads to disk.
   - Ensure default behavior preserves "no logs" unless users opt-in (detail level 0 by default).

4. **Implementation tactics per provider**
   - **OpenAI**: replace `debugSSE` checks with detail level comparisons; emit structured events via `logger.Log` with context information.
   - **Claude**: mirror the OpenAI approach, ensuring parity in event naming and payload redaction.
   - **Gemini**: retrofit streaming loops to use the shared logging helpers.
   - Factor out shared logging helpers (e.g., `logging.LogEvent(ctx, level, msg, attrs...)`).

5. **Environment variable compatibility**
   - Deprecate `GO_AGENT_DEBUG`; map it to detail level (e.g., set level 2 when set to `1`) to avoid breaking existing deployments.
   - Emit a one-time warning log when the env var is used, guiding users to the new configuration.

## Implementation Steps
1. **Bootstrap logging package**
   - Add `logging` package with config struct, default logger setup, helper functions for detail gating, and environment variable shims.
   - Include unit tests for level gating and env var parsing.
2. **Update provider constructors**
   - Replace `WithDebug` options with `WithLoggingConfig(*logging.Config)` variants.
   - Keep temporary shims: `WithDebug(bool)` calling into new config with detail level `2` for backward compatibility, marked deprecated.
3. **Refactor debug statements**
   - Sweep through each provider's streaming logic, replacing `if debug` blocks with `logging.IfLevel(ctx, level, func(logger *slog.Logger))` style helpers.
   - Ensure sensitive payloads are redacted or summarized when logging at lower detail levels.
4. **Adjust CLI and examples**
   - Update `examples/agent-cli` flags to accept an integer `--log-detail` option.
   - Document usage in `README.md` (table of detail levels, examples of enabling JSON logs).
5. **Context integration**
   - Evaluate integration with `chat.WithDebugDir` to optionally log file paths or include detail-level-controlled dumps.
6. **Testing & validation**
   - Unit tests for logging package.
   - Provider-specific tests asserting that enabling detail level `>=2` triggers logging hooks (likely via injectable handler capturing records).
   - Manual smoke test using `examples/agent-cli` with different detail levels.

## Migration Considerations
- Provide clear documentation on the new configuration API and how to migrate from `GO_AGENT_DEBUG` / `WithDebug`.
- Maintain backward compatibility for one release cycle with deprecation warnings.
- Communicate potential performance implications of high detail levels (especially logging raw payloads).

## Open Questions
- Should detail level be global or per-provider? (Plan assumes global config with optional overrides.)
- Should we expose structured logging attributes for tool events (tool name, latency)?
- How to redact PII or secrets in highly detailed logs—need guidelines or automatic filtering.

