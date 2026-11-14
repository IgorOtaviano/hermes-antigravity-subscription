# Antigravity Subscription DirectSDK Provider for Hermes Agent

<p align="center">
  <img src="assets/banner.jpg" alt="Antigravity Subscription DirectSDK Banner" width="100%" />
</p>

[![Hermes Agent Plugin](https://img.shields.io/badge/Hermes%20Agent-Plugin-purple.svg)](https://hermes-agent.nousresearch.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tested with Hermes](https://img.shields.io/badge/Hermes-v0.21%2B-green.svg)](https://github.com/nousresearch/hermes-agent)

Hermes Agent model provider that routes inference through an active Google Antigravity account using the local `agy` binary.

This plugin lets Hermes use Gemini and Claude models through your existing Antigravity quota.

---

## How It Works

- **Token streaming**: Reads stdout chunks from `agy --output-format stream-json` and yields standard completion deltas. On Hermes versions that support `ProviderProfile.supports_streaming`, the profile opts in, so Hermes consumes these deltas as they arrive instead of waiting for the assembled response. Older Hermes versions keep the non-streaming path.
- **Stream timeouts**: Streaming calls use the read budget of the `httpx.Timeout` Hermes passes (for `agy://local`, normally `HERMES_API_TIMEOUT`, 1800 s) as the turn deadline. Scalar timeouts are unchanged; with no timeout the 300 s default applies. A turn that ends without agy's `result` event reports why reading stopped (`stream deadline` or `stdout closed`), the elapsed time, and the age of the last stdout line.
- **Tool call routing**: The model emits `<tool_call>` tags in text. The plugin parses these into OpenAI function call deltas, so Hermes executes tools on the host instead of `agy`. A literal `</tool_call>` inside a JSON argument does not end the frame, multiple calls in one turn get unique indices and ids, and a stream that ends inside an unfinished block fails the turn instead of returning the raw block as assistant text.
- **Early quota detection**: While waiting for the first content event, a log watcher aborts the turn on `RESOURCE_EXHAUSTED`/429 or credit-cap lines written during that turn. Log mtimes are compared in wall-clock time; `init` alone does not disarm the watcher, the first non-init event does.
- **Tool execution guard**: `agy` registers local system tools by default. This plugin runs `agy` in headless mode without permissions skip flags. If `agy` attempts to execute an internal tool step, the stream closes and terminates the child process.
- **Thinking effort mapping**: Maps Hermes reasoning effort settings (`low`, `medium`, `high`) to backend model variants (`gemini-3.8-flash-low`, `gemini-3.8-flash-high`).
- **Subagent concurrency**: Each completion turn runs in its own process group (`start_new_session=True`). Multiple Hermes subagents can request completions concurrently without shared state.
- **Windows console suppression**: On Windows, every console-subsystem child (`agy`, `cmdkey`, `taskkill`, and `agy models`) is spawned with `CREATE_NO_WINDOW`, so a GUI-hosted Hermes without a console of its own gets no visible console window per call. `AGY_CLI_DISABLE_AUTO_UPDATE=true` is set in the child environment so `agy`'s background updater cannot open one either.
- **Filesystem isolation**: Subprocesses run in an isolated, private temporary working directory per client and slash commands are disabled. Local `GEMINI.md` and `AGENTS.md` project files are not read.
- **Long inputs in several messages**: `agy` cuts a single message between 183,000 and 194,000 UTF-8 bytes and drops the rest without an error, while a conversation keeps everything it was told. On the persistent worker a prompt over 140,000 bytes is delivered as consecutive messages of the same conversation, so tool results, history and tool schemas arrive whole. See [Long inputs](#long-inputs-multipart-delivery).
- **Tool output pruning**: applies to a single-message send only (multipart off, or the oneshot path). The prompt builder keeps the last 8 tool results intact and caps older tool outputs at 300 characters, and a 20k character limit with head/tail preservation guards individual tool calls. With multipart on, one tool result is kept up to 400,000 characters and older results are reduced only when the history exceeds 800,000 characters.
- **Error classification and recovery**: `classify_api_error` detects `agy` PubSub stalls (`subscriber fell behind updates, stalled for 5s`), trajectory limits, and empty SUCCESS status (returning `context_overflow` with `should_compress: True`). It classifies credit balance exhaustion and billing caps as `billing` for immediate failover, and maps account verification and Terms of Service notices to `auth_permanent` to surface resolution links without retry loops.

---

## Models

| Model | Suffix | LLM Context | Plugin Declared | Supported Efforts |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-3.8-flash` | `-low`, `-medium`, `-high` | 1M tokens | 239k tokens | `low`, `medium`, `high` |
| `gemini-3.7-flash` | `-low`, `-medium`, `-high` | 1M tokens | 239k tokens | `low`, `medium`, `high` |
| `gemini-3.6-flash` | `-low`, `-medium`, `-high` | 1M tokens | 239k tokens | `low`, `medium`, `high` |
| `gemini-3.1-pro` | `-low`, `-high` | 2M tokens | 239k tokens | `low`, `high` |
| `gpt-oss-120b` | `-medium` only | 128k tokens | 128k tokens | `medium` |
| `claude-sonnet-4-6` | None | 200k tokens | 239k tokens | None (agy rejects `--effort`) |
| `claude-opus-4-6-thinking` | None | 200k tokens | 239k tokens | None (agy rejects `--effort`) |

**Dynamic discovery per account.** Available models and supported reasoning efforts are discovered dynamically from `agy models`. The picker reflects the exact models available on the user's account and subscription tier (including preview models or custom models configured in `agy` settings). A model that agy lists by bare name takes no `--effort`. A requested effort that the model lacks maps to the nearest supported one, the stronger on a tie (`medium` on `gemini-3.1-pro` becomes `high`). The result is cached for one hour; `models.py` provides an emergency fallback when `agy models` fails. Hermes' own `xhigh` and `max` map to `high`.

> **Claude Model Aliases**: Requests for `claude-sonnet-5-5` or `claude-opus-5-5` resolve dynamically if present in the user's `agy models` catalog; if the account lacks them, they gracefully fall back to `claude-sonnet-4-6` and `claude-opus-4-6-thinking`.

> **Trajectory Persistence & Context Length**: Multi-turn sessions persist across worker restarts via `agy --conversation <id>`, sending only incremental deltas on each turn. This bypasses `agy`'s single-turn 100,000-token prompt trimmer. The plugin declares 239,000 tokens to Hermes (configurable via `HERMES_ANTIGRAVITY_CONTEXT_LENGTH` or `ANTIGRAVITY_CONTEXT_LENGTH`), ensuring Hermes auto-compaction triggers safely below `agy`'s internal checkpoint threshold (239,616 tokens / 256k - 16k output tokens) and preserves sovereign context control. Conversation databases are strictly isolated inside the worker's private temp directory and automatically purged on session resets (`/new`).

### Prompt recovery and delivery diagnostics

Tool-result turns restate the latest real user request in both full prompts and
worker deltas. Empty-response recovery nudges remain in the transcript, but do not
replace that request as the active objective. The provider recognizes the exact
known core nudge when private message markers have been stripped by the transport.
A new real user message still supersedes the previous objective.

Output-token-limit errors are retryable server errors without context compression.
Trajectory/context exhaustion continues to request compression.

The delivery warning from `delivery.py` is a heuristic, not a delivery guarantee.
It compares only fresh-context full sends on the same model/effort route. Worker
deltas and concurrent oneshots are excluded because their character counts cannot
be compared with whole-context usage. Cached input counts toward the diagnostic,
without changing the usage reported to Hermes. Missing usage does not erase the
previous valid sample. Set `ANTIGRAVITY_DISABLE_DELIVERY_WATCH=1` to disable it.

Blind spots: the first request, stable-size truncation, and normal incremental tool
loops. Changes in text density can still cause false positives. This warning does
not prevent or repair the tail truncation tracked in
[issue 29](https://github.com/soyelmismo/hermes-antigravity-subscription/issues/29).
Multipart delivery does, on the persistent worker. The oneshot path (a request
that arrives while the worker is busy) still sends one message: there the
120,000 character budget elides the middle so the head and the request survive.
Reasoning text in `stream-json` remains blocked upstream: `agy` exposes no flag for it
([issue 31](https://github.com/soyelmismo/hermes-antigravity-subscription/issues/31)).

### Model Compatibility

`agy` embeds its own system prompt ("you are Antigravity, you have `run_command`, `view_file`, ...") before the plugin's preamble. Some models obey `agy`'s prompt over the plugin's `<tool_call>` protocol and invoke native tools, which headless mode soft-denies. The plugin detects this and terminates the stream, but the turn is lost.

| Model | `<tool_call>` Protocol | Notes |
| :--- | :--- | :--- |
| `gemini-3.8-flash` | Follows | Tested with full toolset (~40 tools) |
| `gemini-3.7-flash` | Follows | Tested with full toolset |
| `gemini-3.6-flash` | Ignores | Goes native even with a single tool schema |
| `gemini-3.1-pro` | Follows | Tested with full toolset |
| `claude-opus-4-6-thinking` | Follows | Tested with full toolset |
| `claude-sonnet-4-6` | Partial | Works with moderate toolsets (tested up to 10 tools); refuses or flips to native with large toolsets (~40 tools) |
| `gpt-oss-120b` | Follows | Tested with full toolset |

Models that ignore the protocol attempt `agy`'s native `RunCommand`/`WriteToFile` steps. The plugin neutralizes these (stream watcher kills the process), but every tool-using turn fails. This is an upstream `agy` limitation: the persona prompt is embedded in the closed binary with no configuration surface to suppress it.

---

## Environment passed to `agy`

Hermes often runs with gateway and dashboard secrets in its environment (`SLACK_BOT_TOKEN`, `TELEGRAM_BOT_TOKEN`, `*_API_KEY`, ...). `agy` is a closed-source binary and does not need any of them, so by default the child environment is the parent environment **minus credential-looking variables**: names containing `TOKEN`, `SECRET`, `PASSWORD`, `API_KEY`, `ACCESS_KEY`, `PRIVATE_KEY`, `CREDENTIAL(S)`, `AUTH`, `WEBHOOK`, `DSN` or `COOKIE`, and database/broker URLs (`DATABASE_URL`, `REDIS_URL`, `MONGO_URI`, ...). `SSH_AUTH_SOCK`, proxy and certificate variables are kept.

This default is a **name heuristic**: it cannot know what an arbitrarily named variable holds. For a guarantee of exactly what `agy` sees, use strict mode.

| Variable | Effect |
| :--- | :--- |
| `ANTIGRAVITY_ENV_PASSTHROUGH` | Comma-separated names to keep even if they look like credentials. |
| `ANTIGRAVITY_ENV_STRICT` | `1`, `true`, `yes` or `on`: pass only the baseline below plus `ANTIGRAVITY_ENV_ALLOWLIST` and `ANTIGRAVITY_ENV_PASSTHROUGH`. |
| `ANTIGRAVITY_ENV_ALLOWLIST` | Comma-separated names to add in strict mode. |

Strict baseline: `PATH`, `LANG`, `LANGUAGE`, `LC_*`, `TZ`, `TERM`, `TMPDIR`/`TEMP`/`TMP`, `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` (and their lowercase spellings on POSIX), `SSL_CERT_FILE`, `SSL_CERT_DIR`, `SSH_AUTH_SOCK`, `DBUS_SESSION_BUS_ADDRESS` and `XDG_RUNTIME_DIR` (Linux keyring sign-in), `SYSTEMROOT`, `WINDIR`, `COMSPEC`, `PATHEXT` (Windows), `AGY_CLI_DISABLE_AUTO_UPDATE` and `AGY_CLI_MODEL_API_MAX_RETRIES`. Names are matched case-insensitively on Windows.

---

## Security Model

```
┌─────────────────────────────────────────────────────────────┐
│                        Hermes Agent                         │
│  - Prompts and system instructions                          │
│  - Tool definitions and approvals                           │
│  - Host execution                                           │
└──────────────┬───────────────────────────────▲──────────────┘
               │ Prompt + tool schemas         │ Text with <tool_call> tags
               │                               │ parsed into completion chunks
┌──────────────▼───────────────────────────────┴──────────────┐
│       Antigravity Subscription DirectSDK Plugin             │
│                                                             │
│  1. Preamble tells the model to emit tool tags in text.     │
│  2. Omits --dangerously-skip-permissions to deny agy tools. │
│  3. Stream watcher kills agy if it emits a tool step.       │
│  4. Runs in private temp dir to ignore rules files.         │
└──────────────────────────────┬──────────────────────────────┘
                               │ Stdin / stdout (NDJSON pipes)
┌──────────────────────────────▼──────────────────────────────┐
│               Local `agy` CLI binary (Go)                   │
└─────────────────────────────────────────────────────────────┘
```

## Turn-home hardening

The CLI reads its home once, at spawn, so everything below is written right before every worker and every oneshot process (`process.prepare_agy_home`). The behaviors are ported from [agentes.chat](https://github.com/agentes-chat/agentes-chat)'s `agy-home.ts` and `brain-antigravity.ts` (ADR 0143), where the same CLI is driven in production.

- **Planted config is dropped.** A turn can leave global agents, skills, rules, hooks, plugins, project settings and the `GEMINI.md` / `AGENTS.md` rule files in the CLI's home, which a later turn would then load as its own config — outside every tool and outside the `<tool_call>` protocol this plugin depends on. `AGY_PLANTED` lists the same set agentes.chat removes. The cleanup stays inside the per-client isolated home: it is skipped when that home resolves to the real home or when its `.gemini` is a link, and a linked `.gemini/config` is unlinked without entering its target, so a user's real `~/.gemini` is never deleted through a link.
- **`settings.json` carries the hardening.** `useG1Credits: false` (a turn past the plan's quota fails instead of silently spending the account's AI credits; set `ANTIGRAVITY_USE_CREDITS=1` to opt a paid account back in), `enableTerminalSandbox: true`, and the three headless-noise switches off (`notifications`, `showTips`, `showFeedbackSurvey`). Everything the plugin does not own is preserved: the sign-in's `auth`, an Enterprise seat's `gcp`, and the per-file media `permissions.allow` rules. `toolPermission` and `allowNonWorkspaceAccess` are deliberately left alone — this plugin wants agy's native tools refused and needs out-of-workspace paths readable for the media attach. A failed write is logged as a warning: that process then runs with the account's own settings.
- **`--sandbox` on every spawn.** agy's terminal restrictions, the flag agentes.chat passes on every turn. It bounds the one case the stream watcher cannot prevent: a model that ignores the protocol and asks for a native terminal step in the window before the process is killed. File reads (the media attach) are permission-gated, not terminal, so they are unaffected; an explicit `ANTIGRAVITY_ARGS` still wins verbatim.
- **`AGY_ERROR` is parsed.** The CLI's own failure line (`AGY_ERROR: {"short_error": …}` on 1.5.8, the richer status object on older builds) becomes the turn's message, with the plan's reset words when it names them, and quota exhaustion is classified as a rate limit so Hermes backs off and fails over instead of retrying a quota that returns in hours. A limit reported only by status (`RESOURCE_EXHAUSTED` / 429) keeps that status in the message, so the verdict does not depend on the wording. `result.finished_reason: "print_timeout"` reads as a timeout instead of an empty result.
- **A lost conversation fails loudly.** `agy` only warns when it doesn't know the `--conversation` it was told to resume, then starts a new one — which would silently drop the context a continuation turn's delta-only prompt relies on. The turn fails with `AntigravityConversationLost` instead. A non-streaming caller retries once as a fresh conversation with the full history, after taking the worker lock back. A streaming caller receives the error, and the next attempt is a full send. A continuation that already fell back to a fresh conversation (broken worker pipe) expects no id.

## Long inputs: multipart delivery

`agy` cuts one user message at a fixed size and reports nothing: the turn completes with exit code 0. Measured on agy 1.5.9, same result on `gemini-3.8-flash-low` and `gemini-4-hydrogen-medium`:

| Single message | Arrived |
|---|---|
| 334,558 chars of English prose | cut at about 186,000 bytes |
| 337,956 chars of Python source | cut between 183,100 and 196,800 bytes |
| 337,956 chars of accented Portuguese (389,148 bytes) | cut between 178,100 and 194,300 bytes |
| 167,916 chars of digit tables (177,208 input tokens) | whole |

The limit is in bytes of the message, not in tokens. A conversation holds far more than one message does: four messages of 139,000 characters each in one conversation (151,755 input tokens) were recalled completely.

So the worker path does not shorten a long prompt. `split_prompt_parts` cuts it into parts of at most `HERMES_AGY_PART_MAX_BYTES` (140,000, frame included), on blank lines where possible, then on line ends. Every part but the last is written as its own turn that asks only for `ACK`; the plugin reads that turn to its end and discards the reply. The last part, filled from the end so the request travels with the newest context, is the turn Hermes sees. Usage of all parts is reported with that turn, and the acknowledgements never enter the history Hermes continuations are matched against.

What this changes, live on agy 1.5.9. The first two rows go through the plugin client, the third through a Hermes agent loop with its file tools:

| Case | Before | After |
|---|---|---|
| Full send, 536,202 chars (47 tool schemas, 16 documents) | middle elided to fit 120,000 chars | 4 parts, 32 of 32 markers recalled, 23 s on `gemini-3.8-flash-low`, 138 s on `gemini-4-hydrogen-medium` |
| Tool result of 334,093 chars in a tool loop | reduced to 15,000 chars | 3 parts, 24 of 24 markers recalled, 16 s and 25 s |
| Hermes review of 12 files, 10,800 lines, one marker hidden in each (`gemini-3.8-flash-medium`, at most 4 files per turn) | 4 of 12 markers, 14 model calls, 52 file reads, 188 s | 12 of 12 markers, 4 model calls, 12 file reads, 40 s |

Costs and limits:

- Each extra part is one more model turn: 4 to 7 seconds on Flash, more on a thinking model, plus the input tokens of the conversation so far (served from cache after the first part).
- The whole input is still bounded: past `HERMES_AGY_WORKER_PROMPT_MAX_CHARS` (800,000) older tool results are reduced to stubs and then the middle of the transcript is elided. The declared context window (239,000 tokens) is unchanged.
- A request restated beside the transcript (`ACTIVE OBJECTIVE`, `USER REQUEST THIS TOOL LOOP IS SERVING`) is bounded to its first 3,000 and last 600 characters when it is longer than 4,000. The request itself is always delivered whole, once.
- A part whose turn does not complete fails the whole turn; the worker is terminated and the next attempt is a full send.
- The oneshot path is not multipart. It has no conversation to deliver parts into.
- Hermes itself bounds the tool results of one turn at 200,000 characters (`tools/budget_config.py`, not configurable): past that it saves the largest results to disk and sends a preview with the path. A model that reads 12 files of 32,000 characters in one turn receives 5 of them whole. This is the same for every provider; the reviewer has to read fewer files per turn or page the saved results.
- `HERMES_AGY_PART_MAX_BYTES=0` turns multipart off and restores the single-message budgets.

## Multimodal media

Image, video and audio attachments are embedded natively: the plugin materializes each attachment (local path, `file://` URI or remote URL) to a cache file and references it in the prompt as `[Image](/abs/path)`, which `agy` reads through its native `read_file` tool and uploads to Google as real pixels. No transcription, no auxiliary descriptions.

Hardening of the materialization path (`prompt.py`):

- Remote downloads only from publicly fetchable hosts: loopback, private, link-local, reserved, multicast and metadata addresses are refused, re-validating every redirect hop.
- Downloads stream to a `.part` file with a hard cap (25 MB) and are renamed into the cache only after passing the cap.
- Only `image/*`, `video/*` and `audio/*` content types are accepted; local paths and `file://` URIs must carry a known media extension. Rejected media is dropped fail-closed, never degraded to a raw reference.
- The cache lives under Hermes' own home (`get_hermes_home()/cache/agy_media`) and files are deleted when the session closes.

`agy` reads `settings.json` permissions only at spawn. When a turn references media the live worker's permission set does not cover, the plugin writes exact `read_file(<path>)` rules and respawns the worker with `--conversation`, preserving the agy conversation (verified end to end: red image on turn 1, worker respawn, blue image on turn 2, both read natively).

The plugin also declares vision capability to Hermes' image router (`agent/models_dev` canonical schema): `gemini-*` and `claude-*` model ids (any effort suffix, aliases resolved, future families included) advertise `supports_vision`, so with `agent.vision_capability_first: true` user-attached images attach natively to the plugin's models and the configured `auxiliary.vision` describer stays reserved for text-only models. The declaration is prefix-based and does not depend on the agy model catalog being warm.

## Ephemeral sessions and cleanup

Every client runs `agy` with an isolated `HOME` inside a temp dir (`hermes_agy_*`); the user's real `~/.gemini` sessions, conversation index and brain are never touched. Cleanup layers:

1. `close()` kills the worker and removes the temp dir plus any materialized media files.
2. A Linux `/proc` sweep on client init (`reap_orphaned_plugin_workers`) terminates leftover plugin workers whose owning temp dir was already deleted — a Hermes process that dies without `close()` would otherwise leave an orphaned worker (PPID 1) holding a conversation open on the account; a 17-hour orphan was observed tangling other sessions' turns (2026-11-11).
3. Leftover temp dirs under Hermes' scratch are pruned by the normal 24h idle sweep.

Footprint is ~14 MB per live session (conversation DBs and logs). Nothing persists on disk after close, and nothing is created server-side beyond the account's normal conversations.

Multiple agy sessions on one account contend for the shared OAuth token: whoever refreshed last wins, and the loser logs "You are not logged into Antigravity" / "error getting token source" and produces no result. The plugin re-syncs the isolated token before every spawn and retries a turn once when the exact contention signature appears. Concurrent cascades are serialized server-side by subscription capacity: a second simultaneous session can surface a 503 "No capacity available" until the first turn finishes.

---

## Requirements

1. **Antigravity CLI**:
   `agy` must be installed and authenticated on the system.
   ```bash
   agy --version
   ```
2. **Hermes Agent**:
   Version `0.21.0` or higher.
   ```bash
   hermes --version
   ```

> **Optional: Linux keyring detection.**
> On Linux with a D-Bus session, `agy` keeps its session in the freedesktop
> Secret Service instead of a token file (credentials `service`/`gemini` +
> `username`/`antigravity`, the go-keyring convention). Detection uses
> `secret-tool`, falling back to `python3` with `secretstorage` when the CLI is
> absent or cannot answer (missing, spawn failure, timeout or non-zero exit).
> `secret-tool` prints the credential on its own stdout, so the plugin closes
> that stream and reads only the attribute lines on stderr; the scripted
> fallback reads attributes and labels and never asks for the secret. Either way
> the plugin never receives the stored secret. Installing `secret-tool` is not
> strictly neutral: the CLI path answers on the exact attribute pair, while the
> scripted path additionally matches a credential whose go-keyring label embeds
> "antigravity". Headless systems without a D-Bus session keep using the token-file
> path, so nothing extra is required there.
> ```bash
> command -v secret-tool   # optional: faster than the python3 fallback
> ```
> Install it with `libsecret-tools` (Debian/Ubuntu) or `libsecret` (Fedora/Arch).

> **SSH session variables.**
> `agy` switches to file-based token storage when it detects `SSH_CONNECTION`,
> `SSH_CLIENT`, or `SSH_TTY`. When no token file exists (credential lives in the
> OS keyring), the plugin strips these variables from the child environment so
> `agy` uses keyring authentication instead of failing. `SSH_AUTH_SOCK` is
> preserved. When a token file exists, the environment is unchanged.

> **macOS keychain.** On macOS `agy` keeps its session in the login keychain
> (a generic password with service `gemini` and account `antigravity`).
> Detection runs `/usr/bin/security find-generic-password` without `-g` or
> `-w`, which prints only the item's attributes, so the plugin never receives
> the secret and no keychain prompt appears. The keychain is found through
> `$HOME/Library/Keychains`, so the plugin links that directory into the
> isolated HOME it gives `agy`; otherwise the child `agy` would ask you to log
> in again. Nothing extra is required. Setting `ANTIGRAVITY_CONFIG_DIR` turns
> both off and keeps the token-file path.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HERMES_ANTIGRAVITY_CONTEXT_LENGTH` | `239000` | Override the declared context window (tokens). Also checks `ANTIGRAVITY_CONTEXT_LENGTH`. Hermes triggers auto-compression based on this value (e.g. at 75% threshold, ~179k tokens). |
| `HERMES_AGY_PART_MAX_BYTES` | `140000` | Largest single message to `agy`, in UTF-8 bytes. A longer prompt is delivered in several messages on the worker path. `0` turns multipart delivery off. Keep it under the 183,000 bytes where `agy` starts cutting. |
| `HERMES_AGY_WORKER_PROMPT_MAX_CHARS` | `800000` | With multipart on: size of the whole input past which older tool results are reduced and the middle of the transcript is elided. |
| `HERMES_AGY_TOOL_RESULT_MAX_CHARS` | `400000` | With multipart on: size of one tool result past which its middle is cut. |
| `HERMES_AGY_PROMPT_MAX_CHARS` | `120000` | Single-message budget (multipart off, and the oneshot path): past it the middle of the transcript is elided. `0` disables it. |
| `ANTIGRAVITY_COMMAND` | `agy` | Path to the `agy` binary. Also checks `AGY_CLI_PATH` and `ANTIGRAVITY_CLI_PATH`. |
| `ANTIGRAVITY_ARGS` | (none) | Extra arguments to pass to the `agy` subprocess. |
| `ANTIGRAVITY_CONFIG_DIR` | (none) | Override the config directory, bypassing keyring and token-file detection. |
| `ANTIGRAVITY_DEBUG_PROMPT_DIR` | (none) | Directory to dump outgoing prompts sent to `agy` for offline diagnosis. |
| `ANTIGRAVITY_WORKER_IDLE_SECONDS` | `900` | Terminate the persistent `agy` worker after this many seconds without a finished turn. The next turn starts a fresh worker with the full prompt. `0` or a negative value disables the bound. |
| `ANTIGRAVITY_WORKER_ABANDON_SECONDS` | one idle window | Terminate a worker that stays silent **and** burns no CPU for this long while its turn still holds the worker lock — the state an abandoned stream leaves behind, because its `finally` runs only when the reference cycle is collected. Silence alone is not enough: a live turn can be quiet for minutes while it reasons, and callers may set a read deadline longer than the grace, so the check also requires under `1` CPU-second of progress since the worker last spoke (Linux only, `/proc/<pid>/stat`; where CPU time cannot be read the worker is left alone). The default is one idle window floored at one read deadline plus a minute (`360`); an explicit value is honoured verbatim. `0` or a negative value disables the check and restores the old re-arm-forever behaviour. |
| `ANTIGRAVITY_PROXY` | (none) | Proxy URL (e.g. `http://127.0.0.1:8080` or `socks5h://127.0.0.1:1080`) agy's subprocess should use. Sets `HTTP(S)_PROXY` and `ALL_PROXY` (both casings) and adds loopback to `NO_PROXY`. Unset means agy inherits the parent process's proxy environment unchanged. |
| `ANTIGRAVITY_ENV_PASSTHROUGH` | (none) | Comma-separated variable names to preserve in the child environment even if they match secret heuristics. |
| `ANTIGRAVITY_ENV_STRICT` | `false` | When enabled (`1`, `true`), only passes variables from `ANTIGRAVITY_ENV_ALLOWLIST` plus a minimal runtime baseline. |
| `ANTIGRAVITY_ENV_ALLOWLIST` | (none) | Additional variables to pass through when `ANTIGRAVITY_ENV_STRICT` is active. |

---

## Installation

Install through the Hermes plugin manager:

```bash
hermes plugins install https://github.com/soyelmismo/hermes-antigravity-subscription
hermes plugins enable antigravity-subscription-directsdk
```

Or clone into the local plugins folder:

```bash
git clone https://github.com/soyelmismo/hermes-antigravity-subscription.git \
  ~/.hermes/plugins/antigravity-subscription-directsdk

hermes plugins enable antigravity-subscription-directsdk
```

> **Directory naming tip**: Hermes Agent discovers plugins by scanning directories in `~/.hermes/plugins/` alphabetically. If you keep backup copies (such as `antigravity-subscription-directsdk.backup`), place them outside `~/.hermes/plugins/` so a later-sorting directory does not override the active plugin.

---

## Usage

### Interactive model picker

Open the model picker inside Hermes:

```text
/model
```

Select **Antigravity Subscription DirectSDK**, then pick a model and effort level.

### Command line flag

```bash
hermes --provider antigravity-subscription-directsdk --model gemini-3.8-flash
```

### Configuration file

Add to `~/.hermes/config.yaml`:

```yaml
agent:
  provider: antigravity-subscription-directsdk
  model: gemini-3.8-flash
  reasoning_effort: high
```

### Subscription Quota & Usage Tracking

The provider profile implements `fetch_account_usage()`, exposing subscription quota limits to Hermes `/usage`, the TUI status bar, and the desktop app via `AccountUsageSnapshot`.

Note: Custom plugin tools and slash commands are not available for `kind: model-provider` plugins because Hermes core skips calling `register(ctx)` for model providers (`hermes_cli/plugins_discovery.py:286`, `hermes_cli/plugin_validate.py:253-258`). All quota visibility is provided through the native `fetch_account_usage()` interface.

Quota queries execute `agy -p "/usage" --output-format json` under an isolated HOME environment without consuming any model tokens or inference turns. Results are cached thread-safely for 60 seconds. Refreshes use single-flight execution so concurrent callers share a single subprocess. An unforced query failure serves the stale cache. Distinct quota groups reported by `agy` (such as Gemini 4 Argon on paid plans alongside pooled Gemini Models) retain distinct labels in `/usage`.

---

## Tests

Run the test suite:

```bash
PYTHONPATH=/usr/local/lib/hermes-agent:. pytest
```

The suite is hermetic. `conftest.py` isolates two host facts that would
otherwise decide which copy of the plugin answers `get_provider_profile()`:

- `$HERMES_HOME/plugins/` may hold another copy of this plugin, for example on a
  development box where the plugin is installed. Hermes discovers it on the
  first lookup and re-registers it under the same name, so field differences
  between the two copies fail assertions that pass on a clean checkout.
- The repository root is itself a Python package, because the plugin entry
  point is `__init__.py`. pytest imports it while collecting and the test
  modules import it again under their own names. Each import runs
  `register_provider()` and the registry is last-writer-wins, so which copy
  answers depends on collection order.

`HERMES_HOME` is pointed at an empty throwaway directory for the first case,
and the source tree's profile is re-registered before every test for the
second.

---

## License

MIT
