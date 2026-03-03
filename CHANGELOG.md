# CHANGELOG


## v0.4.0 (2026-03-03)

### Features

- Add docs sync trigger ([#8](https://github.com/OpenAdaptAI/openadapt-consilium/pull/8),
  [`25d4326`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/25d4326a609142f8185bd2e38a678d34bc05107c))


## v0.3.2 (2026-03-02)

### Bug Fixes

- Rename PyPI package to openadapt-consilium
  ([#6](https://github.com/OpenAdaptAI/openadapt-consilium/pull/6),
  [`6104726`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/61047268fc63070bffc04d8521278b50a9b091d0))

The `consilium` name on PyPI is owned by another project. Rename to `openadapt-consilium` to match
  the repo name. The Python import name remains `consilium` (unchanged).

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>


## v0.3.1 (2026-03-02)

### Bug Fixes

- Resolve lint errors and restore version to 0.3.0
  ([#4](https://github.com/OpenAdaptAI/openadapt-consilium/pull/4),
  [`cd54594`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/cd5459423f4092d50dd874bea2469652e234d733))

- Remove unused imports: `Any` and `DEFAULT_MODELS` from core.py, `sys` from __main__.py - Remove
  unused variable assignment in test_council.py - Restore version to 0.3.0 (reverted by erroneous
  semantic-release run)

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>

- Update URLs from abrichr/consilium to OpenAdaptAI/openadapt-consilium
  ([#2](https://github.com/OpenAdaptAI/openadapt-consilium/pull/2),
  [`73b6474`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/73b647475b7fc2ed00f523c0c9983fa2f7bcc373))

The repo was renamed from abrichr/consilium to OpenAdaptAI/openadapt-consilium. Update all 3
  install/clone URLs in README.md to point to the correct location.

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>


## v0.3.0 (2026-03-02)

### Bug Fixes

- Exclude audio/realtime models from tier classification + fix google genai import
  ([`b1b5522`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/b1b55220f3ae10fbbe238fe86526dcba912fc44a))

- OpenAI tier regex now requires version-like pattern (gpt-N.N) instead of matching any gpt-*
  prefix. This prevents gpt-audio-2025-08-28 from being classified as "flagship" due to its date
  sorting higher than 5.2. - Google model listing now uses google.genai (new SDK) instead of the
  deprecated google.generativeai which lacks Client(). - Added tests for audio/realtime model
  exclusion.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Handle date-suffixed model IDs from provider APIs
  ([`24d68dc`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/24d68dc0bef17b0703283e329cc30fdc110e5840))

get_default_models() now returns "provider/model_id" format so parse_model_string() can resolve any
  model ID returned by the API (e.g. claude-opus-4-20250514). Also adds pattern-based provider
  inference as a safety net for bare model IDs.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Migrate Google provider from deprecated google-generativeai to google-genai
  ([`e3619ad`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/e3619add62364ffe0b9a98b3d464ef83ed7f5478))

Replaces the deprecated `google.generativeai` SDK with the new `google-genai` SDK (`from google
  import genai`). Uses the unified `Client` API for model queries. Eliminates the FutureWarning
  about the deprecated package.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Version extraction for hyphen-separated model IDs + stale references
  ([`2787905`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/278790563b686c8e0461c0ad34944d2ce9d4d376))

- Fix _extract_version_tuple regex to handle hyphen-separated versions (e.g. claude-opus-4-6 now
  correctly returns (4, 6) instead of (4,)) - Update all stale model references in docstrings
  (gpt-4.1 → gpt-5.2, claude-sonnet-4-5-20250514 → claude-sonnet-4-6) - Fix README diagram (GPT-4.1
  → GPT-5.2, Gemini 2.5 Pro → Gemini 3.1 Pro) - Remove duplicate entries in README model table - Add
  test for Claude version sorting (4-6 > 4-5)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Continuous Integration

- Add release automation and CI workflows
  ([#3](https://github.com/OpenAdaptAI/openadapt-consilium/pull/3),
  [`2e5336b`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/2e5336b2af66213f453fad2d1079728ccac7faf6))

- Add test.yml workflow (lint + pytest on Python 3.10-3.12) - Add release.yml workflow
  (python-semantic-release + PyPI publish via OIDC) - Add semantic_release config to pyproject.toml
  - Add project.urls metadata - Bump version to 0.3.0 to match latest PyPI release

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- Add model auto-detection registry
  ([`218ccfa`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/218ccfa78dd180c83f63e4357e0ceea8c63f3e46))

- list_models() queries provider APIs for available models - get_latest() returns best model for a
  tier (flagship/fast/reasoning) - TTL-cached (1 hour) with hardcoded fallback defaults - Council
  auto-detects latest models when none specified

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Auto-discover latest models by filtering provider API responses
  ([#1](https://github.com/OpenAdaptAI/openadapt-consilium/pull/1),
  [`e001654`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/e0016542b038355db0e3a0f99b32b376602be6a7))

* feat: auto-discover latest models by filtering provider API responses

Add chat-model filtering to all three provider listers so the council automatically uses the latest
  models without hardcoded updates:

- OpenAI: allowlist regex (gpt-/o[1-9]) + denylist keywords (embedding, tts, audio, realtime,
  dall-e, whisper, moderation, search) - Anthropic: filter to claude-* models only - Google: require
  generateContent in supported_actions + gemini in name

CLI --models default is now resolved lazily via get_default_models() instead of a hardcoded list.
  Add pytest-timeout, live test marker, and 16 integration tests (auto-skipped without API keys).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

* fix: add image/transcribe to deny list, relax tier coverage test

Live testing revealed gpt-image-* and gpt-*-transcribe* models leaking through the OpenAI filter.
  Add "image" and "transcribe" to the deny keywords. Also relax test_all_classify_into_tiers to
  check that major tiers (flagship, fast, reasoning) are populated rather than requiring every model
  to classify — legacy and date-suffixed variants may not match tier patterns.

* fix: self-review — remove dead code, tighten filters, fix test fidelity

- Add codex/instruct to OpenAI deny keywords (not chat-completions API) - Delete unused
  tests/conftest.py (skip decorators were duplicated inline in test_model_discovery_live.py,
  conftest copies never imported) - Rewrite test_list_google_sdk_integration to actually call
  _list_google through SDK mocks instead of patching the function and reimplementing its logic
  inline - Clean up stale stream-of-consciousness comment - Add codex/instruct assertions to filter
  unit tests

* fix: correct misleading comment on Google mock test data

text-embedding-004 has embedContent (not generateContent), so it's filtered by both the
  supported_actions check AND the Gemini name check.

---------

Co-authored-by: Claude Opus 4.6 <noreply@anthropic.com>

- Initial consilium library
  ([`4b3159c`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/4b3159c64e807693fcd4fa2853b78b75c2893f4d))

Multi-LLM council for consensus-driven AI responses. Three-stage pipeline: query all models →
  cross-review → synthesize.

Includes: - Python API (Council class) and dict-based SDK (council_query) - CLI tool with --json,
  --budget, --no-review, --image options - OpenAI, Anthropic, and Google Gemini provider support -
  Real-time cost tracking with per-model breakdown - Budget enforcement (auto-skip review stages if
  over budget) - Parallel execution of model queries - 34 fully-mocked tests

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Initial consilium library - 3-stage LLM council pipeline
  ([`51941b3`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/51941b3f9ca819945d53019effae1e70c951f361))

Three-stage pipeline for querying multiple LLMs, cross-reviewing, and synthesizing: - Stage 1: Query
  all models in parallel (OpenAI, Anthropic, Google) - Stage 2: Each model anonymously reviews
  others' responses - Stage 3: Chairman synthesizes the best answer

Includes CLI, Python SDK, cost tracking, and budget enforcement.

Inspired by Karpathy's llm-council.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Update to latest models (March 2026)
  ([`2ad217a`](https://github.com/OpenAdaptAI/openadapt-consilium/commit/2ad217a791754e79818e5ffeda8c2a6f583dc32a))

- OpenAI: gpt-5.2 (flagship), gpt-5.2-pro, gpt-5, gpt-5-mini - Anthropic: claude-opus-4-6,
  claude-sonnet-4-6 (Feb 2026) - Google: gemini-3.1-pro, gemini-3-flash (March 2026) - Updated
  pricing table and model aliases - All 34 tests passing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
