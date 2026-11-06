# Lomin Spec: Talker + Thinker Live Architecture

- **Status:** Draft, revised after review of PR #12 (2026-11-05)
- **Target repository:** `mcarmody/lomin`
- **Rule:** nothing household-specific lives in this repo (`AGENTS.md`, "The rule"). Examples use `nodes.example.json` only.

---

## 1. Summary and goals

Lomin's pipeline (`POST /voice/turn`) is built for discrete, utterance-bounded turns. When the brain is a full agent, replies take 4-5 s to the first sentence (README, "Gap-fill and latency metrics"), and the user hears canned gap-fill clips. A generated opener from a text model was tried before and dropped as too slow, so gap-fill stays canned. The user cannot steer, cancel or ask for status during that time.

This spec adds an **orchestrator** pattern. The speaking agent does no work of its own. It speaks one short, specific sentence, delegates to a **Thinker** (a subagent or the existing brain behind `MVG_PLUGIN`), narrates progress, and takes cancel and follow-ups. The value is in that discipline, and it does not depend on the model. A native live model is the fastest way to run it, not the only way.

There are two modes. A node uses one of them.

- **Mode 1, native live orchestrator.** A full-duplex audio model (Gemini Live first) is the orchestrator. It is fast and has native turn-taking. It speaks in its own voice, so there is **no custom voice**. Needs `/voice/live` (PR C).
- **Mode 2, prompted orchestrator on the current pipeline.** Whisper, the classifier, the brain and the TTS chain are unchanged, so custom voices work. The brain is prompted to be an orchestrator: speak one sentence, then delegate to a subagent, choosing the subagent's model and effort for the task. Lomin injects that instruction (section 6.7); the brain does the rest. Needs no new transport and no new dependency (PRs A and B).

Goals, in order:

- **Cheap wins first.** Mode 2, narrated progress and spoken cancel land in `/voice/turn`.
- **Custom voices keep working** wherever Mode 1 is not used.
- **Measure before claiming.** Every speed claim in this spec is a hypothesis until it is measured against the `voice-latency.jsonl` baseline (section 8).
- **Stdlib core.** The default install stays stdlib-only. Mode 1 is a guarded optional dependency, as `onnxruntime` already is.
- **Off by default.** Cloud audio, metered keys, the orchestrator prompt and verbal approvals are all opt-in per node.

## 2. Mode 1: Talker backend

The harness owns the coordination tools (section 4). A backend only turns audio into a stream of events. Mode 1 needs a native, audio-out model; text-out classes (a cascaded text model, or audio-in with whole-response text) are dropped because they repeat the slow generated-opener design.

```python
class TalkerBackend:
    capabilities: set[str]        # audio_in, audio_out, full_duplex, tool_calls, silent_context
    async def open(self, config: dict) -> None: ...
    async def send_audio(self, pcm16_16k: bytes) -> None: ...
    async def send_text(self, text: str, *, silent: bool = False) -> None: ...
    async def send_tool_result(self, call_id: str, result: dict) -> None: ...
    def events(self) -> AsyncIterator[dict]: ...   # audio, text, tool_call, interrupted, turn_end
    async def cancel(self) -> None: ...
    async def close(self) -> None: ...
```

A backend without `silent_context` gets progress updates only on request.

### 2.1 The voice trade-off

A native Talker speaks in its own voice. A node therefore chooses between the Talker's speed and prosody, or a custom voice, not both in one reply. A possible hybrid is for the Talker to speak openers and progress lines natively and for the custom-voice TTS to speak the Thinker's final answer. Mixed voices in one reply may sound wrong. This is a product decision (section 10), not a design default.

### 2.2 Candidate backends (facts as of 2026-11-05, to be re-checked before building)

Sourced from Google's docs, read 2026-11-05:

- **Gemini Live, `gemini-3.8-live`:** listed as stable, latest update September 2026 ([model page](https://aistudio.google.com/docs/models/gemini-3.8-live)). That page states "Audio is the only supported response modality (`response_modalities=["AUDIO"]`)", with `output_audio_transcription` for a text transcript. A text-only Live session is **not** available.
- **Voices:** 30 prebuilt voices plus an "Extended Voice Library" of "hundreds", not listed on the page ([speech generation](https://aistudio.google.com/docs/speech-generation)). The earlier draft's "70 voices" and the "Daikon" and "Constellation" family names were not found there. The implementation takes the voice name as an opaque string and lists nothing.
- **Latency:** no first-audio figure is published on the pages read. No latency number appears in this spec until it is measured (section 8).
- **Audio in, text out (`gemini-3.8-flash`, Interactions API):** request and response, not a stream; inline limit 20 MB ([audio docs](https://aistudio.google.com/docs/audio)). Not a Mode 1 candidate. Whether it could gate an open stream is unconfirmed and out of scope for now.

From secondary sources, provisional (confirm on official docs): OpenAI `gpt-realtime` family, Amazon Nova Sonic, xAI Grok Voice, NVIDIA NemotronLabs VoiceChat (open weights, native tool calling, weak argument accuracy in NVIDIA's own benchmark), NVIDIA PersonaPlex-7B (open weights, tool calling only in community forks).

## 3. System topology

```
Client (watch, desktop mic, browser)
   |  16 kHz PCM or ADPCM up; 24 kHz PCM / WAV down; flush on barge-in
   v
Lomin harness (gateway.py, optional live_harness.py)
   |  VAD, session state, coordination tools, progress relay
   +---- TalkerBackend (configurable)
   +---- Thinker: MVG_PLUGIN brain_call_stream (existing hook)
```

The Thinker is whatever the plugin routes to. This repo names no agent.

## 4. Coordination tools (harness-owned)

Four tools, all with simple arguments, because tool-call argument accuracy varies by backend:

1. `call_thinker(request: str)`: start a Thinker task. The Talker speaks one short specific sentence in the same turn ("let me check the build logs"), replacing generic gap-fill.
2. `cancel_thinker()`: abort the active task.
3. `answer_prompt(prompt_id: str, decision: "yes" | "no")`: answer a pending approval, under section 6.5.
4. `set_progress_updates(enabled: bool)`.

Today the brain hook is a one-shot call with a timeout. `cancel_thinker()` needs a brain-side abort, defined in section 5.

## 5. Required hook extensions (PR A)

All optional; a plugin that lacks one behaves as it does today.

- `brain_call_stream(..., audio_clips=None)`: raw 16 kHz WAV clips alongside the transcript. The gateway passes it only if the plugin's signature accepts it.
- `brain_cancel(task_id)`: abort a running brain call. Without it, cancel stops speech and drops the result but cannot stop the Thinker, and the Talker says so.
- `stages_out` events get a stable schema: `{"stage": str, "detail": str, "t": float}`.
- Approval prompts from the Thinker: `{"prompt_id", "summary", "command", "risk"}`. Defined here; wired in section 6.5.

Each hook gets a test that patches the hook, per `AGENTS.md`.

## 6. Mechanisms

### 6.1 Turn-taking and ambient gating

- Default: unchanged. Open-room gating (Silero, Whisper, pregate, classifier) stays local.
- Full-duplex backends may use control tokens (`👂` wait, `❌` ignore) to replace the Lomin VAD for that node. This sends continuous room audio to the provider, so it is opt-in per node (`live_gate: "backend"`), documented in the README privacy section, and keeps speaker ID and echo filtering active on the live path.
- The harness strips control tokens before audio or TTS and logs a compact `live_gate` event.

### 6.2 Pre-roll and barge-in splice

- `/voice/turn` receives a finished upload, so the server has no audio before speech start. Pre-roll is a **client capture contract** (recommended lead-in 600 ms, to be tuned) and is documented, not implemented server-side.
- On `/voice/live`, the harness keeps a rolling PCM ring buffer, and may splice an utterance that the user restarts within a short window (default 3 s, configurable).
- With `audio_clips` available, the Thinker receives the user's audio alongside the transcript.

### 6.3 Progress and status

- The harness maps `stages_out` events to one-line updates.
- Backends with `silent_context` receive them silently and answer "what step is it on?" from context.
- Other backends get the same lines only when the user asks, or at milestones if `set_progress_updates(true)`, rate-limited so the Talker is not talking over itself.
- Works in `/voice/turn` too: a status readout replaces the timed gap-fill clip when the stage text is available.

### 6.4 Steering

- v1 queues a follow-up behind the running task (`handoff_id` increments). Mid-turn injection into a running Thinker needs a brain API that does not exist; it is out of scope until a plugin supplies one.
- When a task resolves, the harness returns the result matched to its `handoff_id`.

### 6.5 Verbal approvals (default off)

- A new subsystem, not an extension of `pending_and_auth.py` (which handles pending messages and device auth).
- **Allowlist, not denylist.** Only prompts the Thinker marks `risk: "low"` may be approved by voice. Any other prompt, or a prompt with no `risk` field, requires an out-of-band confirmation.
- Each approval binds to a `prompt_id`, and the Talker reads back the summary before the decision counts.
- Requires speaker verification for the approving utterance, reusing what exists in `gateway.py` (`identify_speaker`, `_match_speaker`, `enrolled_roster`; embeddings in `desktop_embed.py`). No new verification subsystem. Ignores audio flagged as ambient or echo.
- A pattern check for destructive commands is a second layer only, never the gate.

### 6.6 Output pacing and barge-in flush

- The harness paces output in fixed frames and tracks how much audio the client has been sent.
- On interruption the server sends `{"event": "clear_audio_buffer", "interrupted": true}` and stops queued synthesis.
- The time until playback actually stops depends on the client buffer and the network. The spec states a **client contract** (flush on the event) and makes no server latency guarantee.

### 6.7 Orchestrator prompt (Mode 2)

`frame_voice_prompt` already wraps every brain call. When `MVG_ORCHESTRATOR_PROMPT=1`, it appends a short, generic instruction block:

- You are the voice orchestrator. Speak one specific sentence first, then delegate.
- Do not do the work yourself. Hand it to a subagent and choose that subagent's model and effort for the task.
- If a follow-up arrives while a task runs, say so. Do not start a second task silently.
- Progress lines come from the harness. Do not invent progress.

Rules:

- The text names no tool, agent or household. The plugin adds the site-specific half (tool names, model choices) through `MVG_ORCHESTRATOR_TEXT` or a plugin hook.
- Off by default, switchable per target.
- A prompt is a request, not a guarantee. Whether it works is decided by the experiment in section 8, and the result goes in the PR.
- Mid-task steering of a running subagent is brain-side and **out of scope for Lomin**. Lomin offers cancel (`brain_cancel`) and queue-behind only.

## 7. Wire protocol, config and fallback

- Mode 1 only. `GET /voice/live`: WebSocket upgrade, JSON control frames and binary PCM. The WebSocket layer is a guarded optional dependency; the default install has no `/voice/live`.
- Per-node keys in `nodes.example.json`: `orchestrator_prompt` (Mode 2), `talker_backend`, `live_voice`, `progress_updates`, `live_gate`, `splice_raw_audio`.
- Auth: the existing per-node bearer tokens (`MVG_DEVICE_TOKENS`) apply to `/voice/live`.
- Fallback: if the live session or Talker backend fails, the node answers through `/voice/turn`, and the failure is logged.
- Cost and keys: backends that call a metered API take their key from an env var only. They are off unless a node enables them. The README states that these calls are billed per use.

## 8. Phased plan and first experiment

Prerequisite: record a baseline of response times (`brain_first_text_ms`, first audio, total) from `voice-latency.jsonl`, so every later claim is measured. Do not publish latency targets before that.

**Experiment 1 (first, no new transport).** Turn records carry an `orchestrator` field. Run the same node and brain with `MVG_ORCHESTRATOR_PROMPT` off, then on, over the same set of requests. Compare `brain_first_text_ms` and `first_audio_ms`. If the orchestrator prompt does not move `brain_first_text_ms`, Mode 2 only helps through narration, and PR C becomes the real speed work.

- **PR A, hooks:** `audio_clips`, `brain_cancel(task_id)`, the `stages_out` schema, with tests. `task_id` is minted by the gateway per brain call and is the same id as `handoff_id`. The gateway passes `audio_clips` or `task_id` only if the plugin signature accepts it. No new endpoint.
- **PR B, Mode 2 and `/voice/turn`:** the orchestrator prompt (6.7), `orchestrator` in the latency record, narration of `stages_out`, spoken cancel with `brain_cancel`, client pre-roll contract in the docs. No new transport or dependency.
- **PR C, Mode 1:** `TalkerBackend`, `/voice/live`, a WebSocket implementation as a guarded optional dependency (optional `websockets` package on its own event loop thread, not a hand-rolled stdlib handshake), and Gemini Live as the first backend. Tests use a mock backend. Decide after Experiment 1 and the baseline.
- **Later:** verbal approvals (6.5), as a separate spec and PR once speaker verification is in place.

## 9. Tests

- Frame tests: the orchestrator block is absent by default, present when enabled, and contains no household names.
- Mock `TalkerBackend` driving (PR C): control-token filtering, handoff matching, silent progress, cancel with and without `brain_cancel`, fallback to `/voice/turn`. On fallback, an in-flight Thinker task is not dropped: its result is spoken on `/voice/turn` and the client receives a protocol event telling it to switch.
- Hook tests patch the hook, per `AGENTS.md`.
- The household-name grep from the PR checklist runs on every PR.
- No test needs the network or a key.

## 10. Open questions

Resolved (maintainer, via review of PR #12): the WebSocket layer is the optional `websockets` dependency, guarded like `onnxruntime`; approval speaker verification reuses the existing code in `gateway.py`.

- **Product:** is a hybrid acceptable (native Talker for openers and progress, custom voice for the final answer), or must a node pick one voice?
- Can an audio-in model stream both input and output? Out of scope for now.
- Splice rule for `/voice/live` (6.2): what counts as a restart, and does the Thinker receive one concatenated clip or two? Pin down before PR C.
