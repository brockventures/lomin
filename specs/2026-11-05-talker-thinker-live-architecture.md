# Lomin Technical Specification: Talker + Thinker Live Architecture & Hybrid Pipeline Integration

- **Status:** Draft Proposal (Branch: `spec/talker-thinker-live-architecture`)
- **Author:** Zero
- **Date:** 2026-11-05 (PT)
- **Target Repository:** `mcarmody/lomin`

---

## 1. Executive Summary & User-Experience Goals

Lomin's existing LAMMEN pipeline (`POST /voice/turn`) excels at discrete push-to-talk (PTT) and utterance-bounded edge turns, with pluggable TTS chains (`chatterbox` -> `piper-desktop` -> `gemini` -> `kokoro` -> `piper`) and lightweight ADPCM framing for constrained devices like Gotchi. However, single-tier voice pipelines face two fundamental UX bottlenecks during interactive engineering and multi-step agent workflows:

1. **The Latency vs. Intelligence Dilemma:** Audio-native models (`gemini-3.8-live`) respond in **< 400 ms** with natural prosody, but degrade sharply when forced to execute complex multi-step code, shell, or repository tools directly. Conversely, frontier reasoning brains (`MVG_PLUGIN` / Claude Opus / `gemini-4-argon`) sit behind **4 to 15 seconds** of dead air (masked only by generic timed gap-fill clips at `1.2 s` / `3.5 s` / `8.0 s`) during which the user cannot converse, steer, or ask for status.
2. **Rigid Silence Cutoffs & ASR Loss:** Fixed silence-window VAD clips users off mid-sentence when they pause for 1.5–2 seconds to think, and speech-to-text (Whisper) routinely mangles repository names, CLI flags, and domain acronyms before the backing brain ever sees the prompt.

This specification introduces the **Talker + Thinker Architecture** into Lomin in two complementary ways:
- **Mode A (Full-Duplex Live Stream — `/voice/live`):** A persistent bidirectional WebSocket session fronted by `gemini-3.8-live` as a zero-tool **Talker** concierge that delegates substantive work via `NON_BLOCKING` async tool calls to Lomin's existing **Thinker** (`MVG_PLUGIN` / `streaming_brain.py`), while retaining the option to route output through Lomin's custom cloned TTS (`chatterbox`) or Gemini's 70-voice library (40 Daikon + 30 Constellation voices).
- **Mode B (Enhanced `/voice/turn` Hybrid Tricks):** Backporting raw 600ms pre-roll WAV splicing, live mid-turn tool progress injection, early multi-request resolution, and verbal confirmation gates into Lomin's existing HTTP `/voice/turn` and `/ask/stream` pipelines so constrained PTT clients (Gotchi) immediately benefit without requiring a continuous open-mic stream.

---

## 2. Voice Customization Matrix (Preset Catalog vs. Custom Voice Cloning)

Because different nodes and agents in Crab Cavern have distinct vocal identity requirements, the Talker + Thinker architecture supports two output audio modes configurable per node in `nodes.json` (`"voice_mode": "native" | "custom_tts"`):

### 2.1 Mode 1: Native `gemini-3.8-live` Audio (`response_modalities=["AUDIO"]`)
- **70-Voice Merged Catalog (`TTS_DAIKON_VOICES`):** Combines the **30 Constellation** voices (`Charon`, `Schedar`, `Orus`, `Aoede`, `Puck`, `Fenrir`, `Kore`) with the **40 Daikon** conversational voices (`Arlo`, `Cruz`, `Finn`, `Neno`, `Fola`, `Cleo`, `Gero`, `Lora`, `Olin`, `Tari`).
  - *Note on Voice Selection:* `Arlo` and `Cruz` provide clean American (Northwest) deep baritone voices; `Schedar` provides an even, composed lower-middle voice; `Neno` uses an American (Inland Southern) profile.
- **Persona & Cadence Conditioning:** While `gemini-3.8-live` does not accept arbitrary `.wav` zero-shot cloning weights directly on the WebSocket, its vocal pacing, warmth, dryness, and inflection are steered directly via the Talker's `system_instruction` and `SpeechConfig(language_code="en-US")`.
- **Latency:** **< 400 ms** time-to-first-audio (24 kHz 16-bit mono PCM).

### 2.2 Mode 2: Custom Cloned Voice Hybrid (`chatterbox` / `kokoro` / `piper-desktop` over Live Text Stream)
- For agents that require a custom zero-shot cloned voice (via Lomin's `desktop_tts.py` Chatterbox Turbo tunnel) or local Kokoro/Piper voices:
  - The `gemini-3.8-live` Talker session is opened with `response_modalities=["TEXT"]` (or text transcript extraction over the Live stream).
  - The Talker still processes raw 16 kHz user audio natively (preserving prosody turn-taking tokens `👂` and `❌` and `NON_BLOCKING` tool calls), but streams its short text tokens directly through Lomin's `SentenceSegmenter` -> `tts_backends.synthesize()`.
- **Trade-off:** Adds ~250–450 ms of local TTS synthesis latency, but preserves 100% custom voice cloning while keeping native audio understanding and prosodic turn-taking on the input side.

---

## 3. System Topology: The 4-Component Split

```
+-------------------------------------------------------------------------+
| 1. Edge / Desktop Client (Gotchi ESP32, Desktop Mic, Browser WebSocket) |
|    - Streams 16 kHz mono PCM (or ADPCM frames) up                       |
|    - Plays 24 kHz PCM / WAV out with instant queue flush on barge-in    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| 2. Lomin Voice Harness (`live_harness.py` / `gateway.py`)               |
|    - 600ms pre-roll PCM ring buffer + 3s barge-in splice (`useraudio`)  |
|    - Pacer & instant interruption flush (`pacer`)                       |
|    - Silent tool progress injector (`turn_complete=False`)              |
+-----------------+-------------------------------------+-----------------+
                  |                                     |
    (Persistent WebSocket)                  (Async Plugin / HTTP Stream)
                  v                                     v
+-----------------------------------+ +-----------------------------------+
| 3. Talker (`gemini-3.8-live`)     | | 4. Thinker (`MVG_PLUGIN` Brain)   |
|    - Sub-400ms audio front-end    | |    - Full Amos / Marvin / Zero /  |
|    - Prosody tokens: `👂` & `❌`   | |      Claude / Antigravity harness |
|    - Zero direct bash/repo tools  | |    - Receives transcript + raw    |
|    - Only 4 coordination tools    | |      16 kHz user WAV clips        |
|      (`behavior="NON_BLOCKING"`)  | |    - Streams `stage` / `status`   |
+-----------------------------------+ +-----------------------------------+
```

### 3.1 Talker Coordination Tool Contract (`behavior="NON_BLOCKING"`)
The Talker is stripped of all slow domain/file tools so its context window stays tiny and fast. It exposes only four coordination tools to the model:
1. `call_reasoning_thinker_agent(request: str, target: Optional[str] = None)`:
   - Dispatches a task or question asynchronously to the Thinker (`_plugin_brain_call_stream` or `streaming_brain.py`) routed via `nodes.json` (`target`).
   - **Same-Turn Spoken Bridge Rule:** The Talker's system instruction enforces speaking *one brief, context-specific sentence* (*"Let me check the build logs on Mirrormere"*, *"I'll pull up that PR diff"*) in the **exact same turn** it emits `call_reasoning_thinker_agent`, eliminating generic canned gap-fillers.
2. `cancel_thinker_agent(task_id: Optional[str] = None)`:
   - Triggered immediately when the user says *"stop"*, *"cancel that"*, or *"never mind"*, aborting the active Thinker stream in **< 50 ms**.
3. `answer_permission_or_question(decision: str, details: Optional[str] = None)`:
   - Resolves pending tool approval gates or clarifying questions surfaced by the Thinker verbally (with a hard guardrail blocking verbal approval of destructive shell patterns like `rm -rf` or force-pushes).
4. `set_progress_updates(enabled: bool)`:
   - Toggles whether the Talker proactively narrates major milestone transitions during long multi-minute builds.

---

## 4. The 6 Core Engineering Mechanisms to Port into Lomin

### Mechanism 1: Prosody-Driven Turn-Taking (`👂` & `❌`) Replacing Multi-Stage Ambient Gates
- **Problem:** In `/ambient` and `/voice/turn`, a 1.5–2 second pause while thinking triggers end-of-speech prematurely, and open-room filtering requires chaining Silero VAD -> Whisper -> pregate -> Laya -> Jev -> Haiku.
- **Lomin Integration:**
  - On `/voice/live`, `gemini-3.8-live` runs with proactive audio enabled and two silent control tokens in its system prompt:
    - **`👂` (Wait / Incomplete Thought):** If the user's vocal pitch stays flat/rising, a vowel is drawn out, or syntax is unfinished (*"Can you check the... uh..."*), the Talker outputs `👂` with zero audio and waits for the user to finish.
    - **`❌` (Ignore Ambient / Unaddressed Speech):** If the audio is background chatter, keyboard noise, or directed at someone else in the room, the Talker outputs `❌` and yields silently.
  - The harness strips `👂` and `❌` before audio/TTS output and logs a compact `live_gate` telemetry event to `logs/lomin.jsonl`.

### Mechanism 2: 600ms Pre-Roll Ring Buffer & Raw User Audio WAV Splicing (`user_audio_buffer.py`)
- **Problem:** Whisper routinely turns technical symbols (`MVG_PLUGIN`, `adpcm_codec.py`, `KSH5`) into phonetic soup (*"M V G plug in"*, *"cash 5"*).
- **Lomin Integration:**
  - Maintain a rolling 16 kHz PCM ring buffer per active session with a **600 ms pre-roll lead-in (`sos_lead_in_ms = 600`)** prior to VAD speech start, plus a **3.0 s barge-in splice window (`resume_within_ms = 3000`)** that concatenates interrupted utterances when a user barges in to correct themselves.
  - When dispatching to the Thinker (`brain_call_stream` / `streaming_brain.py`), package the normalized 16 kHz WAV clip(s) alongside the text transcript (`audio_b64` / `voice_clips`) so multimodal Thinkers hear the exact user pronunciation alongside the transcript.

### Mechanism 3: Silent Mid-Turn Tool Progress Injection (`turn_complete=False`)
- **Problem:** While the brain is running tools in `/voice/turn`, the voice pipeline is blind until the final text response arrives.
- **Lomin Integration:**
  - Lomin's `streaming_brain.py` and `_plugin_brain_call_stream` already emit `status` / `stages_out` callbacks as the brain executes tools.
  - In `live_harness.py`, each `status` event from the Thinker is formatted as a compact 1-line update (`[task progress] Running pytest tests/test_gateway.py`) and injected into the Live Talker session via `session.send_client_content(turns=[...], turn_complete=False)`.
  - Because `turn_complete=False` appends context silently without triggering speech, the Talker remains quiet while the Thinker works—until the user asks *"what step is it on?"*, at which point the Talker answers immediately from its live context window without interrupting the Thinker.

### Mechanism 4: Mid-Turn Steering & Early Partial-Handoff Resolution
- **Problem:** Users frequently think of a constraint two seconds after issuing a command (*"Oh, and only run the unit tests, not integration"*).
- **Lomin Integration:**
  - Each `call_reasoning_thinker_agent` invocation receives a monotonically increasing `handoff_id`.
  - If the user speaks a follow-up while `handoff_id=1` is still running, the Talker acknowledges it in < 400 ms and emits `call_reasoning_thinker_agent` (`handoff_id=2`), which Lomin injects into the active Thinker session as a mid-turn steering update (or queues behind `handoff_id=1`).
  - When any handoff resolves (`Early: true` for completed sub-questions), the harness returns a `FunctionResponse` matched to that `handoff_id` so the Talker speaks the answer immediately while remaining tasks continue.

### Mechanism 5: Out-of-Band Verbal Approvals with Destructive Guardrails
- **Problem:** When a CLI or background agent hits a confirmation gate (`[WAITING]` / Decision Card), voice users are forced to look at a screen or Discord channel to click Approve.
- **Lomin Integration:**
  - Extend `streaming_brain.py` / `pending_and_auth.py` with a `waiting_approval` event schema (`{prompt_id, summary, command, is_destructive}`).
  - Non-destructive prompts are injected into the Talker (`[waiting on the user] Approve running migration check?`), allowing the user to approve or reject verbally via `answer_permission_or_question`.
  - **Hard Safety Invariant:** Commands matching destructive patterns (`rm -rf`, `git push --force`, `DROP TABLE`, system reboot/container stop) set `is_destructive=True` and are blocked from verbal approval at the harness layer, instructing the Talker to require explicit screen/button confirmation.

### Mechanism 6: Output Pacer & Sub-20ms Barge-In Queue Flush
- **Problem:** Live models generate 24 kHz audio 3–5x faster than real-time playback; without frame-level pacing and instant buffer truncation, barging in leaves 1–2 seconds of stale audio playing over the user.
- **Lomin Integration:**
  - Implement a 20 ms frame pacer in `live_harness.py` that tracks exactly how many milliseconds of audio have been delivered to the client and immediately flushes unplayed frames (`{"event": "clear_audio_buffer", "interrupted": true}`) the instant `server_content.interrupted` or client barge-in VAD triggers.

---

## 5. Wire Protocol & Phased Implementation Plan on `lomin`

To keep `gateway.py` clean and preserve 100% backwards compatibility with existing `/voice/turn`, `/ambient`, and `/stt` callers, the implementation is split into three small, independently testable phases:

### Phase 1: Pre-Roll Audio Splicing & Contextual Ack in `/voice/turn` (`user_audio_buffer.py`)
- Add `user_audio_buffer.py` with 600 ms pre-roll ring buffering and 3.0 s barge-in splice stitching.
- Forward stitched raw 16 kHz WAV (`audio_b64`) alongside the Whisper transcript into `streaming_brain.py` and `_plugin_brain_call_stream` by default when `MVG_SPLICE_RAW_AUDIO=1`.
- Replace static timed vocal openers in `_maybe_gap_fill` with an optional ultra-fast contextual opener or immediate Thinker `status` stage readout.

### Phase 2: Modular Talker + Thinker Live Harness (`live_harness.py`)
- Add `live_harness.py` implementing the `TalkerThinkerSession` class over `google-genai` (`client.aio.live.connect` on `gemini-3.8-live`, with stdlib/WebSocket fallback for environments without the SDK).
- Wire the 4 `NON_BLOCKING` coordination tools (`call_reasoning_thinker_agent`, `cancel_thinker_agent`, `answer_permission_or_question`, `set_progress_updates`) directly to Lomin's `_plugin_brain_call_stream` and `streaming_brain.py`.
- Wire silent `turn_complete=False` progress injection from `stages_out` / `on_partial` callbacks and support both `"voice_mode": "native"` (70-voice Daikon/Constellation catalog) and `"voice_mode": "custom_tts"` (routing Talker text deltas through `SentenceSegmenter` -> `tts_backends.synthesize()`).

### Phase 3: Gateway Endpoint, Node Profile Config & Test Suite (`gateway.py`, `tests/test_live_harness.py`)
- Expose `GET/POST /voice/live` (WebSocket upgrade + bidirectional JSON/PCM framing) and per-node config keys in `nodes.json` (`live_model`, `voice_mode`, `live_voice`, `splice_raw_audio`, `progress_updates`).
- Add deterministic unit and mock-stream tests in `tests/test_live_harness.py` covering `👂`/`❌` prosody filtering, `NON_BLOCKING` handoffs, `turn_complete=False` progress injection, destructive approval blocking, and barge-in queue flushing.
