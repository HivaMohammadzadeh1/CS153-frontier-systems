# Streaming Tutor Responses — Design Spec

Date: 2026-06-01
Status: Implemented

## Goal
Stream the tutor's reply token-by-token instead of waiting for the full response,
without losing any of the existing per-turn behavior (context selection, citations,
trace capture, references, mastery bump).

## Approach
- **Shared prep/finalize**: extracted `_prepare_turn()` (conversation, candidate
  pool, context selection, profile, built prompt) and `_finalize_turn()` (trace
  capture, episodic log, auto-title, mastery bump, citation substitution → response
  fields) from the chat endpoint. `/api/chat` now = prep → `llm.complete` → finalize;
  no logic is duplicated.
- **`TutorAgent.build_prompt()`**: extracted so the endpoint and `answer()` build the
  identical system+context prompt for a given selection.
- **`POST /api/chat/stream`** (SSE, `text/event-stream`): runs the same prep up front
  (so the "Context for this answer" data is ready), then streams `llm.stream()` deltas
  as `data:{"delta":"…"}`. When the model finishes it runs `_finalize_turn` and emits
  `data:{"done":true, conversation_id, reply, references, selected, dropped, budget,
  tokens_used, router}`. Errors emit `data:{"error":"…"}`. `/api/chat` stays as the
  non-streaming fallback. Auth gate + session `student_id` override apply as for all
  write routes.
- **Frontend**: `sendMessage` reads the SSE stream via `fetch` + `ReadableStream`
  (EventSource is GET-only). Deltas fill a live `.prose-chat.streaming` bubble (raw
  text + blinking caret); on `done` the bubble is replaced by the fully-rendered
  message (markdown, `[n]` citations, mermaid, footer actions) via the existing
  `appendAssistantMessage` path.

## Why SSE + fetch (not EventSource)
The chat needs a POST body; EventSource only does GET. `fetch` + a stream reader
gives POST + incremental reads.

## Testing
Unit: `build_prompt` (context + profile rendering, empty-profile case). Live: SSE
deltas + `done` frame verified via curl; frontend verified with Playwright
(mid-stream caret → finalized rendered message with citations). Full suite green.
