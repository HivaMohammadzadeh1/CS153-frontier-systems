# Foundry UI Redesign — Design Spec

Date: 2026-05-29
Status: Approved, in implementation

## Goal

Turn the Foundry web app (`web/`) from a competent-but-generic chat clone into a
tutor students *want* to come back to. Optimize for all three of: motivation &
progress, visual identity & delight, and learning clarity. The learning loop
(ask → answer with routing → quiz → diagnostic → mastery update) already works;
this redesign makes that loop visible and rewarding without changing the backend
contract.

## Scope

Frontend only. The three files in `web/`: `index.html`, `styles.css`, `app.js`.
No build step is added (keeps the Tailwind-CDN, no-bundler setup). All existing
backend endpoints are reused as-is. One optional derived-summary endpoint may be
added only if client-side derivation proves messy — current decision is to derive
client-side and add nothing to the backend.

Out of scope: the Streamlit demo (`scripts/app.py`), backend logic, the routing
engine, auth.

## Information architecture

A persistent **left nav rail** switches between four client-side views (single
page, no reloads). The rail collapses to icons on narrow widths.

- 🏠 **Home / Today** — landing view
- 💬 **Chat** — the existing tutor chat; conversation history lives here
- 📊 **Progress** — full mastery breakdown
- 🎯 **Path** — the curriculum/topic map

`app.js` gains a tiny hash-based router (`#/home`, `#/chat`, …). Each view is a
render function. All existing chat, quiz, diagnostic, memory-modal, routing-modal,
and feedback code is preserved and reused; it simply lives inside the Chat view.

## Visual identity ("forge")

Driven by CSS custom properties so light/dark and theming are nearly free.

- Palette: ink base, **ember→violet** accent gradient (amber `#f59e0b` →
  violet `#7c5cff`) for the brand mark, primary actions, and progress fills.
- A small logo mark (spark/anvil glyph) replaces the bare "Foundry" wordmark.
- Inter for body; a heavier display weight for view headers. Rounded `2xl`
  cards with soft depth.
- Micro-interactions: count-up on mastery numbers, streak-flame pulse, card
  hover-lift, cross-fade between views. Honest motion only.
- **Default theme: light**, with a one-click dark toggle persisted in
  `localStorage`. Tokens make both first-class.

## Views

### Home / Today
All from real data:
- Greeting + 🔥 **streak** (consecutive active days, tracked in `localStorage`).
- **Overall mastery** (ring or headline %) + per-**area** breakdown bars
  (A/B/C…), computed from `/api/student/{id}/progress` joined with
  `/api/topics` area tags.
- **Up next →** highest-value next concept (lowest-mastery concept inside an
  in-progress topic). Opens Chat seeded on that topic.
- **Weak spot →** lowest-mastery topic or an active misconception, with a
  one-tap "Fix this" that opens Chat seeded on it.
- Primary CTA **Continue learning →**, plus quick actions **Quiz me** /
  **Diagnostic**.

### Chat
The existing chat experience, restyled to the new tokens: markdown, Mermaid,
citations, the "Context for this answer" routing modal, the 🧠 Memory modal,
quiz/diagnostic inline cards, thumbs feedback, regenerate/copy. Conversation
list and "+ New chat" live here.

### Progress
Per-topic mastery bars with confidence shading, concept lists, and the active
misconceptions panel — straight from `/api/student/{id}/progress`. This is the
existing memory-modal content promoted to a full view.

### Path
The `/api/topics` catalog rendered as a map grouped by area. Each topic node
shows **mastered / in-progress / not-started**, derived by matching the topic's
mastery in `/progress` against thresholds (mastered ≥ 0.7, in-progress > 0).
Clicking a node opens Chat focused on that topic.

## Data flow

```
/api/topics              -> topic catalog + area tags (Path, Home areas)
/api/student/{id}/progress -> per-topic avg_mastery + concepts + misconceptions
                              (Home, Progress, Path node states)
/api/student/{id}/conversations -> Chat history list
/api/chat, /api/quiz/*, /api/diagnostic/*, /api/feedback -> unchanged Chat loop
localStorage              -> streak, theme, sidebar/last-view, studentId
```

Derived client-side: overall mastery (confidence-weighted mean of topic
averages), area rollups, up-next, weak-spot, streak.

## Module boundaries in app.js

Reorganized (same file, no bundler) into clearly separated sections:
- `api` — fetch helpers per endpoint
- `router` — hash routing + view show/hide + nav-rail active state
- `views.home`, `views.progress`, `views.path` — render functions
- `chat` — the existing chat/quiz/diagnostic/memory/routing/feedback code,
  unchanged in behavior
- `theme`, `streak` — small persisted helpers

Each view renderer takes the cached progress data and returns/render into its
container; no view reaches into another's internals.

## Error handling

- Any failed fetch degrades gracefully: views render an empty/"no data yet"
  state rather than breaking (matches existing `loadProgress` behavior).
- New-student / no-mastery state is a first-class empty state on Home, Progress,
  and Path ("Start your first chat to see progress").
- Chat error handling is unchanged.

## Testing / verification

- Render each view (Home/Chat/Progress/Path) in headless Chromium via Playwright
  and screenshot; inspect each screenshot for layout correctness (a blank frame
  is a failure).
- Confirm the chat round-trip still works end-to-end against the live server
  (`POST /api/chat` from the UI returns and renders).
- Verify light/dark toggle and that no existing chat affordance regressed
  (routing modal, quiz, diagnostic, feedback, regenerate, copy, conversation
  switching).

## Non-goals / YAGNI

- No backend schema changes, no new framework, no bundler.
- No fake gamification (no XP inflation, confetti, or streaks not backed by real
  activity). Streak counts real active days only.
- No multi-user accounts; student identity stays the existing free-text field.
