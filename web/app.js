/**
 * Foundry — ML Systems Tutor
 * Claude.ai-style single-column chat frontend.
 */

// ── State ──────────────────────────────────────────────────────
const state = {
  studentId: "you",
  budget: 3000,
  topicId: null,
  messages: [],      // {role, content, reply?, references?, selected?, dropped?, tokens_used?}
  reuseCounts: {},
};

// ── Helpers ────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "")
  .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
  .replace(/"/g,"&quot;").replace(/'/g,"&#39;");

function scrollBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

// ── Mermaid + Marked ───────────────────────────────────────────
mermaid.initialize({ startOnLoad: false, theme: "neutral",
  fontFamily: "Inter, system-ui, sans-serif" });

marked.setOptions({ breaks: true, gfm: true });

function renderMarkdown(text) {
  const blocks = [];
  const ph = (i) => `MERMAID_PLACEHOLDER_${i}`;
  const withPh = text.replace(/```mermaid\s*([\s\S]*?)```/g, (_, code) => {
    blocks.push(code.trim());
    return `\n${ph(blocks.length - 1)}\n`;
  });
  let html = DOMPurify.sanitize(marked.parse(withPh));
  blocks.forEach((code, i) => {
    const safe = DOMPurify.sanitize(code, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] });
    html = html.replace(
      new RegExp(`<p>${ph(i)}</p>|${ph(i)}`, "g"),
      `<pre class="mermaid">${safe}</pre>`
    );
  });
  return html;
}

async function runMermaid(el) {
  const nodes = [...el.querySelectorAll("pre.mermaid")];
  if (!nodes.length) return;
  try { await mermaid.run({ nodes }); } catch (e) {
    nodes.forEach((n) => { n.style.color = "#ef4444"; n.textContent = "[diagram error] " + e.message; });
  }
}

// ── Citation substitution ──────────────────────────────────────
function substituteCitations(text, references) {
  // references is array of {id, n, title, ...}
  if (!references || !references.length) return text;
  let out = text;
  references.forEach((r) => {
    out = out.replaceAll(`[${r.id}]`, `[${r.n}]`);
  });
  return out;
}

// ── Starter cards ──────────────────────────────────────────────
const STARTERS = [
  { icon: "🧠", title: "How does the KV cache work?", sub: "Foundations of inference" },
  { icon: "⚡", title: "Explain PagedAttention", sub: "vLLM internals" },
  { icon: "🔧", title: "Train a two-tower recommender", sub: "Build a recsys" },
  { icon: "📊", title: "What's the math behind RoPE?", sub: "Long context" },
  { icon: "🎯", title: "Compare DDP, FSDP, and tensor parallelism", sub: "Distributed training" },
  { icon: "🚀", title: "Design an ad ranking system", sub: "ML system design" },
];

function buildStarterGrid() {
  const grid = $("starterGrid");
  STARTERS.forEach(({ icon, title, sub }) => {
    const btn = document.createElement("button");
    btn.className = [
      "starter-card text-left bg-white border border-gray-200 rounded-2xl p-4",
      "hover:border-indigo-400 hover:bg-indigo-50/30 transition-colors",
      "shadow-[0_1px_2px_rgba(17,24,39,0.05)]",
    ].join(" ");
    btn.innerHTML = `<div class="text-xl mb-2">${icon}</div>
      <div class="text-sm font-medium text-gray-900 leading-snug">${esc(title)}</div>
      <div class="text-xs text-gray-500 mt-0.5">${esc(sub)}</div>`;
    btn.addEventListener("click", () => sendMessage(title));
    grid.appendChild(btn);
  });
}

// ── Topics ─────────────────────────────────────────────────────
async function loadTopics() {
  try {
    const r = await fetch("/api/topics");
    if (!r.ok) return;
    const topics = await r.json();
    const sel = $("topicSelect");
    topics.forEach((t) => {
      const o = document.createElement("option");
      o.value = t.id; o.textContent = t.title;
      sel.appendChild(o);
    });
  } catch (_) {}
}

// ── Message rendering ──────────────────────────────────────────
function thinkingEl() {
  const d = document.createElement("div");
  d.id = "thinking";
  d.className = "msg-row py-4";
  d.innerHTML = `<div class="thinking-dots flex items-center gap-1 h-5 pl-1">
    <span></span><span></span><span></span>
  </div>`;
  return d;
}

function appendUserBubble(text) {
  const row = document.createElement("div");
  row.className = "msg-row flex justify-end py-2";
  row.innerHTML = `<div class="max-w-[80%] bg-gray-100 rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap text-gray-900">${esc(text)}</div>`;
  $("chat").appendChild(row);
}

function appendAssistantMessage(msg, msgIdx) {
  const row = document.createElement("div");
  row.className = "msg-row group py-4 border-b border-gray-100";
  row.dataset.msgIdx = msgIdx;

  // Substitute citations then render markdown
  const citedText = substituteCitations(msg.content, msg.references);
  const bodyHtml = renderMarkdown(citedText);

  // Build references block
  let refsHtml = "";
  if (msg.references && msg.references.length) {
    const items = msg.references.map((r) =>
      `<div class="text-xs text-gray-500"><span class="font-medium text-gray-700">[${r.n}]</span> ${esc(r.title)}</div>`
    ).join("");
    refsHtml = `<details class="refs mt-3">
      <summary>Sources ▾</summary>
      <div class="mt-1 space-y-0.5 pl-1">${items}</div>
    </details>`;
  }

  row.innerHTML = `
    <div class="prose-chat text-sm text-gray-800 leading-relaxed">${bodyHtml}</div>
    ${refsHtml}
    <div class="msg-footer mt-2 flex items-center gap-3 text-xs text-gray-400">
      <button class="btn-test hover:text-indigo-600 transition-colors" title="Test yourself">🎯 Test yourself</button>
      <button class="btn-ctx hover:text-indigo-600 transition-colors" title="View context">🔍 View context</button>
      <button class="btn-regen hover:text-indigo-600 transition-colors" title="Regenerate">🔁 Regenerate</button>
      <button class="btn-copy hover:text-indigo-600 transition-colors" title="Copy">📋 Copy</button>
    </div>
    <div class="quiz-area mt-2"></div>`;

  // Wire footer buttons
  row.querySelector(".btn-test").addEventListener("click", () => triggerQuiz(row, msgIdx));
  row.querySelector(".btn-ctx").addEventListener("click", () => openRoutingModal(msg));
  row.querySelector(".btn-regen").addEventListener("click", () => regenerate(msgIdx));
  row.querySelector(".btn-copy").addEventListener("click", () => copyReply(msg.content));

  $("chat").appendChild(row);
  runMermaid(row);
  return row;
}

// ── Thinking indicator ─────────────────────────────────────────
function showThinking() {
  const el = thinkingEl();
  $("chat").appendChild(el);
  scrollBottom();
  return el;
}

// ── Send message ───────────────────────────────────────────────
async function sendMessage(text) {
  text = (text ?? $("msgInput").value).trim();
  if (!text) return;

  // Hide welcome, show chat
  $("welcome").classList.add("hidden");

  // Sync settings
  state.studentId = $("studentIdInput").value.trim() || "you";
  state.topicId = $("topicSelect").value || null;
  state.budget = parseInt($("budgetSlider").value, 10) || 3000;

  // Append user bubble
  state.messages.push({ role: "user", content: text });
  appendUserBubble(text);

  // Clear input
  $("msgInput").value = "";
  $("msgInput").style.height = "auto";
  $("sendBtn").disabled = true;

  // Thinking indicator
  const thinking = showThinking();
  setInputLocked(true);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        student_id: state.studentId,
        topic_id: state.topicId,
        question: text,
        budget: state.budget,
        reuse_counts: state.reuseCounts,
      }),
    });
    thinking.remove();

    if (!res.ok) {
      appendError(await res.text());
      return;
    }
    const data = await res.json();

    data.selected?.forEach((it) => {
      state.reuseCounts[it.id] = (state.reuseCounts[it.id] || 0) + 1;
    });

    const msgIdx = state.messages.length;
    const aMsg = {
      role: "assistant",
      content: data.reply,
      references: data.references,
      selected: data.selected,
      dropped: data.dropped,
      tokens_used: data.tokens_used,
    };
    state.messages.push(aMsg);
    appendAssistantMessage(aMsg, msgIdx);
    scrollBottom();
  } catch (err) {
    thinking.remove();
    appendError(err.message);
  } finally {
    setInputLocked(false);
    $("msgInput").focus();
  }
}

function setInputLocked(locked) {
  $("msgInput").disabled = locked;
}

function appendError(text) {
  const d = document.createElement("div");
  d.className = "my-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700";
  d.textContent = "Error: " + text;
  $("chat").appendChild(d);
  scrollBottom();
}

// ── Regenerate ─────────────────────────────────────────────────
async function regenerate(msgIdx) {
  // Find the user message that preceded this assistant turn
  let userText = null;
  for (let i = msgIdx - 1; i >= 0; i--) {
    if (state.messages[i].role === "user") { userText = state.messages[i].content; break; }
  }
  if (!userText) return;
  // Remove the current assistant message element from DOM
  const row = $("chat").querySelector(`[data-msg-idx="${msgIdx}"]`);
  if (row) row.remove();
  state.messages.splice(msgIdx, 1);
  await sendMessage(userText);
}

// ── Copy ───────────────────────────────────────────────────────
async function copyReply(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    prompt("Copy:", text);
  }
}

// ── Routing modal ──────────────────────────────────────────────
function openRoutingModal(msg) {
  const body = $("routingBody");
  const sel = msg.selected || [];
  const dropped = msg.dropped || [];
  const tokensUsed = msg.tokens_used ?? 0;
  const budget = state.budget;

  const pct = Math.min(100, Math.round((tokensUsed / budget) * 100));
  let html = `<div class="mb-4">
    <div class="flex justify-between text-xs text-gray-500 mb-1">
      <span>Token budget</span><span>${tokensUsed} / ${budget}</span>
    </div>
    <div class="h-2 bg-gray-100 rounded-full overflow-hidden">
      <div class="h-full bg-indigo-500 rounded-full" style="width:${pct}%"></div>
    </div>
  </div>`;

  if (sel.length) {
    html += `<p class="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">Selected (${sel.length})</p>`;
    sel.forEach((it) => { html += routingItemHtml(it, "indigo"); });
  }

  if (dropped.length) {
    html += `<details class="mt-4">
      <summary class="cursor-pointer text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">Dropped top (${dropped.length}) ▾</summary>
      <div class="mt-2 space-y-2">${dropped.slice(0,5).map((it) => routingItemHtml(it, "gray")).join("")}</div>
    </details>`;
  }

  body.innerHTML = html;
  const modal = $("routingModal");
  modal.classList.remove("hidden");
  modal.classList.add("flex");
}

function routingItemHtml(it, color) {
  const scores = [
    ["relevance", it.score_relevance],
    ["recency", it.score_recency],
    ["misc.", it.score_misconception],
    ["prereq", it.score_prerequisite],
    ["reuse", it.score_reuse],
    ["total", it.score_total],
  ].filter(([, v]) => v != null);

  const bars = scores.map(([label, val]) => {
    const p = Math.round((val || 0) * 100);
    return `<div class="flex items-center gap-2 text-xs">
      <span class="w-14 text-gray-400 flex-shrink-0">${label}</span>
      <div class="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <div class="h-full bg-${color}-400 rounded-full" style="width:${p}%"></div>
      </div>
      <span class="w-8 text-right text-gray-500">${p}%</span>
    </div>`;
  }).join("");

  return `<div class="rounded-xl border border-gray-100 bg-gray-50 p-3 mb-2">
    <div class="text-sm font-medium text-gray-800 mb-2 leading-snug">${esc(it.title)}</div>
    ${bars}
    ${it.body ? `<p class="mt-2 text-xs text-gray-500 line-clamp-2">${esc(it.body.slice(0,200))}</p>` : ""}
  </div>`;
}

function closeRoutingModal() {
  const modal = $("routingModal");
  modal.classList.add("hidden");
  modal.classList.remove("flex");
}

// ── Quiz ───────────────────────────────────────────────────────
async function triggerQuiz(row, msgIdx) {
  const topicId = state.topicId || $("topicSelect").value || null;
  if (!topicId) { alert("Select a topic first to generate a quiz."); return; }

  const area = row.querySelector(".quiz-area");
  area.innerHTML = `<div class="quiz-card border border-indigo-200"><div class="flex items-center gap-2 text-xs text-gray-500"><div class="thinking-dots"><span></span><span></span><span></span></div> Generating question…</div></div>`;

  try {
    const res = await fetch("/api/quiz/generate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic_id: topicId }),
    });
    if (!res.ok) throw new Error(await res.text());
    const { question, rubric } = await res.json();
    renderQuizQuestion(area, msgIdx, question, rubric, topicId);
  } catch (err) {
    area.innerHTML = `<p class="text-xs text-red-500 mt-1">Quiz error: ${esc(err.message)}</p>`;
  }
}

function renderQuizQuestion(area, msgIdx, question, rubric, topicId) {
  const card = document.createElement("div");
  card.className = "quiz-card border border-indigo-200";
  card.innerHTML = `<div class="flex items-center justify-between mb-2">
    <span class="text-sm font-semibold text-gray-800">🎯 Quick check</span>
    <button class="text-gray-400 hover:text-gray-600 text-xs dismiss-quiz">&#x2715;</button>
  </div>
  <p class="text-sm text-gray-700 mb-3">${esc(question)}</p>
  <textarea rows="3" class="quiz-textarea w-full text-sm border border-gray-200 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400 resize-none bg-gray-50 placeholder-gray-400" placeholder="Your answer…"></textarea>
  <div class="flex justify-end mt-2">
    <button class="quiz-submit text-xs bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg px-3 py-1.5 transition-colors">Submit</button>
  </div>
  <div class="quiz-result mt-2"></div>`;

  card.querySelector(".dismiss-quiz").addEventListener("click", () => { area.innerHTML = ""; });
  card.querySelector(".quiz-submit").addEventListener("click", () => submitQuiz(card, msgIdx, question, rubric, topicId));
  area.innerHTML = "";
  area.appendChild(card);
}

async function submitQuiz(card, msgIdx, question, rubric, topicId) {
  const answer = card.querySelector(".quiz-textarea").value.trim();
  if (!answer) { card.querySelector(".quiz-textarea").focus(); return; }

  const result = card.querySelector(".quiz-result");
  result.innerHTML = `<div class="flex items-center gap-2 text-xs text-gray-500"><div class="thinking-dots"><span></span><span></span><span></span></div> Grading…</div>`;

  try {
    const res = await fetch("/api/quiz/score", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ student_id: state.studentId, topic_id: topicId, question, rubric, answer }),
    });
    if (!res.ok) throw new Error(await res.text());
    const { score, rationale } = await res.json();
    const pct = Math.round(score * 100);
    const color = score >= 0.8 ? "green" : score >= 0.6 ? "amber" : "red";
    result.innerHTML = `<div class="flex items-center gap-2 mb-1">
      <span class="text-xl font-bold text-${color}-600">${pct}%</span>
      <span class="text-xs text-gray-500">${score >= 0.8 ? "Great work!" : score >= 0.6 ? "Partial — keep going." : "Needs more work."}</span>
    </div>
    <p class="text-xs text-gray-600 italic">${esc(rationale)}</p>`;

    if (score < 0.6) {
      const diagBtn = document.createElement("button");
      diagBtn.className = "mt-2 text-xs bg-amber-50 hover:bg-amber-100 text-amber-700 border border-amber-200 font-medium rounded-lg px-3 py-1.5 transition-colors";
      diagBtn.textContent = "🧭 Understand what went wrong";
      diagBtn.addEventListener("click", () => startDiagnostic(card, question, rubric, answer, score));
      result.appendChild(diagBtn);
    }
  } catch (err) {
    result.innerHTML = `<p class="text-xs text-red-500">Error: ${esc(err.message)}</p>`;
  }
}

// ── Diagnostic ─────────────────────────────────────────────────
async function startDiagnostic(quizCard, question, rubric, studentAnswer, score) {
  const area = quizCard.closest(".quiz-area");
  if (!area) return;

  const diagCard = document.createElement("div");
  diagCard.className = "diagnostic-card border border-amber-200";
  diagCard.innerHTML = `<div class="flex items-center gap-2 text-xs text-amber-600"><div class="thinking-dots"><span></span><span></span><span></span></div> Analyzing…</div>`;
  area.appendChild(diagCard);
  scrollBottom();

  try {
    const res = await fetch("/api/diagnostic/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ original_question: question, rubric, student_answer: studentAnswer, score }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderDiagFollowUp(diagCard, data, question, 1);
  } catch (err) {
    diagCard.innerHTML = `<p class="text-xs text-red-500">Error: ${esc(err.message)}</p>`;
  }
}

function renderDiagFollowUp(card, data, originalQuestion, turnIndex) {
  card.innerHTML = `<div class="mb-2">
    <span class="text-sm font-semibold text-amber-800">🧭 Let's figure this out</span>
    <p class="text-xs text-gray-500 italic mt-0.5">${esc(data.diagnosis)}</p>
  </div>
  <p class="text-sm text-gray-800 font-medium mb-3">${esc(data.follow_up_question)}</p>
  <textarea rows="2" class="diag-textarea w-full text-sm border border-amber-200 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-amber-400 resize-none bg-amber-50/50 placeholder-gray-400" placeholder="Your response…"></textarea>
  <div class="flex justify-end mt-2">
    <button class="diag-submit text-xs bg-amber-500 hover:bg-amber-600 text-white font-medium rounded-lg px-3 py-1.5 transition-colors">Continue</button>
  </div>
  <div class="diag-result mt-2"></div>`;

  card.querySelector(".diag-submit").addEventListener("click", () =>
    continueDiagnostic(card, originalQuestion, data, turnIndex)
  );
}

async function continueDiagnostic(card, originalQuestion, diagData, turnIndex) {
  const answer = card.querySelector(".diag-textarea")?.value.trim();
  if (!answer) { card.querySelector(".diag-textarea")?.focus(); return; }

  const result = card.querySelector(".diag-result");
  result.innerHTML = `<div class="flex items-center gap-2 text-xs text-amber-600"><div class="thinking-dots"><span></span><span></span><span></span></div> Thinking…</div>`;

  try {
    const res = await fetch("/api/diagnostic/turn", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        student_id: state.studentId,
        original_question: originalQuestion,
        diagnosis: diagData.diagnosis,
        follow_up_question: diagData.follow_up_question,
        student_answer: answer,
        turn_index: turnIndex,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    result.innerHTML = `<div class="prose-chat text-sm text-gray-700 mb-3">${renderMarkdown(data.next_message || data.explanation || "")}</div>`;
    if (data.next_action && data.next_action !== "wrap_up") {
      const nextBtn = document.createElement("button");
      nextBtn.className = "text-xs bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg px-3 py-1.5 transition-colors";
      nextBtn.textContent = "Next →";
      nextBtn.addEventListener("click", () => {
        renderDiagFollowUp(card, { diagnosis: diagData.diagnosis, follow_up_question: data.next_message || "" }, originalQuestion, turnIndex + 1);
      });
      result.appendChild(nextBtn);
    } else {
      const doneEl = document.createElement("p");
      doneEl.className = "text-xs text-emerald-700 font-medium mt-1";
      doneEl.textContent = "✅ Great — misconception recorded.";
      result.appendChild(doneEl);
    }
  } catch (err) {
    result.innerHTML = `<p class="text-xs text-red-500">Error: ${esc(err.message)}</p>`;
  }
}

// ── Load history ───────────────────────────────────────────────
async function loadHistory() {
  try {
    const r = await fetch(`/api/student/${encodeURIComponent(state.studentId)}/messages?limit=40`);
    if (!r.ok) return;
    const { messages } = await r.json();
    if (!messages || !messages.length) return;

    $("welcome").classList.add("hidden");

    // separator
    const sep = document.createElement("div");
    sep.className = "flex items-center gap-2 my-2";
    sep.innerHTML = `<div class="flex-1 border-t border-gray-200"></div><span class="text-xs text-gray-400 italic px-2">earlier session</span><div class="flex-1 border-t border-gray-200"></div>`;
    $("chat").appendChild(sep);

    messages.forEach((m, i) => {
      const idx = state.messages.length;
      state.messages.push({ role: m.role, content: m.content, references: [] });
      if (m.role === "user") appendUserBubble(m.content);
      else appendAssistantMessage(state.messages[idx], idx);
    });
    scrollBottom();
  } catch (_) {}
}

// ── New chat ───────────────────────────────────────────────────
function newChat() {
  state.messages = [];
  state.reuseCounts = {};
  $("chat").innerHTML = "";
  $("welcome").classList.remove("hidden");
  $("msgInput").focus();
}

// ── Events ─────────────────────────────────────────────────────
$("msgInput").addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 200) + "px";
  $("sendBtn").disabled = !this.value.trim();
});

$("msgInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

$("sendBtn").addEventListener("click", () => sendMessage());

$("newChatBtn").addEventListener("click", newChat);

$("budgetSlider").addEventListener("input", (e) => {
  state.budget = parseInt(e.target.value, 10);
  $("budgetVal").textContent = state.budget;
});

$("studentIdInput").addEventListener("change", (e) => {
  state.studentId = e.target.value.trim() || "you";
  state.reuseCounts = {};
  state.messages = [];
  $("chat").innerHTML = "";
  $("welcome").classList.remove("hidden");
  loadHistory();
});

$("topicSelect").addEventListener("change", (e) => {
  state.topicId = e.target.value || null;
});

$("clearHistoryBtn").addEventListener("click", () => {
  newChat();
  $("overflowMenu").removeAttribute("open");
});

// Routing modal close
$("routingClose").addEventListener("click", closeRoutingModal);
$("routingBackdrop").addEventListener("click", closeRoutingModal);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeRoutingModal(); });

// ── Init ───────────────────────────────────────────────────────
(async () => {
  buildStarterGrid();
  await loadTopics();
  await loadHistory();
  $("msgInput").focus();
})();
