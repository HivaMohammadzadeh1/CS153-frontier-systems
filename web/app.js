/**
 * Learning Memory OS — frontend app
 * Single-page chat UI using the FastAPI backend.
 */

// ============================================================
// Configuration
// ============================================================
const API_BASE = "";  // same origin

// ============================================================
// State
// ============================================================
const state = {
  studentId: "demo-user",
  topicId: null,
  budget: 3000,
  messages: [],       // {role, content, references?, selected?, dropped?, tokens_used?, quiz?, diagnostic?}
  reuseCounts: {},
  lastDecision: null,
};

// ============================================================
// Mermaid init
// ============================================================
mermaid.initialize({
  startOnLoad: false,
  theme: "neutral",
  themeVariables: {
    primaryColor: "#6366f1",
    primaryTextColor: "#fff",
    primaryBorderColor: "#4f46e5",
    lineColor: "#94a3b8",
    secondaryColor: "#eef2ff",
    tertiaryColor: "#f8fafc",
  },
  fontFamily: "Inter, system-ui, sans-serif",
});

// ============================================================
// Marked config
// ============================================================
marked.setOptions({
  breaks: true,
  gfm: true,
});

// ============================================================
// DOM helpers
// ============================================================
const $ = (id) => document.getElementById(id);

function scrollToBottom() {
  const h = $("chat-history");
  h.scrollTop = h.scrollHeight;
}

function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

function handleKeydown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function sendSuggestion(btn) {
  const text = btn.textContent.trim();
  $("chat-input").value = text;
  sendMessage();
}

// ============================================================
// New chat
// ============================================================
function startNewChat() {
  state.messages = [];
  state.reuseCounts = {};
  state.lastDecision = null;
  const chatEl = $("chat-history");
  if (chatEl) chatEl.innerHTML = "";
  // Re-insert welcome card
  const welcome = document.createElement("div");
  welcome.id = "welcome-card";
  welcome.className = "max-w-2xl mx-auto";
  welcome.innerHTML = `
    <div class="bg-white border border-slate-200 rounded-2xl shadow-sm p-6 text-center">
      <div class="w-12 h-12 bg-indigo-100 rounded-2xl mx-auto flex items-center justify-center mb-3">
        <svg class="w-6 h-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
      </div>
      <h2 class="text-lg font-semibold text-slate-900 mb-1">ML Systems Tutor</h2>
      <p class="text-sm text-slate-500 mb-4">Ask me anything about transformers, training, inference, or ML systems engineering.</p>
      <div class="flex flex-wrap justify-center gap-2">
        <button onclick="sendSuggestion(this)" class="text-xs bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-full px-3 py-1.5 hover:bg-indigo-100 transition-colors">What is KV caching and why does it matter?</button>
        <button onclick="sendSuggestion(this)" class="text-xs bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-full px-3 py-1.5 hover:bg-indigo-100 transition-colors">Explain flash attention vs vanilla attention</button>
        <button onclick="sendSuggestion(this)" class="text-xs bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-full px-3 py-1.5 hover:bg-indigo-100 transition-colors">How does pipeline parallelism work?</button>
        <button onclick="sendSuggestion(this)" class="text-xs bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-full px-3 py-1.5 hover:bg-indigo-100 transition-colors">Walk me through the transformer architecture</button>
      </div>
    </div>`;
  if (chatEl) chatEl.appendChild(welcome);
  // Reset routing drawer content
  const routingContent = $("routing-content");
  if (routingContent) {
    routingContent.innerHTML = `<div class="text-xs text-slate-400 text-center py-8">Ask a question to see routing details.</div>`;
  }
  $("chat-input")?.focus();
}
window.startNewChat = startNewChat;

// ============================================================
// Routing drawer toggle
// ============================================================
function toggleRoutingDrawer() {
  const drawer = $("right-drawer");
  const btn = $("routingToggle");
  if (!drawer || !btn) return;
  const isOpen = !drawer.classList.contains("hidden");
  if (isOpen) {
    drawer.classList.add("hidden");
    btn.innerHTML = `<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 0v10" /></svg> Show routing`;
  } else {
    drawer.classList.remove("hidden");
    btn.innerHTML = `<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 0v10" /></svg> Hide routing`;
  }
}
// Legacy alias (used from showRefDetail which opens the drawer directly)
function toggleDrawer() { toggleRoutingDrawer(); }
window.toggleDrawer = toggleDrawer;

// ============================================================
// Load topics
// ============================================================
async function loadTopics() {
  try {
    const res = await fetch(`${API_BASE}/api/topics`);
    if (!res.ok) return;
    const topics = await res.json();

    // Populate header select
    const sel = $("topic-select");
    topics.forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t.id;
      opt.textContent = t.title;
      sel.appendChild(opt);
    });

    // Populate left nav grouped by area
    const list = $("topic-list");
    list.innerHTML = "";

    const areas = {};
    topics.forEach((t) => {
      if (!areas[t.area]) areas[t.area] = [];
      areas[t.area].push(t);
    });

    const areaLabels = {
      A: "Model fundamentals",
      B: "Training systems",
      C: "Inference infra",
      D: "Data & alignment",
      E: "Agent systems",
    };

    Object.entries(areas).forEach(([area, aTopics]) => {
      const areaDiv = document.createElement("div");
      areaDiv.className = "mb-2";

      const header = document.createElement("div");
      header.className = "text-xs font-semibold text-slate-400 uppercase tracking-wider px-2 py-1 mt-1";
      header.textContent = areaLabels[area] || area;
      areaDiv.appendChild(header);

      aTopics.forEach((t) => {
        const item = document.createElement("div");
        item.className = "topic-item text-xs text-slate-600 rounded-lg px-2 py-1.5";
        item.textContent = t.title;
        item.dataset.topicId = t.id;
        item.onclick = () => selectTopic(t.id, item);
        areaDiv.appendChild(item);
      });

      list.appendChild(areaDiv);
    });
  } catch (err) {
    console.error("loadTopics error", err);
  }
}

function selectTopic(topicId, itemEl) {
  // Update state
  state.topicId = topicId;
  $("topic-select").value = topicId;

  // Highlight
  document.querySelectorAll(".topic-item").forEach((el) => el.classList.remove("active"));
  if (itemEl) itemEl.classList.add("active");
}

// ============================================================
// Load chat history
// ============================================================
async function loadHistory() {
  try {
    const r = await fetch(`/api/student/${encodeURIComponent(state.studentId)}/messages?limit=40`);
    if (!r.ok) return;
    const data = await r.json();
    if (!data.messages || data.messages.length === 0) return;

    // Clear existing chat UI + state.messages
    state.messages = [];
    const chatEl = $("chat-history");
    if (chatEl) chatEl.innerHTML = "";

    // Add a subtle "earlier session" separator before restored messages
    const sep = document.createElement("div");
    sep.className = "max-w-3xl mx-auto my-1";
    sep.innerHTML = `<div class="flex items-center gap-2">
      <div class="flex-1 border-t border-slate-200"></div>
      <span class="text-xs text-slate-400 italic px-2">earlier session</span>
      <div class="flex-1 border-t border-slate-200"></div>
    </div>`;
    chatEl.appendChild(sep);

    // Replay each message
    for (const m of data.messages) {
      state.messages.push({
        role: m.role,
        content: m.content,
        references: [],   // not stored; restored messages have inline [shortid] markers
        restored: true,
        timestamp: m.timestamp,
      });
      const idx = state.messages.length - 1;
      let el;
      if (m.role === "user") {
        el = renderUserMessage(state.messages[idx]);
      } else {
        el = renderAssistantMessage(state.messages[idx], idx);
      }
      chatEl.appendChild(el);
    }

    // Hide the welcome card if there are messages
    const welcome = $("welcome-card");
    if (welcome) welcome.style.display = "none";

    scrollToBottom();
  } catch (e) {
    console.warn("loadHistory failed:", e);
  }
}

// ============================================================
// Student state
// ============================================================
async function refreshStudentState() {
  const sid = $("student-id").value.trim() || "demo-user";
  try {
    const res = await fetch(`${API_BASE}/api/student/${encodeURIComponent(sid)}/state`);
    if (!res.ok) return;
    const data = await res.json();
    const el = $("student-state-summary");
    const masteryCount = data.mastery?.length ?? 0;
    const miscCount = data.misconceptions?.length ?? 0;
    el.innerHTML = `
      <div class="flex items-center gap-1.5">
        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
        <span>${masteryCount} mastery entries</span>
      </div>
      <div class="flex items-center gap-1.5">
        <span class="w-1.5 h-1.5 rounded-full ${miscCount > 0 ? 'bg-amber-400' : 'bg-slate-300'}"></span>
        <span>${miscCount} active misconception${miscCount !== 1 ? 's' : ''}</span>
      </div>`;
  } catch (err) {
    console.error("refreshStudentState error", err);
  }
}
window.refreshStudentState = refreshStudentState;

// ============================================================
// Markdown + Mermaid rendering
// ============================================================
function renderMarkdown(text) {
  // Extract mermaid blocks before markdown parsing so they're not escaped
  const mermaidBlocks = [];
  const placeholder = (i) => `__MERMAID_${i}__`;

  const processed = text.replace(/```mermaid\s*([\s\S]*?)```/g, (_, code) => {
    const idx = mermaidBlocks.length;
    mermaidBlocks.push(code.trim());
    return `\n${placeholder(idx)}\n`;
  });

  // Parse markdown
  let html = DOMPurify.sanitize(marked.parse(processed));

  // Substitute mermaid placeholders with <pre class="mermaid">
  mermaidBlocks.forEach((code, i) => {
    const safe = DOMPurify.sanitize(code, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] });
    html = html.replace(
      new RegExp(`<p>${placeholder(i)}</p>|${placeholder(i)}`, "g"),
      `<pre class="mermaid">${safe}</pre>`
    );
  });

  return html;
}

async function renderMermaidInEl(el) {
  const blocks = el.querySelectorAll("pre.mermaid");
  if (blocks.length === 0) return;
  try {
    await mermaid.run({ nodes: Array.from(blocks) });
  } catch (err) {
    blocks.forEach((b) => {
      b.style.color = "#ef4444";
      b.style.fontSize = "0.75rem";
      b.textContent = "[Diagram render error] " + err.message;
    });
  }
}

// ============================================================
// Message rendering
// ============================================================
function createThinkingIndicator() {
  const div = document.createElement("div");
  div.className = "flex items-start gap-3 max-w-3xl mx-auto msg-assistant";
  div.id = "thinking-indicator";
  div.innerHTML = `
    <div class="w-8 h-8 rounded-xl bg-indigo-100 flex-shrink-0 flex items-center justify-center mt-0.5">
      <svg class="w-4 h-4 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.347.345A3.001 3.001 0 0112 21a3.001 3.001 0 01-2.091-.755l-.348-.345z" />
      </svg>
    </div>
    <div class="bg-white border border-slate-200 rounded-2xl px-5 py-4 shadow-sm flex-1">
      <div class="thinking-dots flex items-center gap-1 h-5">
        <span></span><span></span><span></span>
      </div>
    </div>`;
  return div;
}

function renderUserMessage(msg) {
  const div = document.createElement("div");
  div.className = "flex items-end justify-end gap-2 max-w-3xl mx-auto msg-user";
  div.innerHTML = `
    <div class="bg-indigo-600 text-white rounded-2xl rounded-br-md px-4 py-3 max-w-xl shadow-sm">
      <p class="text-sm leading-relaxed whitespace-pre-wrap">${escapeHtml(msg.content)}</p>
    </div>
    <div class="w-7 h-7 rounded-full bg-slate-200 flex-shrink-0 flex items-center justify-center mb-0.5">
      <svg class="w-3.5 h-3.5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
      </svg>
    </div>`;
  return div;
}

function renderAssistantMessage(msg, msgIdx) {
  const div = document.createElement("div");
  div.className = "flex items-start gap-3 max-w-3xl mx-auto msg-assistant";
  div.dataset.msgIdx = msgIdx;

  const bodyHtml = renderMarkdown(msg.content);

  // References section
  let refsHtml = "";
  if (msg.references && msg.references.length > 0) {
    const pills = msg.references
      .map(
        (r) =>
          `<span class="ref-pill" title="${escapeHtml(r.title)}" onclick="showRefDetail(${msgIdx}, '${r.id}')">[${r.n}] ${escapeHtml(r.title.length > 30 ? r.title.slice(0, 30) + "…" : r.title)}</span>`
      )
      .join("");
    refsHtml = `<div class="flex flex-wrap gap-1.5 mt-3 pt-2.5 border-t border-slate-100">${pills}</div>`;
  }

  // Token usage badge
  const tokensBadge =
    msg.tokens_used != null
      ? `<span class="text-xs text-slate-400">${msg.tokens_used} / ${state.budget} tokens</span>`
      : "";

  div.innerHTML = `
    <div class="w-8 h-8 rounded-xl bg-indigo-100 flex-shrink-0 flex items-center justify-center mt-0.5">
      <svg class="w-4 h-4 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.347.345A3.001 3.001 0 0112 21a3.001 3.001 0 01-2.091-.755l-.348-.345z" />
      </svg>
    </div>
    <div class="bg-white border border-slate-200 rounded-2xl px-5 py-4 shadow-sm flex-1 min-w-0">
      <div class="prose-chat text-sm text-slate-700">${bodyHtml}</div>
      ${refsHtml}
      <div class="flex items-center justify-between mt-2.5">
        ${tokensBadge}
        <button onclick="triggerQuizInline(${msgIdx})"
          class="text-xs text-indigo-500 hover:text-indigo-700 flex items-center gap-1 ml-auto">
          <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
          </svg>
          Test me
        </button>
      </div>
      <div class="quiz-container mt-1"></div>
    </div>`;

  return div;
}

function renderQuizCard(msgIdx, quiz, container) {
  container.innerHTML = "";
  if (!quiz) return;

  const card = document.createElement("div");
  card.className = "quiz-card";

  if (quiz.state === "loading") {
    card.innerHTML = `
      <div class="flex items-center gap-2 text-xs text-slate-500">
        <div class="thinking-dots"><span></span><span></span><span></span></div>
        Generating question…
      </div>`;
  } else if (quiz.state === "question") {
    card.innerHTML = `
      <div class="flex items-center gap-2 mb-2">
        <span class="text-base">🎯</span>
        <span class="text-sm font-semibold text-slate-800">Challenge</span>
      </div>
      <p class="text-sm text-slate-700 leading-relaxed mb-3">${escapeHtml(quiz.question)}</p>
      <textarea id="quiz-answer-${msgIdx}" rows="2"
        class="w-full text-sm border border-slate-200 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none bg-slate-50 placeholder-slate-400"
        placeholder="Your answer…"></textarea>
      <div class="flex items-center justify-between mt-2">
        <span class="text-xs text-slate-400 italic">Be as detailed as you like.</span>
        <button onclick="submitQuizAnswer(${msgIdx})"
          class="text-xs bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg px-3 py-1.5 transition-colors">
          Submit answer
        </button>
      </div>`;
  } else if (quiz.state === "scoring") {
    card.innerHTML = `
      <div class="flex items-center gap-2 text-xs text-slate-500">
        <div class="thinking-dots"><span></span><span></span><span></span></div>
        Grading your answer…
      </div>`;
  } else if (quiz.state === "scored") {
    const score = quiz.score ?? 0;
    const pct = Math.round(score * 100);
    const color = score >= 0.8 ? "emerald" : score >= 0.5 ? "amber" : "red";
    const emoji = score >= 0.8 ? "✅" : score >= 0.5 ? "🔶" : "❌";
    const label = score >= 0.8 ? "Excellent" : score >= 0.5 ? "Partial" : "Needs work";

    card.innerHTML = `
      <div class="flex items-center gap-2 mb-3">
        <span class="text-base">${emoji}</span>
        <span class="text-sm font-semibold text-slate-800">Result: ${label}</span>
        <span class="ml-auto text-lg font-bold text-${color}-600">${pct}%</span>
      </div>
      <div class="h-2 bg-slate-100 rounded-full mb-3 overflow-hidden">
        <div class="h-full bg-${color}-500 rounded-full transition-all" style="width:${pct}%"></div>
      </div>
      <p class="text-xs text-slate-500 italic leading-relaxed">${escapeHtml(quiz.rationale ?? "")}</p>
      <div class="quiz-diagnostic-container mt-2"></div>
      ${
        score < 0.6
          ? `<button onclick="startDiagnosticFromQuiz(${msgIdx})"
              class="mt-2 text-xs bg-amber-50 hover:bg-amber-100 text-amber-700 border border-amber-200 font-medium rounded-lg px-3 py-1.5 transition-colors flex items-center gap-1.5 w-full justify-center">
              🧭 Let's figure out what went wrong
            </button>`
          : ""
      }`;
  }

  container.appendChild(card);
}

function renderDiagnosticCard(msgIdx, diag, container) {
  if (!diag) return;
  container.innerHTML = "";

  const card = document.createElement("div");
  card.className = "diagnostic-card";

  if (diag.state === "loading") {
    card.innerHTML = `
      <div class="flex items-center gap-2 text-xs text-amber-600">
        <div class="thinking-dots"><span></span><span></span><span></span></div>
        Analyzing your answer…
      </div>`;
  } else if (diag.state === "follow_up") {
    card.innerHTML = `
      <div class="flex items-center gap-2 mb-2">
        <span class="text-base">🧭</span>
        <span class="text-sm font-semibold text-amber-800">Let's understand what went wrong</span>
      </div>
      <p class="text-xs text-slate-500 italic mb-2.5">${escapeHtml(diag.diagnosis)}</p>
      <p class="text-sm text-slate-700 font-medium leading-relaxed mb-3">${escapeHtml(diag.follow_up_question)}</p>
      <textarea id="diag-answer-${msgIdx}" rows="2"
        class="w-full text-sm border border-amber-200 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-amber-400 resize-none bg-amber-50/50 placeholder-slate-400"
        placeholder="Your response…"></textarea>
      <div class="flex justify-end mt-2">
        <button onclick="submitDiagnosticAnswer(${msgIdx})"
          class="text-xs bg-amber-500 hover:bg-amber-600 text-white font-medium rounded-lg px-3 py-1.5 transition-colors">
          Continue
        </button>
      </div>`;
  } else if (diag.state === "explaining") {
    card.innerHTML = `
      <div class="flex items-center gap-2 text-xs text-amber-600">
        <div class="thinking-dots"><span></span><span></span><span></span></div>
        Formulating response…
      </div>`;
  } else if (diag.state === "result") {
    const actionLabel = { explain: "Continue explaining", re_test: "Re-test me", wrap_up: "Got it!" };
    const actionStyle = {
      explain: "bg-indigo-600 hover:bg-indigo-700 text-white",
      re_test: "bg-amber-500 hover:bg-amber-600 text-white",
      wrap_up: "bg-emerald-600 hover:bg-emerald-700 text-white",
    };
    const nextAction = diag.next_action || "wrap_up";

    card.innerHTML = `
      <div class="flex items-center gap-2 mb-2">
        <span class="text-base">💡</span>
        <span class="text-sm font-semibold text-slate-800">Tutor's response</span>
      </div>
      <div class="prose-chat text-sm text-slate-700 mb-3">${renderMarkdown(diag.next_message || diag.explanation || "")}</div>
      <button onclick="handleDiagnosticAction(${msgIdx}, '${nextAction}')"
        class="text-xs ${actionStyle[nextAction] || actionStyle.wrap_up} font-medium rounded-lg px-3 py-1.5 transition-colors">
        ${actionLabel[nextAction] || "Done"}
      </button>`;
  } else if (diag.state === "done") {
    card.innerHTML = `
      <div class="flex items-center gap-2">
        <span class="text-base">✅</span>
        <span class="text-sm font-semibold text-emerald-700">Great work! Misconception recorded.</span>
      </div>`;
  }

  container.appendChild(card);
}

// ============================================================
// Quiz
// ============================================================
window.triggerQuizInline = async function (msgIdx) {
  const topicId = state.topicId || ($("topic-select").value || null);
  if (!topicId) {
    alert("Select a topic first to generate a quiz question.");
    return;
  }
  const msgEl = document.querySelector(`[data-msg-idx="${msgIdx}"]`);
  if (!msgEl) return;
  const container = msgEl.querySelector(".quiz-container");
  if (!container) return;

  state.messages[msgIdx].quiz = { state: "loading", topicId };
  renderQuizCard(msgIdx, state.messages[msgIdx].quiz, container);

  try {
    const res = await fetch(`${API_BASE}/api/quiz/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic_id: topicId }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    state.messages[msgIdx].quiz = { state: "question", topicId, ...data };
    renderQuizCard(msgIdx, state.messages[msgIdx].quiz, container);
  } catch (err) {
    state.messages[msgIdx].quiz = null;
    container.innerHTML = `<p class="text-xs text-red-500 mt-2">Error generating quiz: ${escapeHtml(err.message)}</p>`;
  }
};

window.triggerQuiz = function () {
  // Triggered from the header quiz button — attaches to the last assistant message
  const idx = state.messages.findLastIndex((m) => m.role === "assistant");
  if (idx < 0) {
    alert("Ask a question first, then I can generate a quiz on the topic.");
    return;
  }
  triggerQuizInline(idx);
};

window.submitQuizAnswer = async function (msgIdx) {
  const msg = state.messages[msgIdx];
  if (!msg?.quiz) return;
  const textarea = $(`quiz-answer-${msgIdx}`);
  const answer = textarea?.value.trim();
  if (!answer) {
    textarea?.focus();
    return;
  }

  const msgEl = document.querySelector(`[data-msg-idx="${msgIdx}"]`);
  const container = msgEl?.querySelector(".quiz-container");
  if (!container) return;

  msg.quiz.state = "scoring";
  msg.quiz.student_answer = answer;
  renderQuizCard(msgIdx, msg.quiz, container);

  const sid = $("student-id").value.trim() || "demo-user";
  try {
    const res = await fetch(`${API_BASE}/api/quiz/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        student_id: sid,
        topic_id: msg.quiz.topicId,
        question: msg.quiz.question,
        rubric: msg.quiz.rubric,
        answer,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    msg.quiz = { ...msg.quiz, state: "scored", ...data };
    renderQuizCard(msgIdx, msg.quiz, container);
  } catch (err) {
    container.innerHTML = `<p class="text-xs text-red-500 mt-2">Error scoring: ${escapeHtml(err.message)}</p>`;
  }
};

// ============================================================
// Diagnostic
// ============================================================
window.startDiagnosticFromQuiz = async function (msgIdx) {
  const msg = state.messages[msgIdx];
  if (!msg?.quiz) return;

  const msgEl = document.querySelector(`[data-msg-idx="${msgIdx}"]`);
  const container = msgEl?.querySelector(".quiz-diagnostic-container");
  if (!container) return;

  msg.diagnostic = { state: "loading" };
  renderDiagnosticCard(msgIdx, msg.diagnostic, container);

  try {
    const res = await fetch(`${API_BASE}/api/diagnostic/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        original_question: msg.quiz.question,
        rubric: msg.quiz.rubric,
        student_answer: msg.quiz.student_answer || "",
        score: msg.quiz.score || 0,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    msg.diagnostic = { state: "follow_up", turnIndex: 1, ...data };
    renderDiagnosticCard(msgIdx, msg.diagnostic, container);
  } catch (err) {
    container.innerHTML = `<p class="text-xs text-red-500 mt-2">Error: ${escapeHtml(err.message)}</p>`;
  }
};

window.submitDiagnosticAnswer = async function (msgIdx) {
  const msg = state.messages[msgIdx];
  if (!msg?.diagnostic) return;
  const textarea = $(`diag-answer-${msgIdx}`);
  const answer = textarea?.value.trim();
  if (!answer) { textarea?.focus(); return; }

  const msgEl = document.querySelector(`[data-msg-idx="${msgIdx}"]`);
  const container = msgEl?.querySelector(".quiz-diagnostic-container");
  if (!container) return;

  msg.diagnostic.state = "explaining";
  msg.diagnostic.student_answer = answer;
  renderDiagnosticCard(msgIdx, msg.diagnostic, container);

  const sid = $("student-id").value.trim() || "demo-user";
  try {
    const res = await fetch(`${API_BASE}/api/diagnostic/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        student_id: sid,
        original_question: msg.quiz.question,
        diagnosis: msg.diagnostic.diagnosis,
        follow_up_question: msg.diagnostic.follow_up_question,
        student_answer: answer,
        turn_index: msg.diagnostic.turnIndex || 1,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    msg.diagnostic = { ...msg.diagnostic, state: "result", ...data };
    renderDiagnosticCard(msgIdx, msg.diagnostic, container);
  } catch (err) {
    container.innerHTML = `<p class="text-xs text-red-500 mt-2">Error: ${escapeHtml(err.message)}</p>`;
  }
};

window.handleDiagnosticAction = async function (msgIdx, action) {
  const msg = state.messages[msgIdx];
  if (!msg?.diagnostic) return;

  const msgEl = document.querySelector(`[data-msg-idx="${msgIdx}"]`);
  const container = msgEl?.querySelector(".quiz-diagnostic-container");

  if (action === "wrap_up") {
    msg.diagnostic = { ...msg.diagnostic, state: "done" };
    renderDiagnosticCard(msgIdx, msg.diagnostic, container);
  } else if (action === "re_test") {
    // Generate fresh quiz question on same topic
    msg.diagnostic = { ...msg.diagnostic, state: "done" };
    renderDiagnosticCard(msgIdx, msg.diagnostic, container);
    triggerQuizInline(msgIdx);
  } else {
    // "explain" — cycle another turn asking a follow-up
    // Reset to follow_up state with a new question from the tutor message
    const followUp = msg.diagnostic.next_message || msg.diagnostic.explanation;
    msg.diagnostic = {
      ...msg.diagnostic,
      state: "follow_up",
      follow_up_question: followUp,
      turnIndex: (msg.diagnostic.turnIndex || 1) + 1,
    };
    renderDiagnosticCard(msgIdx, msg.diagnostic, container);
  }
};

// ============================================================
// References
// ============================================================
window.showRefDetail = function (msgIdx, refId) {
  const msg = state.messages[msgIdx];
  if (!msg) return;
  const item = msg.selected?.find((it) => it.id === refId)
    || msg.dropped?.find((it) => it.id === refId);
  if (!item) return;

  // Show in routing drawer (open it if closed)
  const drawer = $("right-drawer");
  if (drawer.classList.contains("hidden")) {
    toggleRoutingDrawer();
  }
  const content = $("routing-content");
  content.innerHTML = `
    <div class="mb-3">
      <button onclick="renderRoutingDetails(${msgIdx})"
        class="text-xs text-indigo-500 hover:text-indigo-700 flex items-center gap-1">
        ← Back to routing
      </button>
    </div>
    <h3 class="text-sm font-semibold text-slate-800 mb-1">${escapeHtml(item.title)}</h3>
    <div class="flex flex-wrap gap-1.5 mb-3">
      ${scoreChip("Total", item.score_total, "indigo")}
      ${scoreChip("Relevance", item.score_relevance, "blue")}
      ${scoreChip("Recency", item.score_recency, "cyan")}
      ${scoreChip("Misc.", item.score_misconception, "amber")}
      ${scoreChip("Prereq", item.score_prerequisite, "emerald")}
      ${scoreChip("Reuse", item.score_reuse, "rose")}
    </div>
    <div class="text-xs text-slate-500 bg-slate-50 rounded-xl p-3 leading-relaxed max-h-96 overflow-y-auto whitespace-pre-wrap font-mono">${escapeHtml(item.body.slice(0, 1000))}${item.body.length > 1000 ? "\n…" : ""}</div>`;
};

function scoreChip(label, value, color) {
  const pct = Math.round((value || 0) * 100);
  return `<span class="text-xs bg-${color}-50 text-${color}-700 border border-${color}-200 rounded-full px-2 py-0.5">${label}: ${pct}%</span>`;
}

// ============================================================
// Routing drawer
// ============================================================
function renderRoutingDetails(msgIdx) {
  const msg = state.messages[msgIdx];
  if (!msg || !msg.selected) {
    $("routing-content").innerHTML = `<div class="text-xs text-slate-400 text-center py-8">No routing data.</div>`;
    return;
  }

  const sel = msg.selected;
  const dropped = msg.dropped || [];

  let html = `
    <div class="mb-3">
      <div class="flex items-center justify-between text-xs text-slate-500 mb-1">
        <span>Budget</span>
        <span class="font-medium">${msg.tokens_used ?? 0} / ${state.budget}</span>
      </div>
      <div class="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div class="h-full bg-indigo-500 rounded-full" style="width:${Math.min(100, Math.round(((msg.tokens_used ?? 0) / state.budget) * 100))}%"></div>
      </div>
    </div>

    <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Selected (${sel.length})</p>`;

  sel.forEach((it) => {
    html += routingItemHtml(it, "emerald", msgIdx);
  });

  if (dropped.length > 0) {
    html += `<p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mt-3 mb-2">Dropped top (${dropped.length})</p>`;
    dropped.forEach((it) => {
      html += routingItemHtml(it, "slate", msgIdx);
    });
  }

  $("routing-content").innerHTML = html;
}

function routingItemHtml(it, colorClass, msgIdx) {
  const pct = Math.round((it.score_total || 0) * 100);
  return `
    <div class="mb-2 cursor-pointer hover:bg-slate-50 rounded-lg p-1.5 -mx-1" onclick="showRefDetail(${msgIdx}, '${it.id}')">
      <div class="text-xs text-slate-700 font-medium leading-snug mb-1">${escapeHtml(it.title.length > 40 ? it.title.slice(0, 40) + "…" : it.title)}</div>
      <div class="routing-score-row">
        <span class="routing-score-label">total</span>
        <div class="routing-score-track">
          <div class="routing-score-fill bg-${colorClass}-500" style="width:${pct}%"></div>
        </div>
        <span class="text-xs text-slate-500 w-8 text-right">${pct}%</span>
      </div>
    </div>`;
}

// ============================================================
// Send message
// ============================================================
async function sendMessage() {
  const input = $("chat-input");
  const text = input.value.trim();
  if (!text) return;

  // Clear welcome card
  const welcome = $("welcome-card");
  if (welcome) welcome.remove();

  // Sync state from UI
  state.studentId = $("student-id").value.trim() || "demo-user";
  state.topicId = $("topic-select").value || null;
  state.budget = parseInt($("budget-slider").value, 10) || 3000;

  // Push user message
  const userMsg = { role: "user", content: text };
  state.messages.push(userMsg);
  const userEl = renderUserMessage(userMsg);
  $("chat-history").appendChild(userEl);

  // Clear input
  input.value = "";
  input.style.height = "auto";

  // Disable input while waiting
  input.disabled = true;
  $("send-btn").disabled = true;

  // Show thinking
  const thinking = createThinkingIndicator();
  $("chat-history").appendChild(thinking);
  scrollToBottom();

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
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
      const errText = await res.text();
      appendErrorMessage(errText);
      return;
    }

    const data = await res.json();

    // Update reuse counts
    data.selected?.forEach((it) => {
      state.reuseCounts[it.id] = (state.reuseCounts[it.id] || 0) + 1;
    });

    const msgIdx = state.messages.length;
    const assistantMsg = {
      role: "assistant",
      content: data.reply,
      references: data.references,
      selected: data.selected,
      dropped: data.dropped,
      tokens_used: data.tokens_used,
    };
    state.messages.push(assistantMsg);
    state.lastDecision = data;

    const el = renderAssistantMessage(assistantMsg, msgIdx);
    $("chat-history").appendChild(el);

    // Render mermaid diagrams
    await renderMermaidInEl(el);

    // Update routing drawer
    renderRoutingDetails(msgIdx);

    scrollToBottom();
    refreshStudentState();
  } catch (err) {
    thinking.remove();
    appendErrorMessage(err.message);
  } finally {
    input.disabled = false;
    $("send-btn").disabled = false;
    input.focus();
  }
}
window.sendMessage = sendMessage;

function appendErrorMessage(text) {
  const div = document.createElement("div");
  div.className = "max-w-3xl mx-auto";
  div.innerHTML = `
    <div class="bg-red-50 border border-red-200 text-red-700 rounded-2xl px-4 py-3 text-sm">
      <strong>Error:</strong> ${escapeHtml(text)}
    </div>`;
  $("chat-history").appendChild(div);
  scrollToBottom();
}

// ============================================================
// Utilities
// ============================================================
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ============================================================
// Header / settings controls
// ============================================================
$("budget-slider").addEventListener("input", (e) => {
  state.budget = parseInt(e.target.value, 10);
  $("budget-label").textContent = state.budget;
});

$("topic-select").addEventListener("change", (e) => {
  state.topicId = e.target.value || null;
  // Sync left nav highlight
  document.querySelectorAll(".topic-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.topicId === state.topicId);
  });
});

let _historyDebounceTimer = null;

$("student-id").addEventListener("input", (e) => {
  // Debounce: wait 500ms after typing stops before reloading history
  clearTimeout(_historyDebounceTimer);
  const newId = e.target.value.trim() || "demo-user";
  _historyDebounceTimer = setTimeout(() => {
    state.studentId = newId;
    state.reuseCounts = {};  // reset reuse on student switch
    refreshStudentState();
    loadHistory();
  }, 500);
});

$("student-id").addEventListener("change", (e) => {
  state.studentId = e.target.value.trim() || "demo-user";
  state.reuseCounts = {};  // reset reuse on student switch
  refreshStudentState();
});

// New chat + routing drawer toggle
$("newChatBtn")?.addEventListener("click", startNewChat);
$("newChatBtnLeft")?.addEventListener("click", startNewChat);
$("routingToggle")?.addEventListener("click", toggleRoutingDrawer);
$("routingClose")?.addEventListener("click", toggleRoutingDrawer);

// ============================================================
// Bootstrap
// ============================================================
(async () => {
  await loadTopics();
  await refreshStudentState();
  await loadHistory();
  $("chat-input").focus();
})();
