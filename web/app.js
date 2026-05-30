/**
 * Memex — ML Systems Tutor
 * Dashboard-home + chat single-page app. A tiny hash router switches between
 * Home / Chat / Progress / Path views; all chat logic is preserved.
 */

// ── State ──────────────────────────────────────────────────────
const state = {
  studentId: "Hiva",
  budget: 3000,
  topicId: null,
  messages: [],
  reuseCounts: {},
  conversationId: null,
  conversations: [],
  view: "home",
};

const TOPIC_MAP = {};   // id -> { title, area }
const AREA_NAMES = {};  // area letter -> human-readable name
let _progressCache = null;

function areaName(letter) {
  const n = AREA_NAMES[letter];
  if (n) return n.replace(/\s*\(.*?\)\s*$/, "").trim();   // drop course-code suffix
  return letter && letter !== "?" ? `Area ${letter}` : "Other topics";
}

// ── Helpers ────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "")
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

function scrollBottom() {
  const sc = $("chatScroll");
  if (sc) sc.scrollTo({ top: sc.scrollHeight, behavior: "smooth" });
}

function masteryColor(score) { return score >= 0.7 ? "good" : score >= 0.4 ? "warn" : "bad"; }

function countUp(el, target, suffix = "") {
  const start = performance.now(), dur = 700;
  function step(now) {
    const t = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(target * eased) + suffix;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function toast(html) {
  const t = document.createElement("div");
  t.className = "toast";
  t.innerHTML = html;
  $("toastLayer").appendChild(t);
  setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity .4s"; }, 2600);
  setTimeout(() => t.remove(), 3100);
}

// ── Theme ──────────────────────────────────────────────────────
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("memex.theme", theme); } catch (_) {}
  const icon = $("themeIcon");
  if (icon) icon.textContent = theme === "dark" ? "☀️" : "🌙";
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute("data-theme") || "light";
  applyTheme(cur === "dark" ? "light" : "dark");
  // Re-render the active data view so canvas-y bits (ring) recolor cleanly.
  if (state.view !== "chat") showView(state.view, false);
}

// ── Streak ─────────────────────────────────────────────────────
function _today() { return new Date().toISOString().slice(0, 10); }
function _yesterday() { return new Date(Date.now() - 86400000).toISOString().slice(0, 10); }

function currentStreak() {
  const last = localStorage.getItem("memex.streak.last");
  const count = parseInt(localStorage.getItem("memex.streak.count") || "0", 10);
  if (!last) return 0;
  return (last === _today() || last === _yesterday()) ? count : 0;
}
function bumpStreak() {
  const last = localStorage.getItem("memex.streak.last");
  let count = parseInt(localStorage.getItem("memex.streak.count") || "0", 10);
  const today = _today();
  if (last === today) return count;
  count = (last === _yesterday()) ? count + 1 : 1;
  localStorage.setItem("memex.streak.last", today);
  localStorage.setItem("memex.streak.count", String(count));
  renderStreak();
  toast(`🔥 <span class="t-grad">${count}-day streak!</span>`);
  return count;
}
function renderStreak() { const el = $("streakCount"); if (el) el.textContent = currentStreak(); }

// ── Mermaid + Marked ───────────────────────────────────────────
mermaid.initialize({ startOnLoad: false, theme: "neutral", fontFamily: "Inter, system-ui, sans-serif" });
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
    html = html.replace(new RegExp(`<p>${ph(i)}</p>|${ph(i)}`, "g"), `<pre class="mermaid">${safe}</pre>`);
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
function substituteCitations(text, references) {
  if (!references || !references.length) return text;
  let out = text;
  references.forEach((r) => { out = out.replaceAll(`[${r.id}]`, `[${r.n}]`); });
  return out;
}

// ── Router / views ─────────────────────────────────────────────
const VIEWS = ["home", "profile", "chat", "progress", "path"];

function showView(name, pushHash = true) {
  if (!VIEWS.includes(name)) name = "home";
  state.view = name;
  VIEWS.forEach((v) => {
    $(`view-${v}`).classList.toggle("active", v === name);
    document.querySelector(`.nav-item[data-view="${v}"]`)?.classList.toggle("active", v === name);
  });
  if (pushHash) location.hash = `#/${name}`;
  if (name === "home") renderHome();
  else if (name === "profile") renderProfile();
  else if (name === "progress") renderProgress();
  else if (name === "path") renderPath();
  else if (name === "chat") $("msgInput")?.focus();
}

function routeFromHash() {
  const m = (location.hash || "").replace(/^#\/?/, "");
  showView(VIEWS.includes(m) ? m : "home", false);
}

// ── Data ───────────────────────────────────────────────────────
async function loadTopics() {
  try {
    const r = await fetch("/api/topics");
    if (!r.ok) return;
    const topics = await r.json();
    const sel = $("topicSelect");
    topics.forEach((t) => {
      const area = t.area || "?";
      TOPIC_MAP[t.id] = { title: t.title, area };
      if (t.area_title) AREA_NAMES[area] = t.area_title;
      const o = document.createElement("option");
      o.value = t.id; o.textContent = t.title;
      sel.appendChild(o);
    });
  } catch (_) {}
}

async function loadProgress(force = false) {
  if (!force && _progressCache) return _progressCache;
  try {
    const r = await fetch(`/api/student/${encodeURIComponent(state.studentId)}/progress`);
    if (!r.ok) return _progressCache;
    _progressCache = await r.json();
    return _progressCache;
  } catch (_) { return _progressCache; }
}

let _activityCache = null;
async function loadActivity(force = false) {
  if (!force && _activityCache) return _activityCache;
  try {
    const r = await fetch(`/api/student/${encodeURIComponent(state.studentId)}/activity`);
    if (!r.ok) return _activityCache;
    _activityCache = await r.json();
    return _activityCache;
  } catch (_) { return _activityCache; }
}

// Derived rollups from progress + topic catalog.
function deriveSummary(data) {
  const topics = (data && data.topics) || [];
  const misc = (data && data.misconceptions) || [];
  const withData = topics.filter((t) => typeof t.avg_mastery === "number");
  const overall = withData.length
    ? withData.reduce((s, t) => s + t.avg_mastery, 0) / withData.length : 0;

  // Areas
  const areaAgg = {};
  withData.forEach((t) => {
    const area = (TOPIC_MAP[t.topic_id]?.area) || "?";
    (areaAgg[area] ||= []).push(t.avg_mastery);
  });
  const areas = Object.entries(areaAgg)
    .map(([area, vals]) => ({ area, mastery: vals.reduce((a, b) => a + b, 0) / vals.length }))
    .sort((a, b) => a.area.localeCompare(b.area));

  // Up next: lowest in-progress topic, else first untouched catalog topic
  const inProgress = withData.filter((t) => t.avg_mastery < 0.7).sort((a, b) => a.avg_mastery - b.avg_mastery);
  let upNext = inProgress[0] ? { topic_id: inProgress[0].topic_id, mastery: inProgress[0].avg_mastery } : null;
  if (!upNext) {
    const touched = new Set(withData.map((t) => t.topic_id));
    const fresh = Object.keys(TOPIC_MAP).find((id) => !touched.has(id));
    if (fresh) upNext = { topic_id: fresh, mastery: null };
  }

  // Weak spot: lowest mastery topic, else first misconception
  const weakTopic = withData.slice().sort((a, b) => a.avg_mastery - b.avg_mastery)[0] || null;

  return { overall, areas, upNext, weakTopic, misc, topics: withData, hasData: withData.length > 0 };
}

function topicTitle(id) { return TOPIC_MAP[id]?.title || id; }

// ── Home view ──────────────────────────────────────────────────
async function renderHome() {
  const el = $("view-home");
  const [data, act] = await Promise.all([loadProgress(true), loadActivity(true)]);
  const s = deriveSummary(data);
  const ast = (act && act.stats) || {};
  const streak = currentStreak();
  const overallPct = Math.round(s.overall * 100);

  const homeStats = [
    { num: ast.questions ?? 0, lbl: "Questions" },
    { num: ast.quizzes ?? 0, lbl: "Quizzes" },
    { num: ast.topics_touched ?? 0, lbl: "Topics" },
    { num: ast.concepts_mastered ?? 0, lbl: "Mastered" },
  ];
  const homeStatsHtml = `<div class="stat-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:16px">${
    homeStats.map((x) => `<div class="stat"><div class="stat-num" data-count="${x.num}">0</div><div class="stat-lbl">${x.lbl}</div></div>`).join("")
  }</div>`;

  const areasHtml = s.areas.length
    ? s.areas.map((a) => `
      <div class="bar-row">
        <span class="lbl">${esc(areaName(a.area))}</span>
        <span class="pct">${Math.round(a.mastery * 100)}%</span>
        <div class="bar ${masteryColor(a.mastery)}"><i data-w="${Math.round(a.mastery * 100)}"></i></div>
      </div>`).join("")
    : `<p class="view-subtitle">No area data yet.</p>`;

  const upNext = s.upNext;
  const upNextHtml = upNext ? `
    <div class="tile">
      <div class="tile-ico">🚀</div>
      <div class="tile-body">
        <div class="tile-kicker">Up next</div>
        <div class="tile-title">${esc(topicTitle(upNext.topic_id))}</div>
        <div class="tile-sub">${upNext.mastery == null ? "New topic — start fresh" : Math.round(upNext.mastery * 100) + "% mastery so far"}</div>
      </div>
      <button class="btn btn-primary btn-sm" data-seed="${esc(upNext.topic_id)}">Continue →</button>
    </div>` : `<p class="view-subtitle">Pick any topic in Path to begin.</p>`;

  let weakHtml = "";
  if (s.weakTopic) {
    weakHtml = `
      <div class="tile">
        <div class="tile-ico ember">🎯</div>
        <div class="tile-body">
          <div class="tile-kicker">Weak spot</div>
          <div class="tile-title">${esc(topicTitle(s.weakTopic.topic_id))}</div>
          <div class="tile-sub">${Math.round(s.weakTopic.avg_mastery * 100)}% — worth shoring up</div>
        </div>
        <button class="btn btn-ghost btn-sm" data-seed="${esc(s.weakTopic.topic_id)}">Fix this</button>
      </div>`;
  } else if (s.misc.length) {
    weakHtml = `
      <div class="tile">
        <div class="tile-ico ember">⚠️</div>
        <div class="tile-body">
          <div class="tile-kicker">Active misconception</div>
          <div class="tile-title" style="font-size:14px;font-weight:500">${esc(s.misc[0].description)}</div>
        </div>
      </div>`;
  } else {
    weakHtml = `<p class="view-subtitle">No weak spots flagged. Nice.</p>`;
  }

  const greeting = streak > 0
    ? `Welcome back — you're on a <b>${streak}-day</b> streak.`
    : `Welcome to Memex.`;

  el.innerHTML = `
    <div class="view-inner">
      <div class="view-head">
        <div class="view-eyebrow">Today</div>
        <h1 class="view-title">Hey ${esc(state.studentId)} 👋</h1>
        <p class="view-subtitle">${greeting}</p>
      </div>

      ${homeStatsHtml}

      <div class="grid-cards grid-2">
        <div class="card">
          <div class="ring-wrap">
            <div class="ring" id="homeRing"><span class="ring-val"><span id="homeRingVal">0</span><small>%</small></span></div>
            <div>
              <div class="tile-kicker">Overall mastery</div>
              <div style="font-family:'Space Grotesk';font-size:20px;font-weight:600;margin:2px 0 8px">${s.hasData ? overallPct + "% there" : "Just getting started"}</div>
              ${areasHtml}
            </div>
          </div>
        </div>

        <div class="card grid-cards" style="gap:18px">
          ${upNextHtml}
          <div style="height:1px;background:var(--border)"></div>
          ${weakHtml}
        </div>
      </div>

      <div class="card card-grad" style="margin-top:16px;display:flex;align-items:center;gap:16px;flex-wrap:wrap">
        <div style="flex:1;min-width:200px">
          <div class="tile-kicker">Ready to learn?</div>
          <div style="font-family:'Space Grotesk';font-size:18px;font-weight:600">Jump back into the tutor</div>
        </div>
        <button class="btn btn-primary" id="homeContinue">Continue learning →</button>
        <button class="btn btn-ghost" id="homeQuiz">🎯 Quiz me</button>
      </div>
    </div>`;

  // Animate ring + bars
  requestAnimationFrame(() => {
    const ring = $("homeRing");
    if (ring) ring.style.setProperty("--val", overallPct);
    countUp($("homeRingVal"), overallPct);
    el.querySelectorAll(".stat-num[data-count]").forEach((n) => countUp(n, parseInt(n.dataset.count, 10)));
    el.querySelectorAll(".bar > i[data-w]").forEach((i) => { i.style.width = i.dataset.w + "%"; });
  });

  // Wire actions
  el.querySelectorAll("[data-seed]").forEach((b) =>
    b.addEventListener("click", () => seedChat(b.dataset.seed)));
  $("homeContinue")?.addEventListener("click", () => {
    if (upNext) seedChat(upNext.topic_id); else showView("chat");
  });
  $("homeQuiz")?.addEventListener("click", () => {
    if (upNext) { seedChat(upNext.topic_id); } else showView("chat");
  });
}

// Switch to chat focused on a topic.
function seedChat(topicId) {
  if (topicId && TOPIC_MAP[topicId]) {
    $("topicSelect").value = topicId;
    state.topicId = topicId;
  }
  startNewChatLocal();
  showView("chat");
  if (topicId && TOPIC_MAP[topicId]) {
    $("msgInput").value = `Help me learn ${topicTitle(topicId)}.`;
    $("msgInput").dispatchEvent(new Event("input"));
  }
}

// ── Profile view ───────────────────────────────────────────────
function fmtDate(iso) {
  if (!iso) return "—";
  try { return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }); }
  catch (_) { return "—"; }
}

async function renderProfile() {
  const el = $("view-profile");
  const [act, prog] = await Promise.all([loadActivity(true), loadProgress(true)]);
  const st = (act && act.stats) || {};
  const initial = (state.studentId || "?").trim().charAt(0).toUpperCase() || "?";
  const streak = currentStreak();
  const avgQuiz = st.avg_quiz_score == null ? "—" : Math.round(st.avg_quiz_score * 100) + "%";

  const stats = [
    { num: st.questions ?? 0, lbl: "Questions asked", grad: true },
    { num: st.quizzes ?? 0, lbl: "Quizzes taken" },
    { num: avgQuiz, lbl: "Avg quiz score" },
    { num: `${st.concepts_mastered ?? 0}/${st.concepts_assessed ?? 0}`, lbl: "Concepts mastered" },
    { num: st.topics_touched ?? 0, lbl: "Topics explored" },
    { num: streak, lbl: "Day streak" },
  ];
  const statHtml = stats.map((s) => {
    const isNum = typeof s.num === "number";
    return `<div class="stat"><div class="stat-num ${s.grad ? "grad" : ""}" ${isNum ? `data-count="${s.num}"` : ""}>${isNum ? 0 : s.num}</div><div class="stat-lbl">${s.lbl}</div></div>`;
  }).join("");

  // Quiz-score trend
  const qh = (act && act.quiz_history) || [];
  let sparkHtml = "";
  if (qh.length) {
    sparkHtml = `<div class="section-label">Quiz-score history</div><div class="card"><div class="spark">${
      qh.slice(-12).map((q) => {
        const pct = Math.round((q.score || 0) * 100);
        const cls = q.score >= 0.7 ? "good" : q.score >= 0.4 ? "warn" : "bad";
        const t = topicTitle(q.topic_id || "");
        return `<div class="spark-col"><span class="spark-cap">${pct}%</span><div class="spark-bar ${cls}" data-h="${Math.max(3, pct)}"></div><span class="spark-x" title="${esc(t)}">${esc((t.split(" ")[0]) || "")}</span></div>`;
      }).join("")
    }</div></div>`;
  }

  // Strengths / needs-work from mastery
  const topics = (prog && prog.topics) || [];
  const sorted = topics.slice().sort((a, b) => b.avg_mastery - a.avg_mastery);
  const strong = sorted.slice(0, 3);
  const weak = sorted.slice().reverse().slice(0, 3);
  const misc = (prog && prog.misconceptions) || [];
  const barRow = (t, forceGood) =>
    `<div class="bar-row"><span class="lbl">${esc(topicTitle(t.topic_id))}</span><span class="pct">${Math.round(t.avg_mastery * 100)}%</span><div class="bar ${forceGood ? "good" : masteryColor(t.avg_mastery)}"><i data-w="${Math.round(t.avg_mastery * 100)}"></i></div></div>`;
  const strengthHtml = strong.length ? strong.map((t) => barRow(t, true)).join("") : `<p class="view-subtitle">No data yet.</p>`;
  const weakHtml = weak.length ? weak.map((t) => barRow(t, false)).join("") : "";
  const miscHtml = misc.length ? `<div style="margin-top:10px">${misc.slice(0, 3).map((m) => `<div style="display:flex;gap:8px;font-size:12.5px;color:var(--text-muted);margin:6px 0"><span style="color:var(--warn)">⚠</span><span>${esc(m.description)}</span></div>`).join("")}</div>` : "";

  // Activity timeline
  const tl = (act && act.timeline) || [];
  const tlIco = { question: ["q", "💬"], quiz_attempt: ["quiz", "🎯"], feedback: ["fb", "👍"] };
  const tlHtml = tl.length ? tl.slice(0, 25).map((e) => {
    const [cls, ico] = tlIco[e.type] || ["", "•"];
    const score = (e.type === "quiz_attempt" && e.score != null)
      ? `<span class="tl-score" style="color:${e.score >= 0.7 ? "var(--good)" : e.score >= 0.4 ? "var(--warn)" : "var(--bad)"}">${Math.round(e.score * 100)}%</span>` : "";
    const topic = e.topic_id ? topicTitle(e.topic_id) : "";
    return `<div class="tl-item"><div class="tl-ico ${cls}">${ico}</div><div class="tl-body"><div class="tl-label">${esc(e.label)}</div><div class="tl-meta">${topic ? esc(topic) + " · " : ""}${formatRelative(e.occurred_at)}</div></div>${score}</div>`;
  }).join("") : `<p class="view-subtitle">No activity recorded yet.</p>`;

  el.innerHTML = `
    <div class="view-inner">
      <div class="card profile-hero" style="margin-bottom:20px">
        <div class="avatar">${esc(initial)}</div>
        <div>
          <h1 class="view-title" style="font-size:24px">${esc(state.studentId)}</h1>
          <div class="profile-meta">
            <span>🔥 <b>${streak}</b>-day streak</span>
            <span>📅 Learning since <b>${fmtDate(st.first_active)}</b></span>
            <span>⏱️ <b>${st.active_days ?? 0}</b> active days</span>
            <span>Last active <b>${formatRelative(st.last_active)}</b></span>
          </div>
        </div>
      </div>

      <div class="stat-grid">${statHtml}</div>
      ${sparkHtml}

      <div class="grid-cards grid-2" style="margin-top:8px">
        <div class="card"><div class="tile-kicker" style="margin-bottom:10px">💪 Strengths</div>${strengthHtml}</div>
        <div class="card"><div class="tile-kicker" style="margin-bottom:10px">🎯 Needs work</div>${weakHtml}${miscHtml}</div>
      </div>

      <div class="section-label">Recent activity</div>
      <div class="card"><div class="timeline">${tlHtml}</div></div>
    </div>`;

  requestAnimationFrame(() => {
    el.querySelectorAll(".stat-num[data-count]").forEach((n) => countUp(n, parseInt(n.dataset.count, 10)));
    el.querySelectorAll(".bar > i[data-w]").forEach((i) => { i.style.width = i.dataset.w + "%"; });
    el.querySelectorAll(".spark-bar[data-h]").forEach((b) => { b.style.height = b.dataset.h + "%"; });
  });
}

// ── Progress view ──────────────────────────────────────────────
async function renderProgress() {
  const el = $("view-progress");
  const data = await loadProgress(true);
  const topics = (data && data.topics) || [];
  const misc = (data && data.misconceptions) || [];

  if (!topics.length) {
    el.innerHTML = `<div class="view-inner"><div class="empty-state">
      <div class="em-ico">📊</div><div class="em-title">No mastery data yet</div>
      <p>Start your first chat or take a quiz and your progress will appear here.</p>
      <button class="btn btn-primary" style="margin-top:14px" onclick="location.hash='#/chat'">Start a chat →</button>
    </div></div>`;
    return;
  }

  const sorted = topics.slice().sort((a, b) => b.avg_mastery - a.avg_mastery);
  const topicsHtml = sorted.map((t) => {
    const pct = Math.round(t.avg_mastery * 100);
    const concepts = (t.concepts || []).slice(0, 4).map((c) =>
      `<span class="node-tag" style="display:inline-block;margin-right:10px">• ${esc(c.concept_title)} ${Math.round(c.score * 100)}%</span>`).join("");
    return `
      <div class="card" style="padding:16px 18px">
        <div class="bar-row" style="margin:0 0 6px">
          <span class="lbl">${esc(topicTitle(t.topic_id))} <span class="chip chip-area" style="margin-left:6px">${esc(areaName(TOPIC_MAP[t.topic_id]?.area))}</span></span>
          <span class="pct">${pct}%</span>
          <div class="bar ${masteryColor(t.avg_mastery)}"><i data-w="${pct}"></i></div>
        </div>
        <div style="margin-top:8px">${concepts || ""}</div>
      </div>`;
  }).join("");

  const miscHtml = misc.length ? `
    <div class="card" style="margin-top:8px">
      <div class="tile-kicker" style="margin-bottom:10px">Active misconceptions (${misc.length})</div>
      <ul style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:8px">
        ${misc.slice(0, 8).map((m) => `<li style="display:flex;gap:8px;font-size:13px;color:var(--text-muted)"><span style="color:var(--warn)">⚠</span><span>${esc(m.description)}</span></li>`).join("")}
      </ul>
    </div>` : "";

  el.innerHTML = `
    <div class="view-inner">
      <div class="view-head">
        <div class="view-eyebrow">Progress</div>
        <h1 class="view-title">Your mastery</h1>
        <p class="view-subtitle">Confidence-weighted, updated as you chat and quiz.</p>
      </div>
      <div class="grid-cards">${topicsHtml}</div>
      ${miscHtml}
    </div>`;

  requestAnimationFrame(() => {
    el.querySelectorAll(".bar > i[data-w]").forEach((i) => { i.style.width = i.dataset.w + "%"; });
  });
}

// ── Path view ──────────────────────────────────────────────────
async function renderPath() {
  const el = $("view-path");
  const data = await loadProgress(true);
  const masteryById = {};
  ((data && data.topics) || []).forEach((t) => { masteryById[t.topic_id] = t.avg_mastery; });

  // Group catalog topics by area
  const byArea = {};
  Object.entries(TOPIC_MAP).forEach(([id, meta]) => { (byArea[meta.area] ||= []).push({ id, ...meta }); });
  const areas = Object.keys(byArea).sort();

  if (!areas.length) {
    el.innerHTML = `<div class="view-inner"><div class="empty-state"><div class="em-ico">🎯</div><div class="em-title">No topics loaded</div></div></div>`;
    return;
  }

  const areasHtml = areas.map((area) => {
    const nodes = byArea[area].map((t) => {
      const m = masteryById[t.id];
      const pct = m == null ? 0 : Math.round(m * 100);
      const cls = m == null ? "" : m >= 0.7 ? "mastered" : "inprogress";
      const stateIco = m == null ? "○" : m >= 0.7 ? "✅" : "◐";
      const tag = m == null ? "Not started" : m >= 0.7 ? "Mastered" : `In progress · ${pct}%`;
      return `
        <button class="node ${cls}" data-seed="${esc(t.id)}">
          <span class="node-state">${stateIco}</span>
          <div class="node-title">${esc(t.title)}</div>
          <div class="bar node-bar ${m == null ? "" : masteryColor(m)}"><i data-w="${pct}"></i></div>
          <div class="node-tag">${tag}</div>
        </button>`;
    }).join("");
    return `
      <div class="path-area">
        <div class="path-area-h"><span class="chip chip-area">${esc(area === "?" ? "Other" : "Area " + area)}</span><h3>${esc(areaName(area))} · ${byArea[area].length}</h3></div>
        <div class="path-grid">${nodes}</div>
      </div>`;
  }).join("");

  el.innerHTML = `
    <div class="view-inner">
      <div class="view-head">
        <div class="view-eyebrow">Path</div>
        <h1 class="view-title">Curriculum map</h1>
        <p class="view-subtitle">Click any topic to start a focused session. ✅ mastered · ◐ in progress · ○ not started.</p>
      </div>
      ${areasHtml}
    </div>`;

  requestAnimationFrame(() => {
    el.querySelectorAll(".bar > i[data-w]").forEach((i) => { i.style.width = i.dataset.w + "%"; });
  });
  el.querySelectorAll("[data-seed]").forEach((b) => b.addEventListener("click", () => seedChat(b.dataset.seed)));
}

// ── Conversation list ──────────────────────────────────────────
function formatRelative(iso) {
  if (!iso) return "";
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}
async function loadConversations() {
  try {
    const r = await fetch(`/api/student/${encodeURIComponent(state.studentId)}/conversations`);
    if (!r.ok) return;
    state.conversations = (await r.json()).conversations || [];
    renderConversationList();
  } catch (e) { console.warn(e); }
}
function renderConversationList() {
  const el = $("conversationList");
  if (!el) return;
  el.innerHTML = "";
  if (!state.conversations.length) {
    el.innerHTML = `<div style="font-size:12px;color:var(--text-faint);padding:8px 12px">No conversations yet.</div>`;
    return;
  }
  for (const c of state.conversations) {
    const item = document.createElement("button");
    item.className = "conv-item" + (c.id === state.conversationId ? " active" : "");
    item.textContent = c.title || "New chat";
    item.title = `${c.title} — ${formatRelative(c.last_message_at)}`;
    item.addEventListener("click", () => { loadConversation(c.id); $("historyMenu").removeAttribute("open"); });
    el.appendChild(item);
  }
}
async function loadConversation(conversationId) {
  state.conversationId = conversationId;
  state.messages = []; state.reuseCounts = {};
  $("chat").innerHTML = "";
  $("welcome").classList.add("hidden");
  showView("chat");
  try {
    const r = await fetch(`/api/conversations/${conversationId}/messages`);
    if (!r.ok) return;
    const data = await r.json();
    for (const m of data.messages) {
      const idx = state.messages.length;
      state.messages.push({ role: m.role, content: m.content, references: [], restored: true });
      if (m.role === "user") appendUserBubble(m.content);
      else appendAssistantMessage(state.messages[idx], idx);
    }
    renderConversationList();
    scrollBottom();
  } catch (e) { console.warn(e); }
}
function startNewChatLocal() {
  state.conversationId = null;
  state.messages = []; state.reuseCounts = {};
  $("chat").innerHTML = "";
  $("welcome").classList.remove("hidden");
  renderConversationList();
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
    btn.className = "starter-card";
    btn.innerHTML = `<div class="starter-ico">${icon}</div>
      <div class="starter-title">${esc(title)}</div>
      <div class="starter-sub">${esc(sub)}</div>`;
    btn.addEventListener("click", () => sendMessage(title));
    grid.appendChild(btn);
  });
}

// ── Message rendering ──────────────────────────────────────────
function thinkingEl() {
  const d = document.createElement("div");
  d.id = "thinking";
  d.className = "msg-row assistant";
  d.innerHTML = `<div class="thinking-dots" style="height:20px"><span></span><span></span><span></span></div>`;
  return d;
}
function appendUserBubble(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `<div class="bubble-user">${esc(text)}</div>`;
  $("chat").appendChild(row);
}
function appendAssistantMessage(msg, msgIdx) {
  const row = document.createElement("div");
  row.className = "msg-row assistant group";
  row.dataset.msgIdx = msgIdx;

  const citedText = substituteCitations(msg.content, msg.references);
  const bodyHtml = renderMarkdown(citedText);

  let refsHtml = "";
  if (msg.references && msg.references.length) {
    const items = msg.references.map((r) =>
      `<div class="ref-line"><b>[${r.n}]</b> ${esc(r.title)}</div>`).join("");
    refsHtml = `<details class="refs" style="margin-top:12px"><summary>📚 ${msg.references.length} source${msg.references.length > 1 ? "s" : ""} ▾</summary>
      <div style="margin-top:6px;display:flex;flex-direction:column;gap:3px;padding-left:4px">${items}
        <button class="ref-ctx link-btn" style="margin-top:6px;text-align:left">See why these were chosen →</button>
      </div></details>`;
  }

  row.innerHTML = `
    <div class="prose-chat">${bodyHtml}</div>
    ${refsHtml}
    <div class="msg-footer">
      <button class="msg-act btn-test" title="Test yourself">🎯 Test</button>
      <button class="msg-act btn-ctx" title="View context">🔍 Context</button>
      <button class="msg-act btn-regen" title="Regenerate">🔁 Regenerate</button>
      <button class="msg-act btn-copy" title="Copy">📋 Copy</button>
      <span style="flex:1"></span>
      <button class="msg-act up btn-thumbsup" title="Helpful">👍</button>
      <button class="msg-act down btn-thumbsdown" title="Not helpful">👎</button>
    </div>
    <div class="quiz-area" style="margin-top:8px"></div>`;

  row.querySelector(".btn-test").addEventListener("click", () => triggerQuiz(row, msgIdx));
  row.querySelector(".btn-ctx").addEventListener("click", () => openRoutingModal(msg));
  row.querySelector(".ref-ctx")?.addEventListener("click", () => openRoutingModal(msg));
  row.querySelector(".btn-regen").addEventListener("click", () => regenerate(msgIdx));
  row.querySelector(".btn-copy").addEventListener("click", () => copyReply(msg.content));

  const selectedIds = (msg.selected || []).map((it) => it.id);
  row.querySelector(".btn-thumbsup").addEventListener("click", function () {
    sendFeedback(msgIdx, 1, selectedIds); this.textContent = "👍✓"; this.disabled = true;
    row.querySelector(".btn-thumbsdown").disabled = true;
  });
  row.querySelector(".btn-thumbsdown").addEventListener("click", function () {
    sendFeedback(msgIdx, -1, selectedIds); this.textContent = "👎✓"; this.disabled = true;
    row.querySelector(".btn-thumbsup").disabled = true;
  });

  $("chat").appendChild(row);
  runMermaid(row);
  return row;
}
function showThinking() { const el = thinkingEl(); $("chat").appendChild(el); scrollBottom(); return el; }

// ── Send message ───────────────────────────────────────────────
async function sendMessage(text) {
  text = (text ?? $("msgInput").value).trim();
  if (!text) return;
  showView("chat");
  $("welcome").classList.add("hidden");

  state.studentId = $("studentIdInput").value.trim() || "Hiva";
  state.topicId = $("topicSelect").value || null;

  state.messages.push({ role: "user", content: text });
  appendUserBubble(text);

  $("msgInput").value = ""; $("msgInput").style.height = "auto"; $("sendBtn").disabled = true;
  const thinking = showThinking();
  setInputLocked(true);

  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        student_id: state.studentId, conversation_id: state.conversationId,
        topic_id: state.topicId, question: text, budget: state.budget, reuse_counts: state.reuseCounts,
      }),
    });
    thinking.remove();
    if (!res.ok) { appendError(await res.text()); return; }
    const data = await res.json();
    if (data.conversation_id) state.conversationId = data.conversation_id;
    data.selected?.forEach((it) => { state.reuseCounts[it.id] = (state.reuseCounts[it.id] || 0) + 1; });

    const msgIdx = state.messages.length;
    const aMsg = { role: "assistant", content: data.reply, references: data.references, selected: data.selected, dropped: data.dropped, tokens_used: data.tokens_used };
    state.messages.push(aMsg);
    appendAssistantMessage(aMsg, msgIdx);
    scrollBottom();

    bumpStreak();
    loadConversations();
    loadProgress(true);
    loadActivity(true);
  } catch (err) {
    thinking.remove();
    appendError(err.message);
  } finally {
    setInputLocked(false); $("msgInput").focus();
  }
}
function setInputLocked(locked) { $("msgInput").disabled = locked; }
function appendError(text) {
  const d = document.createElement("div");
  d.className = "err-box";
  d.textContent = "Error: " + text;
  $("chat").appendChild(d); scrollBottom();
}

async function regenerate(msgIdx) {
  let userText = null;
  for (let i = msgIdx - 1; i >= 0; i--) {
    if (state.messages[i].role === "user") { userText = state.messages[i].content; break; }
  }
  if (!userText) return;
  $("chat").querySelector(`[data-msg-idx="${msgIdx}"]`)?.remove();
  state.messages.splice(msgIdx, 1);
  await sendMessage(userText);
}
async function copyReply(text) {
  try { await navigator.clipboard.writeText(text); } catch (_) { prompt("Copy:", text); }
}

// ── Routing modal ──────────────────────────────────────────────
function openRoutingModal(msg) {
  const body = $("routingBody");
  const sel = msg.selected || [], dropped = msg.dropped || [];

  let html = `<p class="view-subtitle" style="margin:0 0 14px">The tutor picked these memory items to answer you, scored by relevance, recency, your misconceptions, prerequisites, and reuse.</p>`;

  if (sel.length) {
    html += `<p class="tile-kicker" style="margin-bottom:8px">Selected (${sel.length})</p>`;
    sel.forEach((it) => { html += routingItemHtml(it); });
  }
  if (dropped.length) {
    html += `<details style="margin-top:14px"><summary class="tile-kicker" style="cursor:pointer">Dropped top (${dropped.length}) ▾</summary>
      <div style="margin-top:10px">${dropped.slice(0, 5).map((it) => routingItemHtml(it)).join("")}</div></details>`;
  }
  body.innerHTML = html;
  $("routingModal").classList.remove("hidden");
}
function routingItemHtml(it) {
  const scores = [
    ["relevance", it.score_relevance], ["recency", it.score_recency], ["misc.", it.score_misconception],
    ["prereq", it.score_prerequisite], ["reuse", it.score_reuse], ["total", it.score_total],
  ].filter(([, v]) => v != null);
  const bars = scores.map(([label, val]) => {
    const p = Math.round((val || 0) * 100);
    return `<div class="routing-score"><span class="rs-lbl">${label}</span><div class="bar"><i style="width:${p}%"></i></div><span class="rs-val">${p}%</span></div>`;
  }).join("");
  return `<div class="routing-item">
    <div style="font-size:14px;font-weight:600;margin-bottom:6px">${esc(it.title)}</div>${bars}
    ${it.body ? `<p style="margin-top:8px;font-size:12px;color:var(--text-muted)">${esc(it.body.slice(0, 200))}</p>` : ""}
  </div>`;
}
function closeRoutingModal() { $("routingModal").classList.add("hidden"); }

// ── Quiz ───────────────────────────────────────────────────────
async function triggerQuiz(row, msgIdx) {
  const topicId = state.topicId || $("topicSelect").value || null;
  if (!topicId) { alert("Select a topic first to generate a quiz."); return; }
  const area = row.querySelector(".quiz-area");
  area.innerHTML = `<div class="quiz-card"><div style="display:flex;align-items:center;gap:8px;font-size:12px;color:var(--text-muted)"><div class="thinking-dots"><span></span><span></span><span></span></div> Generating question…</div></div>`;
  try {
    const res = await fetch("/api/quiz/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ topic_id: topicId, student_id: state.studentId }) });
    if (!res.ok) throw new Error(await res.text());
    const { question, rubric, difficulty } = await res.json();
    renderQuizQuestion(area, msgIdx, question, rubric, topicId, difficulty);
  } catch (err) { area.innerHTML = `<p style="font-size:12px;color:var(--bad);margin-top:4px">Quiz error: ${esc(err.message)}</p>`; }
}
const DIFF_STYLE = {
  Easy: "background:var(--ember-soft);color:var(--good)",
  Medium: "background:var(--accent-soft);color:var(--accent)",
  Hard: "background:var(--ember-soft);color:var(--ember)",
  Expert: "background:color-mix(in srgb,var(--bad) 14%,transparent);color:var(--bad)",
};
function renderQuizQuestion(area, msgIdx, question, rubric, topicId, difficulty) {
  const card = document.createElement("div");
  card.className = "quiz-card";
  const diffBadge = difficulty
    ? `<span class="chip" style="${DIFF_STYLE[difficulty] || DIFF_STYLE.Easy}">${esc(difficulty)}</span>` : "";
  card.innerHTML = `<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
    <span style="font-size:14px;font-weight:600">🎯 Quick check ${diffBadge}</span>
    <button class="modal-close dismiss-quiz" style="font-size:14px">&#x2715;</button></div>
    <p style="font-size:14px;margin-bottom:12px">${esc(question)}</p>
    <textarea rows="3" class="quiz-textarea input-area" placeholder="Your answer…"></textarea>
    <div style="display:flex;justify-content:flex-end;margin-top:10px"><button class="btn btn-primary btn-sm quiz-submit">Submit</button></div>
    <div class="quiz-result" style="margin-top:10px"></div>`;
  card.querySelector(".dismiss-quiz").addEventListener("click", () => { area.innerHTML = ""; });
  card.querySelector(".quiz-submit").addEventListener("click", () => submitQuiz(card, msgIdx, question, rubric, topicId));
  area.innerHTML = ""; area.appendChild(card);
}
async function submitQuiz(card, msgIdx, question, rubric, topicId) {
  const answer = card.querySelector(".quiz-textarea").value.trim();
  if (!answer) { card.querySelector(".quiz-textarea").focus(); return; }
  const result = card.querySelector(".quiz-result");
  result.innerHTML = `<div style="display:flex;align-items:center;gap:8px;font-size:12px;color:var(--text-muted)"><div class="thinking-dots"><span></span><span></span><span></span></div> Grading…</div>`;
  try {
    const res = await fetch("/api/quiz/score", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ student_id: state.studentId, topic_id: topicId, question, rubric, answer }) });
    if (!res.ok) throw new Error(await res.text());
    const { score, rationale } = await res.json();
    const pct = Math.round(score * 100);
    const col = score >= 0.8 ? "var(--good)" : score >= 0.6 ? "var(--warn)" : "var(--bad)";
    result.innerHTML = `<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
      <span style="font-size:22px;font-weight:700;color:${col}">${pct}%</span>
      <span style="font-size:12px;color:var(--text-muted)">${score >= 0.8 ? "Great work!" : score >= 0.6 ? "Partial — keep going." : "Needs more work."}</span></div>
      <p style="font-size:12px;color:var(--text-muted);font-style:italic">${esc(rationale)}</p>`;
    loadProgress(true);
    if (score < 0.6) {
      const diagBtn = document.createElement("button");
      diagBtn.className = "btn btn-ghost btn-sm";
      diagBtn.style.marginTop = "10px";
      diagBtn.textContent = "🧭 Understand what went wrong";
      diagBtn.addEventListener("click", () => startDiagnostic(card, question, rubric, answer, score));
      result.appendChild(diagBtn);
    }
  } catch (err) { result.innerHTML = `<p style="font-size:12px;color:var(--bad)">Error: ${esc(err.message)}</p>`; }
}

// ── Diagnostic ─────────────────────────────────────────────────
async function startDiagnostic(quizCard, question, rubric, studentAnswer, score) {
  const area = quizCard.closest(".quiz-area");
  if (!area) return;
  const diagCard = document.createElement("div");
  diagCard.className = "diagnostic-card";
  diagCard.innerHTML = `<div style="display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ember)"><div class="thinking-dots"><span></span><span></span><span></span></div> Analyzing…</div>`;
  area.appendChild(diagCard); scrollBottom();
  try {
    const res = await fetch("/api/diagnostic/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ original_question: question, rubric, student_answer: studentAnswer, score }) });
    if (!res.ok) throw new Error(await res.text());
    renderDiagFollowUp(diagCard, await res.json(), question, 1);
  } catch (err) { diagCard.innerHTML = `<p style="font-size:12px;color:var(--bad)">Error: ${esc(err.message)}</p>`; }
}
function renderDiagFollowUp(card, data, originalQuestion, turnIndex) {
  card.innerHTML = `<div style="margin-bottom:8px">
    <span style="font-size:14px;font-weight:600;color:var(--ember)">🧭 Let's figure this out</span>
    <p style="font-size:12px;color:var(--text-muted);font-style:italic;margin-top:2px">${esc(data.diagnosis)}</p></div>
    <p style="font-size:14px;font-weight:500;margin-bottom:12px">${esc(data.follow_up_question)}</p>
    <textarea rows="2" class="diag-textarea input-area" placeholder="Your response…"></textarea>
    <div style="display:flex;justify-content:flex-end;margin-top:10px"><button class="btn btn-primary btn-sm diag-submit">Continue</button></div>
    <div class="diag-result" style="margin-top:10px"></div>`;
  card.querySelector(".diag-submit").addEventListener("click", () => continueDiagnostic(card, originalQuestion, data, turnIndex));
}
async function continueDiagnostic(card, originalQuestion, diagData, turnIndex) {
  const answer = card.querySelector(".diag-textarea")?.value.trim();
  if (!answer) { card.querySelector(".diag-textarea")?.focus(); return; }
  const result = card.querySelector(".diag-result");
  result.innerHTML = `<div style="display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ember)"><div class="thinking-dots"><span></span><span></span><span></span></div> Thinking…</div>`;
  try {
    const res = await fetch("/api/diagnostic/turn", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ student_id: state.studentId, original_question: originalQuestion, diagnosis: diagData.diagnosis, follow_up_question: diagData.follow_up_question, student_answer: answer, turn_index: turnIndex }) });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    result.innerHTML = `<div class="prose-chat" style="margin-bottom:12px">${renderMarkdown(data.next_message || data.explanation || "")}</div>`;
    if (data.next_action && data.next_action !== "wrap_up") {
      const nextBtn = document.createElement("button");
      nextBtn.className = "btn btn-primary btn-sm";
      nextBtn.textContent = "Next →";
      nextBtn.addEventListener("click", () => renderDiagFollowUp(card, { diagnosis: diagData.diagnosis, follow_up_question: data.next_message || "" }, originalQuestion, turnIndex + 1));
      result.appendChild(nextBtn);
    } else {
      const done = document.createElement("p");
      done.style.cssText = "font-size:12px;color:var(--good);font-weight:600;margin-top:6px";
      done.textContent = "✅ Great — misconception recorded.";
      result.appendChild(done);
      loadProgress(true);
    }
  } catch (err) { result.innerHTML = `<p style="font-size:12px;color:var(--bad)">Error: ${esc(err.message)}</p>`; }
}

// ── Feedback ───────────────────────────────────────────────────
async function sendFeedback(msgIdx, rating, selectedItemIds) {
  try {
    await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ student_id: state.studentId, message_idx: msgIdx, rating, selected_item_ids: selectedItemIds || [] }) });
    loadProgress(true);
  } catch (_) {}
}

// ── Welcome-back ───────────────────────────────────────────────
async function maybeShowWelcomeBack() {
  const data = await loadProgress();
  if (!data || !data.topics || !data.topics.length) return;
  const top3 = data.topics.slice(0, 3).map((t) => topicTitle(t.topic_id)).join(", ");
  $("welcomeBackText").textContent = `Welcome back! We've worked on: ${top3}.`;
  $("welcomeBackBanner").classList.remove("hidden");
}

// ── Events ─────────────────────────────────────────────────────
document.querySelectorAll(".nav-item[data-view]").forEach((b) =>
  b.addEventListener("click", () => showView(b.dataset.view)));

$("themeToggle").addEventListener("click", toggleTheme);

$("msgInput").addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 200) + "px";
  $("sendBtn").disabled = !this.value.trim();
});
$("msgInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
$("sendBtn").addEventListener("click", () => sendMessage());

$("newChatBtn").addEventListener("click", () => { startNewChatLocal(); $("historyMenu").removeAttribute("open"); showView("chat"); });
$("newChatTopBtn").addEventListener("click", () => { startNewChatLocal(); showView("chat"); });

$("studentIdInput").addEventListener("change", (e) => {
  state.studentId = e.target.value.trim() || "you";
  state.reuseCounts = {}; state.messages = []; state.conversationId = null;
  _progressCache = null; _activityCache = null;
  $("chat").innerHTML = ""; $("welcome").classList.remove("hidden");
  loadConversations();
  if (state.view !== "chat") showView(state.view, false);
});
$("topicSelect").addEventListener("change", (e) => { state.topicId = e.target.value || null; });

$("routingClose").addEventListener("click", closeRoutingModal);
$("routingBackdrop").addEventListener("click", closeRoutingModal);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeRoutingModal(); });

$("welcomeBackOpenMemory").addEventListener("click", () => showView("progress"));

window.addEventListener("hashchange", routeFromHash);

// ── Init ───────────────────────────────────────────────────────
(async () => {
  applyTheme(document.documentElement.getAttribute("data-theme") || "light");
  renderStreak();
  buildStarterGrid();
  await loadTopics();
  await loadConversations();
  await maybeShowWelcomeBack();
  routeFromHash();   // default #/home
})();
