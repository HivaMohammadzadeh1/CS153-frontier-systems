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
  router: "",        // "" = heuristic; else a finetuned size id
};

const TOPIC_MAP = {};   // id -> { title, area }
const AREA_NAMES = {};  // area letter -> human-readable name
let _progressCache = null;

function areaName(letter) {
  const n = AREA_NAMES[letter];
  if (n) return n.replace(/\s*\(.*?\)\s*$/, "").trim();   // drop course-code suffix
  return letter && letter !== "?" ? `Area ${letter}` : "Other topics";
}

// Curriculum TEACHING order (not alphabetical): fundamentals → inference serving →
// training systems → production ops → data/alignment → agents → system design.
// Inference-first on purpose — the ML-infra interview bar lives in serving.
const AREA_ORDER = ["A", "C", "B", "F", "D", "E", "G"];
function sortAreaKey(a, b) {
  if (a === "?") return 1;
  if (b === "?") return -1;
  const ia = AREA_ORDER.indexOf(a), ib = AREA_ORDER.indexOf(b);
  if (ia !== -1 && ib !== -1) return ia - ib;
  if (ia !== -1) return -1;
  if (ib !== -1) return 1;
  return String(a).localeCompare(String(b));
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

// ── Confetti 🎉 ─────────────────────────────────────────────────
function confetti(opts = {}) {
  const canvas = $("confettiCanvas");
  if (!canvas) return;
  try { if (matchMedia("(prefers-reduced-motion: reduce)").matches) return; } catch (_) {}
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = window.innerWidth, H = window.innerHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + "px"; canvas.style.height = H + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const colors = ["#e0922f", "#cf6630", "#cd8a1e", "#a23d1f", "#2f7d4f", "#1b1a17", "#d8a24a"];
  const count = opts.count || 90;
  const cx = opts.x != null ? opts.x : W / 2;
  const cy = opts.y != null ? opts.y : H * 0.32;
  const parts = [];
  for (let i = 0; i < count; i++) {
    const a = Math.random() * Math.PI * 2, sp = 4 + Math.random() * 9;
    parts.push({
      x: cx, y: cy, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 4,
      g: 0.16 + Math.random() * 0.12, size: 5 + Math.random() * 7,
      rot: Math.random() * Math.PI, vr: (Math.random() - .5) * .3,
      color: colors[(Math.random() * colors.length) | 0], life: 0, ttl: 90 + Math.random() * 40,
    });
  }
  let frame = 0;
  (function tick() {
    ctx.clearRect(0, 0, W, H);
    let alive = false;
    for (const p of parts) {
      if (p.life > p.ttl) continue;
      alive = true;
      p.life++; p.vy += p.g; p.x += p.vx; p.y += p.vy; p.rot += p.vr; p.vx *= .99;
      ctx.save(); ctx.globalAlpha = Math.max(0, 1 - p.life / p.ttl);
      ctx.translate(p.x, p.y); ctx.rotate(p.rot); ctx.fillStyle = p.color;
      ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6); ctx.restore();
    }
    if (alive && ++frame < 240) requestAnimationFrame(tick);
    else ctx.clearRect(0, 0, W, H);
  })();
}

// ── XP / Levels ────────────────────────────────────────────────
const LEVEL_TITLES = ["Newcomer", "Curious Mind", "Apprentice", "Explorer", "Tinkerer",
  "Practitioner", "Engineer", "Architect", "Specialist", "Systems Sage", "Grandmaster"];
function xpFromStats(st) {
  st = st || {};
  return (st.questions || 0) * 10 + (st.quizzes || 0) * 30
    + (st.concepts_mastered || 0) * 60 + (st.active_days || 0) * 20;
}
function levelInfo(xp) {
  let lvl = 1, need = 150, cum = 0;
  while (xp >= cum + need) { cum += need; lvl++; need = Math.round(need * 1.35); }
  const into = xp - cum;
  return {
    level: lvl, title: LEVEL_TITLES[Math.min(lvl - 1, LEVEL_TITLES.length - 1)],
    pct: Math.max(0, Math.min(100, Math.round((into / need) * 100))), toNext: need - into,
  };
}
// Celebrate only on a genuine increase after the first recorded level.
function maybeLevelUp(level) {
  const prev = parseInt(localStorage.getItem("memex.level") || "0", 10);
  if (level > prev) {
    localStorage.setItem("memex.level", String(level));
    if (prev > 0) { toast(`⭐ <span class="t-grad">Level ${level} — leveled up!</span>`); confetti({ count: 120 }); }
  }
}

// ── Achievements ───────────────────────────────────────────────
const ACHIEVEMENTS = [
  { ico: "🌱", name: "First Steps", sub: "Ask 1 question", test: (s) => (s.questions || 0) >= 1 },
  { ico: "🔥", name: "On Fire", sub: "5-day streak", test: (s) => s._streak >= 5 },
  { ico: "🧠", name: "Curious", sub: "25 questions", test: (s) => (s.questions || 0) >= 25 },
  { ico: "🎯", name: "Quizzer", sub: "5 quizzes", test: (s) => (s.quizzes || 0) >= 5 },
  { ico: "🏅", name: "Sharpshooter", sub: "80% avg quiz", test: (s) => (s.avg_quiz_score || 0) >= 0.8 },
  { ico: "🏆", name: "Master", sub: "10 concepts", test: (s) => (s.concepts_mastered || 0) >= 10 },
  { ico: "🧭", name: "Explorer", sub: "10 topics", test: (s) => (s.topics_touched || 0) >= 10 },
  { ico: "⚡", name: "Dedicated", sub: "7 active days", test: (s) => (s.active_days || 0) >= 7 },
];
function renderAchievements(st) {
  const s = Object.assign({ _streak: currentStreak() }, st || {});
  return ACHIEVEMENTS.map((a) => {
    const on = a.test(s);
    return `<div class="ach ${on ? "unlocked" : "locked"}" title="${esc(a.name)} — ${esc(a.sub)}">
      <div class="ach-ico">${a.ico}</div><div class="ach-name">${esc(a.name)}</div>
      <div class="ach-sub">${on ? "Unlocked" : esc(a.sub)}</div></div>`;
  }).join("");
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
  confetti({ count: 60 });
  return count;
}
function renderStreak() { const el = $("streakCount"); if (el) el.textContent = currentStreak(); }

// ── Mermaid + Marked ───────────────────────────────────────────
mermaid.initialize({ startOnLoad: false, theme: "neutral", fontFamily: "Inter, system-ui, sans-serif" });
marked.setOptions({ breaks: true, gfm: true });

function _renderTex(tex, display) {
  if (typeof katex === "undefined") return esc(tex);
  try {
    return katex.renderToString(tex.trim(), { displayMode: display, throwOnError: false, strict: false });
  } catch (_) {
    return `<code>${esc(tex)}</code>`;
  }
}

function renderMarkdown(text) {
  // 1) Pull mermaid + math out BEFORE markdown parsing so the parser can't mangle
  //    LaTeX (underscores, backslashes, braces). Each becomes an inert placeholder.
  const mermaidBlocks = [];
  const mph = (i) => `MERMAIDPLACEHOLDER${i}ENDPH`;
  let src = text.replace(/```mermaid\s*([\s\S]*?)```/g, (_, code) => {
    mermaidBlocks.push(code.trim());
    return `\n${mph(mermaidBlocks.length - 1)}\n`;
  });

  const math = [];
  const xph = (i) => `MATHPLACEHOLDER${i}ENDPH`;
  const push = (tex, display) => { math.push({ tex, display }); return xph(math.length - 1); };
  src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => push(t, true));     // $$ … $$
  src = src.replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => push(t, true));     // \[ … \]
  src = src.replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => push(t, false));    // \( … \)
  src = src.replace(/\\begin\{([a-z*]+)\}[\s\S]+?\\end\{\1\}/gi, (m) => push(m, true)); // bare env
  src = src.replace(/\$([^$\n]+?)\$/g, (_, t) => push(t, false));        // $ … $ (inline)

  let html = DOMPurify.sanitize(marked.parse(src));

  mermaidBlocks.forEach((code, i) => {
    const safe = DOMPurify.sanitize(code, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] });
    html = html.replace(new RegExp(`<p>${mph(i)}</p>|${mph(i)}`, "g"), () => `<pre class="mermaid">${safe}</pre>`);
  });
  math.forEach((m, i) => {
    const rendered = _renderTex(m.tex, m.display);   // KaTeX output is XSS-safe
    html = html.replace(new RegExp(`<p>${xph(i)}</p>|${xph(i)}`, "g"), () => rendered);
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
const VIEWS = ["home", "profile", "readiness", "chat", "progress", "path", "yourai", "interview"];

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
  else if (name === "readiness") renderReadiness();
  else if (name === "interview") renderInterview();
  else if (name === "progress") renderProgress();
  else if (name === "path") renderPath();
  else if (name === "yourai") renderYourAI();
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

let _reviewCache = null;
async function loadReview(force = false) {
  if (!force && _reviewCache) return _reviewCache;
  try {
    const r = await fetch(`/api/student/${encodeURIComponent(state.studentId)}/review`);
    if (!r.ok) return _reviewCache;
    _reviewCache = await r.json();
    return _reviewCache;
  } catch (_) { return _reviewCache; }
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
    .sort((a, b) => sortAreaKey(a.area, b.area));

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
  const [data, act, review] = await Promise.all([loadProgress(true), loadActivity(true), loadReview(true)]);
  const s = deriveSummary(data);
  const dueList = (review && review.due) || [];
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

  const xp = xpFromStats(ast);
  const li = levelInfo(xp);
  const levelHtml = `
    <div class="card level-card" style="margin-bottom:16px">
      <div class="level-badge"><div><b>${li.level}</b><small>LVL</small></div></div>
      <div class="level-body">
        <div class="level-top"><span class="level-name">${esc(li.title)}</span>
          <span class="level-xp">${xp} XP · ${li.toNext} to L${li.level + 1}</span></div>
        <div class="level-bar"><i data-w="${li.pct}"></i></div>
      </div>
    </div>`;

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

  let reviewHtml = "";
  if (dueList.length) {
    const first = dueList[0];
    const more = dueList.length > 1 ? ` +${dueList.length - 1} more` : "";
    reviewHtml = `
      <div style="height:1px;background:var(--border)"></div>
      <div class="tile">
        <div class="tile-ico">🔁</div>
        <div class="tile-body">
          <div class="tile-kicker">Due for review</div>
          <div class="tile-title">${esc(first.title)}${more}</div>
          <div class="tile-sub">Spaced repetition — revisit before it fades</div>
        </div>
        <button class="btn btn-ghost btn-sm" data-seed="${esc(first.topic_id || "")}">Review</button>
      </div>`;
  }

  el.innerHTML = `
    <div class="view-inner">
      <div class="view-head">
        <div class="view-eyebrow">Today</div>
        <h1 class="view-title">Hey ${esc(state.studentId)} 👋</h1>
        <p class="view-subtitle">${greeting}</p>
      </div>

      ${homeStatsHtml}
      ${levelHtml}

      <div class="grid-cards grid-2">
        <div class="card">
          <div class="ring-wrap">
            <div class="ring" id="homeRing"><span class="ring-val"><span id="homeRingVal">0</span><small>%</small></span></div>
            <div>
              <div class="tile-kicker">Overall mastery</div>
              <div style="font-family:var(--font-display);font-size:22px;font-weight:600;margin:2px 0 8px">${s.hasData ? overallPct + "% there" : "Just getting started"}</div>
              ${areasHtml}
            </div>
          </div>
        </div>

        <div class="card grid-cards" style="gap:18px">
          ${upNextHtml}
          <div style="height:1px;background:var(--border)"></div>
          ${weakHtml}
          ${reviewHtml}
        </div>
      </div>

      <div class="card card-grad" style="margin-top:16px;display:flex;align-items:center;gap:16px;flex-wrap:wrap">
        <div style="flex:1;min-width:200px">
          <div class="tile-kicker">Ready to learn?</div>
          <div style="font-family:var(--font-display);font-size:19px;font-weight:600">Jump back into the tutor</div>
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
    el.querySelectorAll(".bar > i[data-w], .level-bar > i[data-w]").forEach((i) => { i.style.width = i.dataset.w + "%"; });
  });
  maybeLevelUp(li.level);

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
    renderStarters(topicId);
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

  const pxp = xpFromStats(st);
  const pli = levelInfo(pxp);
  const levelHtml = `
    <div class="card level-card" style="margin-bottom:20px">
      <div class="level-badge"><div><b>${pli.level}</b><small>LVL</small></div></div>
      <div class="level-body">
        <div class="level-top"><span class="level-name">${esc(pli.title)}</span>
          <span class="level-xp">${pxp} XP · ${pli.toNext} to L${pli.level + 1}</span></div>
        <div class="level-bar"><i data-w="${pli.pct}"></i></div>
      </div>
    </div>`;
  const achHtml = `<div class="section-label">Achievements</div>
    <div class="card"><div class="ach-grid">${renderAchievements(st)}</div></div>`;

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

      ${levelHtml}

      <div class="stat-grid">${statHtml}</div>
      ${achHtml}
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
    el.querySelectorAll(".bar > i[data-w], .level-bar > i[data-w]").forEach((i) => { i.style.width = i.dataset.w + "%"; });
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
  const areas = Object.keys(byArea).sort(sortAreaKey);

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

// ── Your AI view (personalization + captured data) ─────────────
async function loadProfile() {
  try {
    const r = await fetch(`/api/student/${encodeURIComponent(state.studentId)}/profile`);
    return r.ok ? await r.json() : null;
  } catch (_) { return null; }
}
async function loadTraceSummary() {
  try {
    const r = await fetch(`/api/student/${encodeURIComponent(state.studentId)}/traces/summary`);
    return r.ok ? await r.json() : null;
  } catch (_) { return null; }
}

function chipList(items, cls) {
  if (!items || !items.length) return `<span class="view-subtitle">None yet.</span>`;
  return items.map((t) => `<span class="chip ${cls}" style="margin:0 6px 6px 0">${esc(t)}</span>`).join("");
}

async function renderYourAI() {
  const el = $("view-yourai");
  const [profile, traces] = await Promise.all([loadProfile(), loadTraceSummary()]);
  const p = profile || {};
  const overallPct = Math.round((p.overall_mastery || 0) * 100);
  const count = (traces && traces.count) || 0;
  const recent = (traces && traces.recent) || [];

  const rewardDot = (r) => {
    if (r == null) return `<span class="rd rd-none" title="No outcome yet"></span>`;
    const cls = r >= 0.6 ? "rd-good" : r > 0 ? "rd-warn" : "rd-bad";
    return `<span class="rd ${cls}" title="reward ${r}"></span>`;
  };
  const recentHtml = recent.length
    ? recent.slice(0, 10).map((t) => `
        <div class="tl-item">
          ${rewardDot(t.reward)}
          <div class="tl-body">
            <div class="tl-label">${esc(t.task_text)}</div>
            <div class="tl-meta">${t.n_selected} item${t.n_selected === 1 ? "" : "s"} selected · ${formatRelative(t.occurred_at)}</div>
          </div>
        </div>`).join("")
    : `<p class="view-subtitle">No turns captured yet — ask the tutor something and it starts learning your style.</p>`;

  el.innerHTML = `
    <div class="view-inner">
      <div class="view-head">
        <div class="view-eyebrow">Your AI</div>
        <h1 class="view-title">What Memex has learned about you</h1>
        <p class="view-subtitle">Memex adapts to you every turn — and remembers your context so it can be fine-tuned to teach <em>you</em> better over time.</p>
      </div>

      ${p.learning_style ? `<div class="card" style="margin-bottom:16px;background:var(--grad-soft);border-color:transparent">
        <div class="tile-kicker" style="margin-bottom:6px">🧭 Your learning style</div>
        <div style="font-size:15px;line-height:1.55">${esc(p.learning_style)}</div>
      </div>` : ""}

      <div class="grid-cards grid-2">
        <div class="card">
          <div class="ring-wrap">
            <div class="ring" id="aiRing"><span class="ring-val"><span id="aiRingVal">0</span><small>%</small></span></div>
            <div>
              <div class="tile-kicker">Overall mastery</div>
              <div style="font-family:var(--font-display);font-size:22px;font-weight:600;margin:2px 0 10px">${overallPct ? overallPct + "% there" : "Just getting started"}</div>
              <div class="tile-kicker" style="margin-bottom:6px">💪 Strengths</div>
              <div>${chipList(p.strengths, "chip-area")}</div>
              <div class="tile-kicker" style="margin:12px 0 6px">🎯 Working on</div>
              <div>${chipList(p.weaknesses, "chip-area")}</div>
            </div>
          </div>
        </div>

        <div class="card grid-cards" style="gap:16px">
          <div class="tile">
            <div class="tile-ico">📦</div>
            <div class="tile-body">
              <div class="tile-kicker">Personalization data captured</div>
              <div class="tile-title"><span data-count="${count}">0</span> learning trace${count === 1 ? "" : "s"}</div>
              <div class="tile-sub">Each turn (your question, the context chosen, the reply, and how it landed) is saved to fine-tune on.</div>
            </div>
          </div>
          <div>
            <div class="tile-kicker" style="margin-bottom:6px">⚠️ Active misconceptions</div>
            <div>${(p.misconceptions && p.misconceptions.length) ? p.misconceptions.map((m) => `<div style="display:flex;gap:8px;font-size:12.5px;color:var(--text-muted);margin:5px 0"><span style="color:var(--warn)">⚠</span><span>${esc(m)}</span></div>`).join("") : `<span class="view-subtitle">None flagged.</span>`}</div>
            <div class="tile-kicker" style="margin:12px 0 6px">🔁 Due for review</div>
            <div>${chipList(p.due_for_review, "chip-area")}</div>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:2px">
            <button class="btn btn-ghost btn-sm" id="aiExport">⬇ Export my data (JSONL)</button>
            <button class="btn btn-ghost btn-sm" id="aiReset" style="color:var(--bad)">Reset my data</button>
          </div>
        </div>
      </div>

      <div class="section-label">Recently captured turns</div>
      <div class="card"><div class="timeline">${recentHtml}</div></div>
    </div>`;

  requestAnimationFrame(() => {
    const ring = $("aiRing");
    if (ring) ring.style.setProperty("--val", overallPct);
    countUp($("aiRingVal"), overallPct);
    el.querySelectorAll("[data-count]").forEach((n) => countUp(n, parseInt(n.dataset.count, 10)));
  });

  $("aiExport")?.addEventListener("click", () => {
    window.open(`/api/student/${encodeURIComponent(state.studentId)}/traces/export`, "_blank");
  });
  $("aiReset")?.addEventListener("click", async () => {
    if (!confirm("Erase all captured personalization data for this user? This can't be undone.")) return;
    try {
      await fetch(`/api/student/${encodeURIComponent(state.studentId)}/traces`, { method: "DELETE" });
      toast("🧹 <span class=\"t-grad\">Your data was reset.</span>");
      renderYourAI();
    } catch (_) {}
  });
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
async function loadConversation(conversationId, { switchView = true } = {}) {
  state.conversationId = conversationId;
  state.messages = []; state.reuseCounts = {};
  $("chat").innerHTML = "";
  $("welcome").classList.add("hidden");
  if (switchView) showView("chat");
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
// Topic-tailored starter prompts (instant, no LLM) — shown when a topic is picked.
const STARTER_TEMPLATES = [
  { icon: "🧠", q: (t) => `Explain ${t} from first principles`, sub: "Core idea" },
  { icon: "⚖️", q: (t) => `What are the key tradeoffs in ${t}?`, sub: "Design tradeoffs" },
  { icon: "🔧", q: (t) => `How does ${t} work in production?`, sub: "In practice" },
  { icon: "⚠️", q: (t) => `What's the most common misconception about ${t}?`, sub: "Avoid pitfalls" },
  { icon: "🎤", q: (t) => `How would ${t} come up in an ML-systems interview?`, sub: "Interview angle" },
  { icon: "🔢", q: (t) => `Do the back-of-the-envelope math for ${t}`, sub: "Quantitative" },
];

function renderStarters(topicId) {
  const grid = $("starterGrid");
  if (!grid) return;
  grid.innerHTML = "";
  const cards = (topicId && TOPIC_MAP[topicId])
    ? STARTER_TEMPLATES.map((s) => ({ icon: s.icon, title: s.q(TOPIC_MAP[topicId].title), sub: s.sub }))
    : STARTERS;
  cards.forEach(({ icon, title, sub }) => {
    const btn = document.createElement("button");
    btn.className = "starter-card";
    btn.innerHTML = `<div class="starter-ico">${icon}</div>
      <div class="starter-title">${esc(title)}</div>
      <div class="starter-sub">${esc(sub)}</div>`;
    btn.addEventListener("click", () => sendMessage(title));
    grid.appendChild(btn);
  });
}
function buildStarterGrid() { renderStarters(state.topicId || null); }

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

  const thinkHtml = (msg.thinking && msg.thinking.trim())
    ? `<details class="thinking-box"><summary>💭 Thought process</summary><div class="thinking-content">${esc(msg.thinking)}</div></details>`
    : "";

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
    ${thinkHtml}
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

  state.topicId = $("topicSelect").value || null;

  state.messages.push({ role: "user", content: text });
  appendUserBubble(text);

  $("msgInput").value = ""; $("msgInput").style.height = "auto"; $("sendBtn").disabled = true;
  const thinking = showThinking();
  setInputLocked(true);

  // A live bubble: the model's thinking streams into a collapsible box, then the
  // answer fills below. On completion the whole row is re-rendered (markdown,
  // citations, footer) with the thinking preserved as a collapsed details.
  let streamRow = null, streamBody = null, thinkContent = null, acc = "", thinkAcc = "";
  function ensureRow() {
    if (streamRow) return;
    thinking.remove();
    streamRow = document.createElement("div");
    streamRow.className = "msg-row assistant";
    $("chat").appendChild(streamRow);
  }
  function ensureThink() {
    ensureRow();
    if (thinkContent) return;
    const tb = document.createElement("details");
    tb.className = "thinking-box"; tb.open = true;
    tb.innerHTML = '<summary>💭 Thinking…</summary><div class="thinking-content"></div>';
    streamRow.appendChild(tb);
    thinkContent = tb.querySelector(".thinking-content");
  }
  function ensureBody() {
    ensureRow();
    if (streamBody) return;
    streamBody = document.createElement("div");
    streamBody.className = "prose-chat streaming";
    streamRow.appendChild(streamBody);
  }

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        student_id: state.studentId, conversation_id: state.conversationId,
        topic_id: state.topicId, question: text, budget: state.budget, reuse_counts: state.reuseCounts,
        router: state.router || null,
      }),
    });
    if (!res.ok || !res.body) { thinking.remove(); appendError(await res.text().catch(() => "stream failed")); return; }

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "", done = null;
    while (true) {
      const { value, done: rdone } = await reader.read();
      if (rdone) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, nl); buf = buf.slice(nl + 2);
        if (!frame.startsWith("data:")) continue;
        let payload;
        try { payload = JSON.parse(frame.slice(5).trim()); } catch (_) { continue; }
        if (payload.thinking != null) {
          ensureThink();
          thinkAcc += payload.thinking;
          thinkContent.textContent = thinkAcc;
          scrollBottom();
        } else if (payload.delta != null) {
          ensureBody();
          acc += payload.delta;
          streamBody.textContent = acc;   // raw text while streaming (safe)
          scrollBottom();
        } else if (payload.error) {
          thinking.remove(); streamRow?.remove();
          appendError(payload.error); return;
        } else if (payload.done) {
          done = payload;
        }
      }
    }
    thinking.remove(); streamRow?.remove();
    if (!done) { appendError("No response received."); return; }

    if (done.conversation_id) state.conversationId = done.conversation_id;
    done.selected?.forEach((it) => { state.reuseCounts[it.id] = (state.reuseCounts[it.id] || 0) + 1; });

    const msgIdx = state.messages.length;
    const aMsg = { role: "assistant", content: done.reply, references: done.references, selected: done.selected, dropped: done.dropped, tokens_used: done.tokens_used, router: done.router, thinking: thinkAcc || null };
    state.messages.push(aMsg);
    appendAssistantMessage(aMsg, msgIdx);   // final render: markdown + citations + mermaid + footer
    scrollBottom();

    bumpStreak();
    loadConversations();
    loadProgress(true);
    loadActivity(true);
    loadReview(true);
  } catch (err) {
    thinking.remove(); streamRow?.remove();
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

  const ftRouter = msg.router && msg.router !== "heuristic";
  const routerLabel = ftRouter ? `Fine-tuned router (${esc(msg.router)})` : "Heuristic engine";
  let html = `<p class="view-subtitle" style="margin:0 0 14px">Selected by <b>${routerLabel}</b>. ${ftRouter ? "These are the items the fine-tuned model chose for this answer." : "Scored by relevance, recency, your misconceptions, prerequisites, and reuse."}</p>`;

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
  // No topic picked? Infer it from the question that prompted this answer — the
  // student shouldn't have to choose a topic to test themselves on what they asked.
  let qText = null;
  for (let i = Math.min(msgIdx, state.messages.length - 1); i >= 0; i--) {
    if (state.messages[i] && state.messages[i].role === "user") { qText = state.messages[i].content; break; }
  }
  const area = row.querySelector(".quiz-area");
  area.innerHTML = `<div class="quiz-card"><div style="display:flex;align-items:center;gap:8px;font-size:12px;color:var(--text-muted)"><div class="thinking-dots"><span></span><span></span><span></span></div> Generating question…</div></div>`;
  try {
    const res = await fetch("/api/quiz/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ topic_id: topicId, student_id: state.studentId, question: qText }) });
    if (!res.ok) throw new Error(await res.text());
    const { question, rubric, difficulty, topic_id } = await res.json();
    renderQuizQuestion(area, msgIdx, question, rubric, topic_id || topicId, difficulty);
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
    if (score >= 0.8) { toast(`🎉 <span class="t-grad">Nailed it — ${pct}%!</span>`); confetti({ count: 85 }); }
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
    const res = await fetch("/api/diagnostic/turn", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ student_id: state.studentId, original_question: originalQuestion, diagnosis: diagData.diagnosis, follow_up_question: diagData.follow_up_question, student_answer: answer, turn_index: turnIndex, topic_id: state.topicId || $("topicSelect").value || null }) });
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

$("topicSelect").addEventListener("change", (e) => { state.topicId = e.target.value || null; renderStarters(state.topicId); });
$("routerSelect").addEventListener("change", (e) => { state.router = e.target.value || ""; });

$("routingClose").addEventListener("click", closeRoutingModal);
$("routingBackdrop").addEventListener("click", closeRoutingModal);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeRoutingModal(); });

$("welcomeBackOpenMemory").addEventListener("click", () => showView("progress"));

window.addEventListener("hashchange", routeFromHash);

async function loadRouters() {
  try {
    const r = await fetch("/api/routers");
    if (!r.ok) return;
    const d = await r.json();
    const sel = $("routerSelect");
    if (!sel) return;
    const labels = { qwen2_5_0_5b: "0.5B", qwen2_5_1_5b: "1.5B", qwen2_5_3b: "3B", qwen2_5_7b: "7B" };
    (d.finetuned || []).forEach((id) => {
      const o = document.createElement("option");
      o.value = id;
      o.textContent = id === "remote" ? "Router: Hosted (vLLM)" : "Router: Finetuned " + (labels[id] || id);
      sel.appendChild(o);
    });
  } catch (_) {}
}

// ── Auth gate ──────────────────────────────────────────────────
const auth = { mode: "login" };
function setAuthMode(mode) {
  auth.mode = mode;
  document.querySelectorAll(".auth-tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === mode));
  document.querySelectorAll(".auth-signup-only").forEach((el) => el.classList.toggle("hidden", mode !== "signup"));
  $("authSubmit").textContent = mode === "signup" ? "Create account" : "Sign in";
  $("authUsername").placeholder = mode === "signup" ? "Username" : "Username or email";
  $("authError").classList.add("hidden");
}
async function submitAuth(e) {
  e.preventDefault();
  const u = $("authUsername").value.trim(), em = $("authEmail").value.trim(), pw = $("authPassword").value;
  const err = $("authError");
  err.classList.add("hidden");
  try {
    const res = auth.mode === "signup"
      ? await fetch("/api/auth/signup", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username: u, email: em, password: pw }) })
      : await fetch("/api/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ login: u, password: pw }) });
    if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || "Something went wrong"); }
    onLoggedIn((await res.json()).username);
  } catch (e2) { err.textContent = e2.message; err.classList.remove("hidden"); }
}
function onLoggedIn(username) {
  state.studentId = username;
  setTimeout(checkOnboarding, 300);
  const tag = $("authUserTag"); if (tag) tag.textContent = username;
  $("authGate").classList.add("hidden");
  renderStreak();
  bootApp();
}
async function logout() {
  try { await fetch("/api/auth/logout", { method: "POST" }); } catch (_) {}
  location.reload();
}

async function bootApp() {
  await loadTopics();
  await loadRouters();
  await loadConversations();
  // Resume the user's most recent conversation so logging back in picks up
  // exactly where they left off (their data is persisted server-side per user).
  if (state.conversations.length) {
    await loadConversation(state.conversations[0].id, { switchView: false });
  }
  await maybeShowWelcomeBack();
  routeFromHash();   // default #/home
  $("obSave")?.addEventListener("click", saveOnboarding);
  checkOnboarding();
}

// ── Interview Readiness view (B2C wedge) ───────────────────────
async function renderReadiness() {
  const el = $("view-readiness");
  let d;
  try {
    const r = await fetch(`/api/student/${encodeURIComponent(state.studentId)}/readiness`);
    if (!r.ok) throw new Error("http " + r.status);
    d = await r.json();
  } catch (_) {
    el.innerHTML = '<div class="view-inner"><p class="view-subtitle">Could not load readiness.</p></div>';
    return;
  }
  const stripCode = (x) => (x || "").replace(/\s*\(.*?\)\s*$/, "");
  const pct = Math.round((d.overall_readiness || 0) * 100);
  const readyLabel = pct >= 70 ? "Interview-ready" : pct >= 40 ? "Getting there" : "Early — lots to drill";
  const areas = (d.areas || []).map((a) => {
    const ap = Math.round(a.readiness * 100);
    const cls = ap >= 70 ? "good" : ap >= 40 ? "warn" : "bad";
    return `<div class="bar-row"><span class="lbl">${esc(stripCode(a.area_title) || ("Area " + a.area))} <span class="pct" style="color:var(--text-faint)">${a.covered}/${a.total}</span></span><span class="pct">${ap}%</span><div class="bar ${cls}"><i data-w="${ap}"></i></div></div>`;
  }).join("");
  const next = d.next_up;
  const pro = !!d.pro;                       // server-enforced entitlement
  // Hire-bar verdict (from interview history) — the core hook.
  const v = d.verdict || {};
  const tierClass = { frontier: "good", ready: "good", borderline: "warn", not_ready: "bad", remediation: "bad" }[v.tier] || "warn";
  const tierEmoji = { frontier: "🏆", ready: "✅", borderline: "🟡", not_ready: "🔴", remediation: "🧱", no_data: "🎤" }[v.tier] || "🎤";
  let verdictBlock = "";
  if (v.tier === "no_data") {
    verdictBlock = `<div class="card" style="background:var(--grad-soft);border:1px solid var(--border)">
        <div style="display:flex;gap:14px;align-items:center">
          <div style="font-size:34px">🎤</div>
          <div><div style="font-family:var(--font-display);font-weight:600;font-size:18px">No verdict yet</div>
          <div class="tile-sub">${esc(v.label || "Take a mock interview to get your readiness verdict.")}</div></div>
          <button class="btn btn-primary" style="margin-left:auto" data-go="interview">Start a mock interview →</button>
        </div></div>`;
  } else if (v.tier) {
    const crit = (v.critical_failures || []).map((c) =>
      `<span class="pill bad">${esc(stripCode(c.category.replace(/_/g, " ")))} ${c.score}</span>`).join(" ");
    verdictBlock = `<div class="card verdict-card">
        <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
          <div style="font-size:40px;line-height:1">${tierEmoji}</div>
          <div style="flex:1;min-width:220px">
            <div class="tile-kicker">Hire-bar verdict</div>
            <div style="font-family:var(--font-display);font-weight:700;font-size:22px;margin:2px 0">${esc(v.label || "")}</div>
            <div class="tile-sub">Blended score <b>${v.score != null ? v.score : "—"}</b> · ${v.interview_count} interview${v.interview_count === 1 ? "" : "s"} · avg ${v.interview_avg != null ? v.interview_avg : "—"} · consistency ${v.consistency != null ? v.consistency : "—"}</div>
            ${crit ? `<div style="margin-top:10px"><span class="tile-sub">Critical gaps blocking a pass:</span><br>${crit}</div>` : ""}
          </div>
          <div class="verdict-score ${tierClass}">${v.score != null ? Math.round(v.score) : "—"}</div>
        </div>
        <div class="tile-sub" style="margin-top:10px;opacity:.8">Verdict uses a real hire bar: 60% interview average + 20% trajectory + 20% consistency, gated by your four load-bearing skills. Ready = avg ≥ 80 over ≥ 3 interviews with no critical gap.</div>
      </div>`;
  }
  const allGaps = d.gaps || [];
  const total = d.gaps_total != null ? d.gaps_total : allGaps.length;
  const gapRow = (g) =>
    `<div class="tl-item"><div class="tl-ico quiz">${g.started ? "◐" : "○"}</div><div class="tl-body"><div class="tl-label">${esc(g.title)}</div><div class="tl-meta">${esc(stripCode(g.area_title))} · ${Math.round(g.mastery * 100)}%</div></div><button class="btn btn-ghost btn-sm" data-seed="${esc(g.topic_id)}">Drill</button></div>`;
  const lockedRow = () =>
    `<div class="tl-item" style="filter:blur(5px);pointer-events:none;user-select:none"><div class="tl-ico quiz">○</div><div class="tl-body"><div class="tl-label">████████ ███████</div><div class="tl-meta">██████ · ██%</div></div></div>`;
  let gapsBlock;
  if (total === 0) {
    gapsBlock = `<div class="card"><p class="view-subtitle">No gaps — you're interview-ready. 🎯</p></div>`;
  } else if (pro) {
    gapsBlock = `<div class="card"><div class="timeline">${allGaps.slice(0, 8).map(gapRow).join("")}</div></div>`;
  } else {
    // Free: server returns ONE real gap; the rest are locked behind the paywall.
    const teaser = allGaps.length ? gapRow(allGaps[0]) : "";
    const more = Math.max(0, total - 1);
    const blurred = Array.from({ length: Math.min(3, more) }, lockedRow).join("");
    gapsBlock = `<div class="card">
        <div class="timeline">${teaser}${blurred}</div>
        <div style="margin-top:14px;padding:16px;border-radius:14px;background:var(--grad-soft);border:1px solid var(--border)">
          <div style="font-family:var(--font-display);font-weight:600;font-size:16px">🔒 Unlock your full gap analysis + drill plan</div>
          <div class="tile-sub" style="margin:4px 0 12px">See all <b>${more}</b> remaining gaps across the curriculum, a prioritized study plan, and your readiness trend over time.</div>
          <button class="btn btn-primary upgrade-btn" data-src="readiness_gaps">Get Pro — see your full report →</button>
          <div class="tile-sub" style="margin-top:8px">Plans: Free · <b>Pro</b> · Interview Bootcamp · Premium Human+AI</div>
        </div>
      </div>`;
  }
  el.innerHTML = `
    <div class="view-inner">
      <div class="view-head">
        <div class="view-eyebrow">Interview Readiness</div>
        <h1 class="view-title">Are you ready to interview?</h1>
        <p class="view-subtitle">Scored across the full ML-systems curriculum — untested topics count against you, just like a real interview.</p>
      </div>
      ${verdictBlock}
      <div class="grid-cards grid-2">
        <div class="card"><div class="ring-wrap">
          <div class="ring" id="rdyRing"><span class="ring-val"><span id="rdyVal">0</span><small>%</small></span></div>
          <div>
            <div class="tile-kicker">Overall readiness</div>
            <div style="font-family:var(--font-display);font-size:20px;font-weight:600;margin:2px 0 8px">${readyLabel}</div>
            <div class="tile-sub">${d.topics_covered}/${d.topics_total} topics touched · ${d.concepts_mastered} mastered</div>
            ${next ? `<button class="btn btn-primary btn-sm" style="margin-top:10px" data-seed="${esc(next.topic_id)}">Drill next: ${esc(next.title)} →</button>` : ""}
          </div>
        </div></div>
        <div class="card"><div class="tile-kicker" style="margin-bottom:10px">Readiness by area</div>${areas}</div>
      </div>
      <div class="section-label">Top gaps to close</div>
      ${gapsBlock}
    </div>`;
  requestAnimationFrame(() => {
    const ring = $("rdyRing"); if (ring) ring.style.setProperty("--val", pct);
    if ($("rdyVal")) countUp($("rdyVal"), pct);
    el.querySelectorAll(".bar > i[data-w]").forEach((i) => { i.style.width = i.dataset.w + "%"; });
  });
  el.querySelectorAll("[data-seed]").forEach((b) => b.addEventListener("click", () => seedChat(b.dataset.seed)));
  el.querySelectorAll(".upgrade-btn").forEach((b) => b.addEventListener("click", () => upgradeIntent(b.dataset.src)));
  el.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => showView(b.dataset.go)));
}

// Willingness-to-pay: record the "Get Pro" click; open checkout if configured.
async function upgradeIntent(source) {
  let url = null;
  try {
    const r = await fetch("/api/billing/checkout", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: source || "readiness" }),
    });
    if (r.ok) url = (await r.json()).checkout_url;
  } catch (_) {}
  if (url) window.open(url, "_blank");
  else if (typeof toast === "function") toast(`✨ <span class="t-grad">You're on the Pro waitlist!</span>`);
  else alert("You're on the Pro waitlist!");
}

// ── Mock Interview (AI Judge) ──────────────────────────────────
let _ivQuestion = null, _ivTopic = null, _ivTopicTitle = "", _ivLevel = "intermediate", _ivMode = "interview";
let _ivTranscript = [];   // multi-turn design interview: [{q,a}, ...]
const IV_MAX_TURNS = 3;   // Q1 + up to 2 follow-up probes

const IV_MODES = {
  interview: {
    title: "ML-systems design interview", eyebrow: "Mock Interview",
    sub: "A staff-engineer AI judge runs a real back-and-forth — it asks follow-up probes on your weak spots, then scores the whole conversation across 10 rubric categories.",
    gen: "/api/interview/question", qkey: "question", evalUrl: "/api/interview/evaluate",
    qlabel: "Design question", spinning: "Generating a design question…",
    placeholder: "Reason out loud: clarify requirements, identify bottlenecks, do the latency/throughput/memory math, name the tradeoffs…",
  },
  debug: {
    title: "Production debugging", eyebrow: "Incident response",
    sub: "A realistic production incident with simulated logs & metrics. You're graded on your debugging process — hypotheses, evidence, root cause, fix.",
    gen: "/api/debug/incident", qkey: "incident", evalUrl: "/api/debug/evaluate",
    qlabel: "Incident", spinning: "Spinning up a production incident…",
    placeholder: "Diagnose it: hypotheses, what the logs/metrics tell you, the root cause, and the fix…",
  },
  forward: {
    title: "Forward-deployed engineer", eyebrow: "Customer scenario",
    sub: "A customer reports a vague problem (\"our agent feels slow\"). You're graded on the 7 forward-deployed sub-skills: framing, asking for the right metrics, localizing, iterating, the fix, its cost/SLA tradeoff, and explaining it to a non-expert.",
    gen: "/api/forward/scenario", qkey: "scenario", evalUrl: "/api/forward/evaluate",
    qlabel: "Customer says", spinning: "Drafting a customer scenario…",
    placeholder: "Handle the customer: what metrics do you ask for, how do you localize the bottleneck, what's the fix + its cost/SLA tradeoff, and how do you explain it to them in plain language?",
  },
};

async function renderInterview() {
  const el = $("view-interview");
  el.innerHTML = `
    <div class="view-inner">
      <div class="view-head">
        <div class="view-eyebrow" id="ivEyebrow">Mock Interview</div>
        <h1 class="view-title" id="ivTitle">ML-systems design interview</h1>
        <p class="view-subtitle" id="ivSub">${IV_MODES.interview.sub}</p>
      </div>
      <div class="card">
        <div class="tile-kicker" style="margin-bottom:8px">Set up</div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
          <select id="ivMode" class="topic-select">
            <option value="interview">Design interview (multi-turn)</option>
            <option value="debug">Production debugging</option>
            <option value="forward">Forward-deployed (customer)</option>
          </select>
          <select id="ivLevel" class="topic-select">
            <option value="beginner">Beginner</option>
            <option value="intermediate" selected>Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
          <select id="ivTopic" class="topic-select"><option value="">Weakest topic (auto)</option></select>
          <button class="btn btn-primary" id="ivGen">Start →</button>
        </div>
      </div>
      <div id="ivBody"></div>
    </div>`;
  const ts = $("ivTopic");
  Object.entries(TOPIC_MAP).forEach(([id, m]) => { const o = document.createElement("option"); o.value = id; o.textContent = m.title; ts.appendChild(o); });
  $("ivMode").addEventListener("change", () => {
    const m = IV_MODES[$("ivMode").value] || IV_MODES.interview;
    $("ivEyebrow").textContent = m.eyebrow; $("ivTitle").textContent = m.title; $("ivSub").textContent = m.sub;
    $("ivGen").textContent = $("ivMode").value === "interview" ? "Start →" : "Generate →";
  });
  $("ivGen").addEventListener("click", ivGenerate);
}

async function ivGenerate() {
  const body = $("ivBody"), btn = $("ivGen");
  _ivLevel = $("ivLevel").value;
  const topic = $("ivTopic").value || null;
  _ivMode = $("ivMode") ? $("ivMode").value : "interview";
  _ivTranscript = [];
  const m = IV_MODES[_ivMode];
  btn.disabled = true;
  body.innerHTML = `<div class="card"><span class="thinking-dots"><span></span><span></span><span></span></span> ${m.spinning}</div>`;
  try {
    const r = await fetch(m.gen, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ student_id: state.studentId, topic_id: topic, level: _ivLevel }) });
    const d = await r.json();
    _ivQuestion = d[m.qkey]; _ivTopic = d.topic_id; _ivTopicTitle = d.topic_title || "";
    ivRenderTurn();
  } catch (_) { body.innerHTML = `<p class="view-subtitle">Could not start. Try again.</p>`; }
  finally { btn.disabled = false; }
}

// Render the current question (plus prior transcript, for multi-turn) + answer box.
function ivRenderTurn() {
  const body = $("ivBody"), m = IV_MODES[_ivMode];
  const multi = _ivMode === "interview";
  const turnNo = _ivTranscript.length + 1;
  const prior = multi && _ivTranscript.length
    ? `<div class="card" style="background:var(--surface-2)"><div class="tile-kicker" style="margin-bottom:8px">Interview so far</div><div class="timeline">${
        _ivTranscript.map((t, i) => `<div class="tl-item"><div class="tl-ico">${i === 0 ? "Q" : "↳"}</div><div class="tl-body"><div class="tl-label" style="white-space:normal">${esc(t.q)}</div><div class="tl-meta" style="white-space:normal;opacity:.8">${esc(t.a)}</div></div></div>`).join("")
      }</div></div>` : "";
  const qLabel = multi
    ? (turnNo === 1 ? m.qlabel : `Follow-up ${turnNo - 1}`) + ` · turn ${turnNo} of up to ${IV_MAX_TURNS}`
    : `${m.qlabel} · ${esc(_ivTopicTitle)} · ${esc(_ivLevel)}`;
  // multi-turn: allow finishing early once at least one answer is in
  const finishBtn = multi && _ivTranscript.length >= 1
    ? `<button class="btn btn-ghost" id="ivFinish">Finish &amp; get verdict</button>` : "";
  const primaryLabel = multi
    ? (turnNo >= IV_MAX_TURNS ? "Submit final answer →" : "Answer →")
    : "Submit for evaluation";
  body.innerHTML = `
    ${prior}
    <div class="card">
      <div class="tile-kicker">${qLabel}</div>
      <div class="prose-chat" style="margin:8px 0 14px">${renderMarkdown(_ivQuestion)}</div>
      <textarea id="ivAnswer" class="input-area" rows="${multi ? 7 : 10}" placeholder="${m.placeholder}"></textarea>
      <div style="display:flex;gap:10px;justify-content:flex-end;margin-top:10px">${finishBtn}<button class="btn btn-primary" id="ivSubmit">${primaryLabel}</button></div>
      <div id="ivResult" style="margin-top:14px"></div>
    </div>`;
  $("ivSubmit").addEventListener("click", () => ivStep(false));
  $("ivFinish")?.addEventListener("click", () => ivStep(true));
}

// One step of the interview: for debug/forward this grades immediately; for the
// multi-turn design interview it either asks a follow-up or finalizes the verdict.
async function ivStep(finalize) {
  const ansEl = $("ivAnswer"); const ans = ansEl.value.trim();
  if (!ans) { ansEl.focus(); return; }
  const res = $("ivResult"), btn = $("ivSubmit"); btn.disabled = true;
  const m = IV_MODES[_ivMode];

  if (_ivMode !== "interview") {
    res.innerHTML = `<span class="thinking-dots"><span></span><span></span><span></span></span> The AI judge is grading…`;
    const payload = _ivMode === "debug"
      ? { student_id: state.studentId, topic_id: _ivTopic, level: _ivLevel, incident: _ivQuestion, diagnosis: ans }
      : { student_id: state.studentId, topic_id: _ivTopic, level: _ivLevel, scenario: _ivQuestion, response: ans };
    try { const r = await fetch(m.evalUrl, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); ivShowReport(await r.json(), res); }
    catch (_) { res.innerHTML = `<p style="color:var(--bad)">Evaluation failed. Try again.</p>`; btn.disabled = false; }
    return;
  }

  // multi-turn design interview
  _ivTranscript.push({ q: _ivQuestion, a: ans });
  const done = finalize || _ivTranscript.length >= IV_MAX_TURNS;
  if (done) {
    res.innerHTML = `<span class="thinking-dots"><span></span><span></span><span></span></span> The AI judge is grading the full interview…`;
    try {
      const r = await fetch(m.evalUrl, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ student_id: state.studentId, topic_id: _ivTopic, level: _ivLevel, transcript: _ivTranscript }) });
      ivShowReport(await r.json(), res);
    } catch (_) { res.innerHTML = `<p style="color:var(--bad)">Evaluation failed. Try again.</p>`; btn.disabled = false; }
    return;
  }
  // otherwise ask a follow-up probe
  res.innerHTML = `<span class="thinking-dots"><span></span><span></span><span></span></span> The interviewer is thinking of a follow-up…`;
  try {
    const r = await fetch("/api/interview/followup", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ student_id: state.studentId, topic_id: _ivTopic, level: _ivLevel, transcript: _ivTranscript }) });
    const d = await r.json();
    _ivQuestion = d.question;
    ivRenderTurn();
  } catch (_) { res.innerHTML = `<p style="color:var(--bad)">Could not generate a follow-up. Try again.</p>`; btn.disabled = false; }
}

function ivShowReport(ev, res) {
  res.innerHTML = ivReport(ev);
  res.querySelectorAll(".bar > i[data-w]").forEach((i) => { i.style.width = i.dataset.w + "%"; });
  res.querySelector(".iv-next")?.addEventListener("click", () => { if (_ivTopic) seedChat(_ivTopic); });
}
function ivReport(ev) {
  const pct = ev.overall_score || 0;
  const col = pct >= 70 ? "good" : pct >= 40 ? "warn" : "bad";
  const cats = Object.entries(ev.category_scores || {}).map(([k, v]) =>
    `<div class="bar-row"><span class="lbl">${esc(k.replace(/_/g, " "))}</span><span class="pct">${v}</span><div class="bar ${v >= 70 ? "good" : v >= 40 ? "warn" : "bad"}"><i data-w="${v}"></i></div></div>`).join("");
  const list = (arr, ic) => ((arr || []).slice(0, 5).map((x) =>
    `<div class="tl-item"><div class="tl-ico">${ic}</div><div class="tl-body"><div class="tl-label" style="white-space:normal">${esc(typeof x === "string" ? x : (x.description || ""))}</div></div></div>`).join("") || '<p class="view-subtitle">—</p>');
  return `
    <div class="section-label">Evaluation</div>
    <div class="grid-cards grid-2">
      <div class="card"><div style="display:flex;align-items:center;gap:16px">
        <div style="font-family:var(--font-display);font-size:42px;font-weight:700;color:var(--${col})">${pct}</div>
        <div><div class="tile-kicker">Overall / 100${ev.turns ? ` · ${ev.turns}-turn interview` : ""}</div><div class="tile-sub">Next: <b>${esc(ev.next_topic || "")}</b> · ${esc((ev.recommended_exercise_type || "").replace(/_/g, " "))}</div>
        <button class="btn btn-ghost btn-sm iv-next" style="margin-top:8px">Drill the gap →</button></div>
      </div></div>
      <div class="card">${cats}</div>
    </div>
    <div class="grid-cards grid-2" style="margin-top:8px">
      <div class="card"><div class="tile-kicker" style="margin-bottom:8px">💪 Strengths</div><div class="timeline">${list(ev.strengths, "✓")}</div></div>
      <div class="card"><div class="tile-kicker" style="margin-bottom:8px">🎯 Weaknesses</div><div class="timeline">${list(ev.weaknesses, "•")}</div></div>
    </div>
    <div class="card" style="margin-top:8px"><div class="tile-kicker" style="margin-bottom:8px">⚠ Detected misconceptions</div><div class="timeline">${list(ev.misconceptions, "⚠")}</div></div>
    <details class="card" style="margin-top:8px"><summary style="cursor:pointer;font-weight:600">📝 See a senior-level model answer</summary><div class="prose-chat" style="margin-top:10px">${(ev.improved_answer || "").trim() ? renderMarkdown(ev.improved_answer) : '<p class="view-subtitle" style="margin:0">The judge didn\'t return a model answer this time — regenerate to see one.</p>'}</div></details>`;
}

// ── Onboarding intake ──────────────────────────────────────────
async function checkOnboarding() {
  try {
    const r = await fetch(`/api/student/${encodeURIComponent(state.studentId)}/onboarding`);
    if (!r.ok) return;
    const d = await r.json();
    if (!d.onboarding || !d.onboarding.goal) {
      $("onboardModal")?.classList.remove("hidden");
    }
  } catch (_) {}
}
async function saveOnboarding() {
  const body = {
    goal: $("obGoal").value, level: $("obLevel").value,
    target: $("obTarget").value || null, learning_style: $("obStyle").value,
  };
  try {
    await fetch(`/api/student/${encodeURIComponent(state.studentId)}/onboarding`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
  } catch (_) {}
  $("onboardModal")?.classList.add("hidden");
  if (typeof toast === "function") toast(`✨ <span class="t-grad">Training tailored to you.</span>`);
}

// ── Init ───────────────────────────────────────────────────────
(async () => {
  applyTheme(document.documentElement.getAttribute("data-theme") || "light");
  buildStarterGrid();
  const SUBS = [
    "Ask anything about ML systems engineering.",
    "Your AI study buddy for ML systems 🚀",
    "No question is too small — let's dig in.",
    "Learn it, quiz it, master it. 🎯",
    "Turn confusion into mastery, one question at a time.",
  ];
  const subEl = document.querySelector(".welcome-sub");
  if (subEl) subEl.textContent = SUBS[Math.floor(Math.random() * SUBS.length)];

  document.querySelectorAll(".auth-tab").forEach((t) => t.addEventListener("click", () => setAuthMode(t.dataset.tab)));
  $("authForm").addEventListener("submit", submitAuth);
  $("logoutBtn")?.addEventListener("click", logout);
  setAuthMode("login");

  // Gate the app behind a session.
  try {
    const r = await fetch("/api/auth/me");
    if (r.ok) onLoggedIn((await r.json()).username);
    else $("authGate").classList.remove("hidden");
  } catch (_) { $("authGate").classList.remove("hidden"); }
})();
