// Minimal client: POST /api/ask and read the Server-Sent Events stream with fetch().
// (EventSource only supports GET, so we parse the SSE frames ourselves.)
const $ = (id) => document.getElementById(id);
const EXAMPLES = [
  "Which province is home to the Bridge over the River Kwai, and what is its population?",
  "What does the name Phitsanulok mean?",
  "Which Northern province has the highest population density?",
  "How many times larger is Chiang Mai than Phuket by area?",
  "Where is Thailand's largest natural lake?",
  "How many provinces have more than one million people?",
];

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function renderSources(list) {
  $("sources").innerHTML = list.length
    ? list.map((s) => `<li value="${s.n}"><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a> · ${esc(s.section)}
        <span class="snip">${esc(s.snippet.slice(0, 220))}…</span></li>`).join("")
    : '<li class="muted">No document sources.</li>';
}

function linkCitations(text, sources) {
  const byN = Object.fromEntries(sources.map((s) => [s.n, s]));
  return esc(text).replace(/\[(\d+)\]/g, (m, n) => byN[n] ? `<a class="cite" href="${esc(byN[n].url)}" target="_blank" rel="noopener" title="${esc(byN[n].title)}">${n}</a>` : m);
}

function sqlTable(t) {
  if (!t || !t.columns) return "";
  const head = t.columns.map((c) => `<th>${esc(c)}</th>`).join("");
  const rows = t.rows.slice(0, 12).map((r) => `<tr>${r.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("");
  return `<table class="sql"><tr>${head}</tr>${rows}</table>`;
}

function addTrace(html) {
  const li = document.createElement("li");
  li.innerHTML = html;
  $("trace").appendChild(li);
}

let ragSources = [];
let ragText = "";

function handle(ev) {
  switch (ev.type) {
    case "step":
      addTrace(`<span class="tool">${esc(ev.action)}</span> ${ev.thought ? `<div class="thought">${esc(ev.thought)}</div>` : ""}<pre>${esc(JSON.stringify(ev.args))}</pre>`);
      break;
    case "observation":
      addTrace(`<span class="tool">↳ result (${esc(ev.tool)})</span>${ev.table ? sqlTable(ev.table) : `<pre>${esc(ev.text.slice(0, 700))}</pre>`}`);
      break;
    case "error":
      addTrace(`<span class="err">⚠ ${esc(ev.message)}</span>`);
      break;
    case "answer": {
      $("answer").classList.remove("muted");
      const cites = ev.citations || [];
      const sources = cites.map((s) => s);
      $("answer").innerHTML = linkCitations(ev.answer, sources);
      renderSources(sources);
      const bits = [`steps: ${ev.steps}`, `tool calls: ${ev.tool_calls}`, `stop: ${esc(ev.stop_reason)}`];
      if (ev.used_sql) bits.push("numbers: SQL table (Dec 2024)");
      if (ev.repairs) bits.push(`<span class="warn">JSON repairs: ${ev.repairs}</span>`);
      if (ev.invalid_citations?.length) bits.push(`<span class="warn">dropped invalid citations: ${ev.invalid_citations.join(", ")}</span>`);
      $("meta").innerHTML = bits.map((b) => `<span>${b}</span>`).join("");
      break;
    }
    case "sources":
      ragSources = ev.sources;
      renderSources(ev.sources);
      addTrace(`<span class="tool">retrieve</span> ${esc(ev.retriever)} · ${ev.sources.length} chunks · top cosine ${ev.top_cosine?.toFixed(3) ?? "—"}`);
      break;
    case "token":
      ragText += ev.text;
      $("answer").classList.remove("muted");
      $("answer").innerHTML = linkCitations(ragText, ragSources);
      break;
    case "done":
      $("meta").innerHTML = ev.refused ? '<span class="warn">refused: below min_score floor</span>' : "<span>single-shot RAG</span>";
      break;
    case "fatal":
      addTrace(`<span class="err">server error: ${esc(ev.message)}</span>`);
      break;
  }
}

async function ask(question, mode) {
  $("go").disabled = true;
  $("answer").textContent = "Thinking…";
  $("answer").classList.add("muted");
  $("meta").innerHTML = "";
  $("sources").innerHTML = "";
  $("trace").innerHTML = "";
  ragSources = []; ragText = "";
  try {
    const resp = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, mode }) });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const data = frame.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
        if (data) handle(JSON.parse(data));
      }
    }
  } catch (e) {
    $("answer").textContent = `Error: ${e.message}`;
  } finally {
    $("go").disabled = false;
  }
}

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("question").value.trim();
  if (q) ask(q, new FormData(e.target).get("mode"));
});

$("examples").innerHTML = EXAMPLES.map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join("");
$("examples").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) { $("question").value = e.target.textContent; $("question").focus(); }
});

fetch("/api/health").then((r) => r.json()).then((h) => {
  $("status").textContent = `${h.chunks} chunks · ${h.retriever} · min_score ${h.min_score}`;
  $("status").classList.add(h.retriever === "bm25" ? "bad" : "ok");
}).catch(() => { $("status").textContent = "server offline"; $("status").classList.add("bad"); });
