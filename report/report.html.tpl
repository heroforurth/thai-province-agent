<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Thai Province Agent: Capstone Report</title>
<style>
  @page { size: A4; margin: 17mm 16mm 17mm 16mm; }
  body { font: 10.4pt/1.45 "Segoe UI", Arial, sans-serif; color: #1d2329; }
  h1 { font-size: 20pt; margin: 0 0 2px; letter-spacing: -0.3px; }
  h2 { font-size: 13pt; margin: 18px 0 6px; padding-bottom: 3px; border-bottom: 1.5px solid #2a7d6b; color: #183f37; }
  h3 { font-size: 11pt; margin: 12px 0 4px; color: #23313b; }
  p { margin: 5px 0 7px; text-align: justify; }
  .sub { color: #56636e; margin: 0 0 4px; }
  .meta { font-size: 9.2pt; color: #56636e; margin-bottom: 10px; }
  .meta a { color: #1f6f5e; }
  table { border-collapse: collapse; width: 100%; margin: 6px 0 4px; font-size: 8.8pt; page-break-inside: avoid; }
  th, td { border: 1px solid #c9d2d8; padding: 3px 6px; text-align: left; vertical-align: top; }
  th { background: #eef3f5; font-weight: 600; }
  tr.best td { background: #e8f6f2; }
  .cap { font-size: 8.8pt; color: #56636e; margin: 2px 0 10px; }
  .fig { text-align: center; margin: 6px 0 2px; page-break-inside: avoid; }
  .fig img { width: 100%; max-width: 172mm; }
  code { font: 8.8pt Consolas, monospace; background: #f1f4f6; padding: 0 3px; border-radius: 3px; }
  ul { margin: 4px 0 8px; padding-left: 20px; } li { margin: 2px 0; }
  .box { border: 1px solid #c9d2d8; border-left: 4px solid #b5473a; padding: 6px 10px; margin: 8px 0; background: #fdf6f5; page-break-inside: avoid; }
  .ok { border-left-color: #2a7d6b; background: #f2faf7; }
  .q { font-style: italic; }
  .pb { page-break-before: always; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
</style>
</head>
<body>

<h1>Thai Province Agent</h1>
<p class="sub">Capstone report: Building &amp; Evaluating an LLM-Powered Application</p>
<p class="meta">Author: <b>{{author}}</b> · Repository: <a href="https://github.com/heroforurth/thai-province-agent">github.com/heroforurth/thai-province-agent</a> · Model: <code>gemma-4-E4B-it</code> via the class proxy · Embeddings: <code>bge-m3</code></p>

<h2>1. Problem and who it is for</h2>
<p>Facts about Thailand's 77 provinces are scattered across long Wikipedia articles (history, landmarks, festivals, name
meanings) and a separate statistics table (population, area, density, region). Students, travellers and teachers often need
one <i>checkable</i> fact, e.g. <span class="q">"Which Northern province is the most densely populated?"</span> or
<span class="q">"What is the population of the province with the Bridge over the River Kwai?"</span>. A plain chatbot answers
these confidently but often wrongly. Our closed-book baseline (the same model with no tools) invented a "Tawin Lake" as
Thailand's largest lake and claimed Bangkok is the only province with over a million people (the true count is 21).
The Thai Province Agent answers from two grounded sources and shows its evidence: numbered Wikipedia citations for
prose facts, and the SQL query plus result rows for numbers.</p>
<p>The corpus (77 province articles plus the "Provinces of Thailand" table, joined with the six-region grouping) is
outside the reference app's domain. It is committed to the repository together with a fetch script
(<code>scripts/fetch_corpus.py</code>).</p>

<h2>2. Design</h2>
<div class="fig">{{architecture}}</div>
<p class="cap">Figure 1. Architecture. The HTTP layer (<code>main.py</code>) only frames requests and SSE events. The logic
lives in <code>agent.py</code> and <code>rag.py</code>. Data access lives in <code>tools.py</code>, <code>retrieval.py</code>
and <code>sqldb.py</code>. <code>llm.py</code> is the only module that calls a model, which lets 48 tests run offline with a
fake model.</p>

<table>
<thead><tr><th style="width:25%">Course technique</th><th>How it is used, and why</th></tr></thead>
<tbody>
<tr><td>Structured output + schema validation</td><td>Every agent turn must be one JSON object validated by a Pydantic
<i>discriminated union</i> on <code>action</code> (<code>search_docs | run_sql | calculator | final_answer</code>). The proxy
ignores <code>response_format</code>, so we extract the first balanced JSON object, validate it, and on failure send the
validation error back for one repair turn. The union doubles as the tool allow-list: an unknown tool can never validate,
so it can never run.</td></tr>
<tr><td>RAG with citations</td><td>The articles are split into section-aware chunks (120 words, 30 words overlap), each
prefixed with a contextual header <code>Province &gt; Section</code>. Templated boilerplate repeated across 74 articles is
dropped. Answers cite <code>[n]</code>, and the final answer's citation numbers are checked against the sources actually
retrieved; invented numbers are dropped and counted.</td></tr>
<tr><td>Retrieval evaluation</td><td>Recall@k and MRR on 43 hand-labelled queries, with baselines, a chunk-size sweep and an
ablation (Section 4.1).</td></tr>
<tr><td>Agent loop with tools (agentic)</td><td>A loop of call → validate → run tool → feed back the result. Stop conditions:
a final answer, max 6 steps, an identical repeated call, or two invalid turns in a row (each followed by one forced final
turn, then a safe fallback). The model must call at least one tool before answering. Tool results are clipped and framed
as untrusted data.</td></tr>
<tr><td>Text-to-SQL (advanced, agentic)</td><td>Ranking and aggregate questions can't be answered by retrieval (no chunk
says "the 21 provinces over one million"), so the agent writes SQL. Four guards apply: a single SELECT/WITH statement
with a keyword deny-list, a read-only connection (<code>mode=ro</code>), an SQLite authorizer allowing only reads of the
<code>provinces</code> table, and a step limit plus a 50-row cap. The calculator parses with <code>ast</code> against an
allow-list and never uses <code>eval</code>.</td></tr>
<tr><td>Served interface</td><td>FastAPI <code>POST /api/ask</code> streams Server-Sent Events. The UI shows each tool call,
SQL result tables, the answer and clickable citations live. There is also a single-shot "Quick RAG" mode with token
streaming.</td></tr>
</tbody></table>
<p>Everything evaluated runs at <code>temperature=0</code>. The configuration (models, retriever, <code>top_k</code>,
<code>min_score</code>) and secrets come from environment variables, and <code>.env</code> is git-ignored.</p>

<h2>3. Evaluation design</h2>
<h3>3.1 Test sets (hand-built from the corpus)</h3>
<ul>
<li><b>Retrieval set</b>: 43 queries, each labelled with <i>(article, evidence span)</i>, e.g. <span class="q">"Where is
the highest mountain in Thailand?"</span> → <code>chiang-mai</code>, evidence "Doi Inthanon". 30 are <i>implicit</i> (the
province is not named) and 13 are <i>explicit</i>. A chunk counts as relevant if it comes from the labelled article
<b>and</b> contains the evidence span, so the same labels score every chunking fairly. Each query has one answer-bearing
passage, so Recall@k equals Hit@k. There are also 12 off-corpus queries for the <code>min_score</code> floor.</li>
<li><b>Answer set</b>: 34 questions: 14 <i>text</i> (from articles), 11 <i>sql</i> (from the table), 7 <i>multi-hop</i>
(both, sometimes with arithmetic) and 2 <i>unanswerable</i> (the correct behaviour is to refuse). Gold answers for sql and
multi-hop items were computed from the CSV, not typed by hand.</li>
</ul>
<h3>3.2 Metrics and baselines</h3>
<p><b>Retrieval:</b> Recall@1/3/5 and MRR@10. Baselines: naive fixed 200-word chunks with no header and no boilerplate
removal, plus each chunking idea ablated. <b>Answers:</b> deterministic accuracy. An answer is correct if it contains a gold
alias, or a number within the item's tolerance (e.g. ±1%). Unanswerable items count as correct only for a refusal. We then
<b>read every answer marked correct</b> and overrode false positives (listed with reasons in
<code>eval/data/manual_audit.jsonl</code>), and we report both the automatic and the audited score. Baselines: the
<b>closed-book</b> model (no retrieval, no tools) and <b>single-shot RAG</b> (top-5 retrieval, cited answer, no tools).</p>
<div class="box"><b>What could and couldn't be measured.</b> Midway through the evaluation, the class proxy's embedding
endpoint began failing every request ("401 … column t.tpd_limit does not exist", a proxy-side database error) or timing
out, for over an hour. All 1,851 chunk embeddings were built before the outage, but the query embeddings needed for the
dense and hybrid evaluation could not be obtained. <b>All retrieval and RAG/agent numbers below therefore use BM25
retrieval</b>, and the shipped default (<code>RETRIEVER=bm25</code>) matches what was measured. The dense/hybrid path and
the <code>min_score</code> sweep are implemented and unit-tested; <code>python -m eval.eval_retrieval</code> produces those
numbers from the cache once the endpoint works. The server also degrades to BM25 automatically when embeddings fail.</div>

<h2>4. Results</h2>
<h3>4.1 Retrieval (BM25, 43 labelled queries)</h3>
{{retrieval_main}}
<p class="cap">Table 1. Our section-aware chunking vs the naive fixed-window default.</p>
<p>The comparison is mixed, and we report it as such. The naive fixed-200 chunks rank the answer first more often
(R@1 0.814 vs 0.721; MRR 0.868 vs 0.802), mainly on <i>explicit</i> questions: a large chunk containing the province name
plus the answer wins on term frequency. Our section chunks tie on R@5 (0.907) but are better on the 30 <i>implicit</i>
questions (R@5 0.933 vs 0.867), which are the hard case this design targets. The agent consumes the top-4 results, so
recall in the top few slots matters more to it than rank 1, and we kept section chunks. With a stronger first-stage
ranker (hybrid or a re-ranker), we would revisit this.</p>
{{retrieval_sweep}}<p class="cap">Table 2. Chunk-size sweep (section chunks with header, overlap = size/4).</p>
{{retrieval_ablation}}<p class="cap">Table 3. Ablation of the two chunking ideas at size 120 (first row = our configuration).</p>
<div class="box ok"><b>Measured improvement over a naive default.</b> The contextual header is the clearest win.
Removing it drops R@5 from <b>0.907 to 0.837</b>, MRR from 0.802 to 0.764, and R@5 on explicit questions from 0.846 to
0.692. Without the header, mid-article chunks don't contain the province name the user typed. Size 120 gives the best R@5
in the sweep: 80-word chunks split evidence away from context (R@5 0.860), and 300-word chunks dilute it (0.884).
Boilerplate removal has no effect on this test set (MRR 0.798 → 0.802), which we report as a null result.</div>

<h3>4.2 End-to-end answers (34 questions)</h3>
{{answers_table}}
<p class="cap">Table 4. Answer accuracy by system and question type (BM25 retrieval for RAG and the agent). Audited
accuracy is the headline; "Automatic" is before removing matcher false positives. Wilson 95% intervals on audited
accuracy: closed-book 0.17–0.46, RAG 0.34–0.66, agent 0.81–0.98.</p>
{{agent_stats}}
<p class="cap">Table 5. Agent behaviour on the 34 questions.</p>
<p><b>The agent beats both baselines by a margin larger than the uncertainty</b>: audited accuracy is 0.941 vs 0.500
(single-shot RAG) vs 0.294 (closed-book), and the confidence intervals don't overlap. The per-type breakdown shows where the
gain comes from. Retrieval fixes <i>text</i> questions (closed-book 4/14 → RAG 12/14), but RAG can't do <i>sql</i> (2/11)
or <i>multi-hop</i> (1/7). No passage contains "the total population of the Southern region", so RAG correctly says
"the sources don't say". The agent routes these to SQL (11/11) and chains <code>search_docs → run_sql → calculator</code>
for multi-hop (7/7), e.g. <i>"How many times larger is the province containing Doi Inthanon than Phuket?"</i> →
search finds Chiang Mai, SQL returns 22,311 and 547 km², and the calculator gives 40.79. This costs latency (median 2.2 s
vs 0.4 s) and 1.29 tool calls per question. The model followed the JSON protocol well: one repair in 34 runs, and no
invented citation numbers.</p>

<h2>5. Failure analysis</h2>
<div class="box"><b>Failure 1: retrieval vocabulary mismatch (agent, t05).</b> <span class="q">"What does the name
Phitsanulok mean?"</span> The answer is in the article's first chunk, written as <i>lit. 'Vishnu's realm'</i>, with no word
"mean" or "name". BM25 instead ranks the <i>Toponymy</i> sections of Chonburi, Samut Songkhram and Nakhon Sawan first,
because those sections contain "name" and "meaning". The agent repeated the identical search, the repeated-call stop fired,
and the forced final turn honestly answered "I could not find the meaning…". This is the right behaviour (no
hallucination) but a miss. It is exactly the paraphrase case dense embeddings target, and the first thing to re-measure
once the embedding endpoint returns.</div>
<div class="box"><b>Failure 2: a correct-looking number that is wrong (agent, t12), caught only by auditing.</b> Asked for
the size of Kaeng Krachan National Park, the agent wrote that <i>Phetchaburi province</i> "has an area of 2,915 km²" and the
park covers "approximately 1,457.5 km²", having misread "covers nearly half of the province" and then used the calculator
on that misreading. In fact the <i>park</i> is 2,915 km². Our number matcher found "2,915" and scored it correct, so the
automatic accuracy (0.971) overstated the agent; the audited score is 0.941. A similar hedged RAG answer (two conflicting
areas for Nakhon Ratchasima) was also overridden. Lesson: number-contains metrics need a claim-level check (or an
LLM judge with a rubric) for numeric questions.</div>
<div class="box"><b>Failure 3: the baselines.</b> Closed-book answers are fluent and wrong on numbers: "Bangkok is the only
province exceeding one million" (true: 21), Phuket "approximately 400,000" (table: 430,000), the newest province is
"Mueang Khlong" (it is Bueng Kan, created 2011). Single-shot RAG got 15 of the 18 sql/multi-hop items wrong. 9 of those 15 were explicit
refusals ("the sources don't say"), which is a safe failure. The rest show a subtler problem: answering a <i>table</i>
question from <i>prose</i>. Phuket's population came back as "360,905 at the end of 2012" (a stale figure in the
article), and "the densest Northern province" came back as Phitsanulok, the only density figure among the retrieved
chunks. Routing these to SQL is what fixes them.</div>
<p><b>Where the evaluation is weak.</b> (i) The two <i>unanswerable</i> items don't separate the systems: even closed-book
refuses them, so refusal quality needs a larger adversarial set. (ii) 34 and 43 items are small; the intervals above are
wide for the baselines. (iii) I wrote both the questions and the labels, from the same corpus, so phrasing may favour
lexical retrieval; held-out questions from other people would be a fairer test. (iv) The table is one snapshot
(registered population, Dec 2024), while article text sometimes quotes other figures. Bangkok's article says 9.1 M (metro
estimate) vs the table's 5.46 M registered.</p>

<h2>6. Limitations and next steps</h2>
<ul>
<li><b>Measure dense and hybrid retrieval</b> (already implemented) and pick <code>min_score</code> from the in-corpus vs
off-corpus sweep once the proxy's embedding endpoint is stable. Then re-run the agent with hybrid retrieval, starting with
t05.</li>
<li><b>Better numeric scoring</b>: check that the number is attached to the right entity (claim-level), or add an
LLM-as-judge from a different model family with a fixed rubric, calibrated against the manual audit.</li>
<li><b>Bigger, independent test sets</b>: about 100 questions written by people other than the author, more unanswerable
and adversarial questions (e.g. prompt-injection text placed inside an article), and Thai-language questions (the BM25
tokenizer currently ignores Thai script).</li>
<li><b>Robustness</b>: the proxy produced intermittent spurious 401s even when healthy. The client retries with backoff,
the embedding cache persists batch by batch, and serving falls back to BM25. A local embedding model would remove this
single point of failure.</li>
</ul>

<h2>Appendix: reproducing the numbers</h2>
<p><code>pip install -r requirements.txt</code> · put the course key in <code>.env</code> ·
<code>python scripts/ingest.py</code> · <code>python -m pytest -q</code> (48 tests, offline) ·
<code>python -m eval.eval_retrieval --bm25-only</code> (Tables 1–3, no key needed) ·
<code>python -m eval.eval_answers --systems closed_book rag agent --retriever bm25</code> (Table 4) ·
<code>python -m eval.eval_answers --rescore</code> (re-score the saved predictions without calling the model) ·
<code>python report/build_report.py</code> (this PDF, tables generated from <code>eval/results/</code>).</p>
</body>
</html>
