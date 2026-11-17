# Answer evaluation

34 questions: 14 text (articles), 11 sql (statistics table), 7 multi-hop (both), 2 unanswerable. Metric: deterministic accuracy (see eval/metrics.py), then hand-audited: false positives listed in eval/data/manual_audit.jsonl are overridden. Model temperature 0.

| system | audited overall | automatic | text | sql | multi | unanswerable | median latency (s) |
|---|---|---|---|---|---|---|---|
| closed_book | **0.294** (10/34) | 0.294 | 0.29 (4/14) | 0.18 (2/11) | 0.29 (2/7) | 1.00 (2/2) | 0.49 |
| rag | **0.500** (17/34) | 0.529 | 0.86 (12/14) | 0.18 (2/11) | 0.14 (1/7) | 1.00 (2/2) | 0.4 |
| agent | **0.941** (32/34) | 0.971 | 0.86 (12/14) | 1.00 (11/11) | 1.00 (7/7) | 1.00 (2/2) | 2.22 |

## Agent behaviour

- grounded answers (>=1 valid citation or SQL), answerable items: 0.97
- answers containing an invalid (hallucinated) citation number: 0
- mean tool calls per question: 1.29
- JSON repairs needed (total): 1
- stop reasons: {'final_answer': 33, 'forced:repeated_call': 1}
