# 自动生成的评估摘要

- Schema: `1.0`
- Source: `python scripts/run_eval_showcase.py`

| Suite | Metric | Result | n | Evidence | Standard |
|---|---|---:|---:|---|---|
| Agent 整体 | Task Success Rate / Pass¹ | 100.0% (95% CI 92.9–100.0%) | 50 | deterministic | [τ-bench Pass¹](https://arxiv.org/abs/2406.12045) |
| Agent 整体 | Provider Canary Pass³ | NOT RUN | — | llm_judge | [τ-bench Pass^k](https://arxiv.org/abs/2406.12045) |
| Agent 整体 | Router Accuracy | 100.0% (95% CI 92.7–100.0%) | 49 | deterministic | [Standard multiclass classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| Agent 整体 | Router Macro Precision | 100.0% (95% CI 76.9–100.0%) | 49 | deterministic | [Standard multiclass classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| Agent 整体 | Router Macro Recall | 100.0% (95% CI 76.9–100.0%) | 49 | deterministic | [Standard multiclass classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| Agent 整体 | Router Macro-F1 | 100.0% (95% CI 76.9–100.0%) | 49 | deterministic | [Standard multiclass classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| Tool Calling | Tool Selection Accuracy | 100.0% (95% CI 79.6–100.0%) | 15 | deterministic | [BFCL-compatible local evaluation](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Tool Calling | AST Exact Match Accuracy | 100.0% (95% CI 79.6–100.0%) | 15 | deterministic | [BFCL-compatible local evaluation](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Tool Calling | Executable Accuracy | 100.0% (95% CI 43.9–100.0%) | 3 | deterministic | [BFCL-compatible local evaluation](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Tool Calling | Relevance Detection Accuracy | 93.3% (95% CI 70.2–98.8%) | 15 | deterministic | [BFCL-compatible local evaluation](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Tool Calling | Multi-turn State Accuracy | 100.0% (95% CI 43.9–100.0%) | 3 | deterministic | [BFCL-compatible local evaluation](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| Tool Calling | Multi-turn Response Accuracy | 100.0% (95% CI 43.9–100.0%) | 3 | deterministic | [BFCL-compatible local evaluation](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| RAG Retrieval | Precision@5 | NOT RUN | — | deterministic | [BEIR retrieval metric](https://openreview.net/pdf?id=wCu6T5xFjeJ) |
| RAG Retrieval | Recall@5 | NOT RUN | — | deterministic | [BEIR retrieval metric](https://openreview.net/pdf?id=wCu6T5xFjeJ) |
| RAG Retrieval | Recall@20 | NOT RUN | — | deterministic | [BEIR retrieval metric](https://openreview.net/pdf?id=wCu6T5xFjeJ) |
| RAG Retrieval | MRR@10 | NOT RUN | — | deterministic | [BEIR retrieval metric](https://openreview.net/pdf?id=wCu6T5xFjeJ) |
| RAG Retrieval | nDCG@10 | NOT RUN | — | deterministic | [BEIR retrieval metric](https://openreview.net/pdf?id=wCu6T5xFjeJ) |
| RAG Retrieval | Hit Rate@5 | NOT RUN | — | deterministic | [BEIR retrieval metric](https://openreview.net/pdf?id=wCu6T5xFjeJ) |
| RAG Generation 与 Citation | Context Precision | NOT RUN | — | llm_judge | [RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) |
| RAG Generation 与 Citation | Context Recall | NOT RUN | — | llm_judge | [RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) |
| RAG Generation 与 Citation | Faithfulness | NOT RUN | — | llm_judge | [RAGAS/ARES](https://aclanthology.org/2024.naacl-long.20/) |
| RAG Generation 与 Citation | Answer Relevance | NOT RUN | — | llm_judge | [RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) |
| RAG Generation 与 Citation | Citation Precision | NOT RUN | — | llm_judge | [ALCE](https://arxiv.org/abs/2305.14627) |
| RAG Generation 与 Citation | Citation Recall / Completeness | NOT RUN | — | llm_judge | [ALCE](https://arxiv.org/abs/2305.14627) |
| RAG Generation 与 Citation | Judge–Human Cohen’s κ | NOT RUN | — | llm_judge | [Cohen’s kappa](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| Scientific 闭环 | Data Quality Precision | 100.0% (95% CI 56.6–100.0%) | 5 | deterministic | [Standard classification metric](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| Scientific 闭环 | Data Quality Recall | 100.0% (95% CI 56.6–100.0%) | 5 | deterministic | [Standard classification metric](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| Scientific 闭环 | Data Quality F1 | 100.0% (95% CI 100.0–100.0%) | 8 | deterministic | [Standard classification metric](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| Scientific 闭环 | Growth Rate MAE | 2.732e-17 (95% CI 2.342e-17–3.123e-17) | 4 | deterministic | [Standard regression metric](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) |
| Scientific 闭环 | Growth Rate RMSE | 2.761e-17 (95% CI 2.343e-17–3.123e-17) | 4 | deterministic | [Standard regression metric](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) |
| Scientific 闭环 | Candidate Cause Hit@1 | 100.0% (95% CI 51.0–100.0%) | 4 | deterministic | [Standard ranking metric](https://openreview.net/pdf?id=wCu6T5xFjeJ) |
| Scientific 闭环 | Candidate Cause Hit@3 | 100.0% (95% CI 51.0–100.0%) | 4 | deterministic | [Standard ranking metric](https://openreview.net/pdf?id=wCu6T5xFjeJ) |
| Scientific 闭环 | Candidate Cause MRR | 100.0% (95% CI 100.0–100.0%) | 4 | deterministic | [Standard ranking metric](https://openreview.net/pdf?id=wCu6T5xFjeJ) |
| Scientific 闭环 | Simple Regret | 0 (95% CI 0–0) | 5 | deterministic | [Constrained optimization metric](https://arxiv.org/abs/1807.02811) |
| Scientific 闭环 | Constraint Violation Rate | 0.0% (95% CI 0.0–79.3%) | 1 | deterministic | [Constrained optimization metric](https://arxiv.org/abs/1807.02811) |
| Scientific 闭环 | Task Success Rate | 100.0% (95% CI 20.7–100.0%) | 1 | deterministic | [τ-bench task success](https://arxiv.org/abs/2406.12045) |
| User Memory | Extraction Exact Match | NOT RUN | — | llm_judge | [Standard extraction metric](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| User Memory | Extraction Precision | NOT RUN | — | llm_judge | [Standard extraction metric](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| User Memory | Extraction Recall | NOT RUN | — | llm_judge | [Standard extraction metric](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| User Memory | Extraction F1 | NOT RUN | — | llm_judge | [Standard extraction metric](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| User Memory | Memory Recall@5 | 70.0% (95% CI 40.0–100.0%) | 10 | deterministic | [Standard information retrieval metric](https://openreview.net/pdf?id=wCu6T5xFjeJ) |
| User Memory | Memory MRR@5 | 29.0% (95% CI 12.7–48.7%) | 10 | deterministic | [Standard information retrieval metric](https://openreview.net/pdf?id=wCu6T5xFjeJ) |
| User Memory | Data Leakage Rate | 0.0% (95% CI 0.0–16.1%) | 20 | deterministic | [Cross-tenant isolation failure rate](https://owasp.org/API-Security/editions/2023/en/0xa1-broken-object-level-authorization/) |
| User Memory | Proactive Safety Task Success Rate | NOT RUN | — | deterministic | [τ-bench-style task success](https://arxiv.org/abs/2406.12045) |
| Context 与效率 | Input Tokens | 2,160 (95% CI 1893–2454) | 8 | deterministic | [Tokenizer input token count](https://platform.openai.com/tokenizer) |
| Context 与效率 | Compression Ratio | 41.9% (95% CI 20.8–67.2%) | 8 | deterministic | [Compression ratio](https://en.wikipedia.org/wiki/Data_compression_ratio) |
| Context 与效率 | P50 Latency | 20.7 ms (95% CI 19.81–26.84) | 8 | deterministic | [Latency percentile](https://opentelemetry.io/docs/specs/otel/metrics/) |
| Context 与效率 | P95 Latency | 27.5 ms (95% CI 22–27.89) | 8 | deterministic | [Latency percentile](https://opentelemetry.io/docs/specs/otel/metrics/) |

> 本文件由 `scripts/run_eval_showcase.py` 生成；请勿手工修改分数或样本量。
