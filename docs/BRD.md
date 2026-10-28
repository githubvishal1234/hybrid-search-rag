# Business Requirements Document (BRD)

**Product:** Hybrid Search RAG over Indian Finance Regulations
**Version:** 1.0 (as built and deployed)
**Date:** 2026-10-28
**Owner:** Vishal Kondaparthy
**Status:** Delivered; live at https://hybrid-rag-finance.streamlit.app/
**Related:** [FRD](FRD.md) (functional requirements), [Runbook](RUNBOOK.md) (operations)

---

## 1. Executive summary

Indian financial regulation is spread across long, dense PDFs: SEBI master circulars of several
hundred pages, the Union Budget and Finance Bill, IRDAI regulations. People who need one specific
answer ("What does section 148A provide?", "What is the TCS rate on LRS remittances for
education?") either search the PDFs by hand or ask a general chatbot that may invent an answer.

This product answers questions in plain English **only from the official documents**, cites the
document and page for every claim, and says so when the documents don't contain the answer. Its
retrieval quality is **measured**, not asserted: on a 74-question test set it finds the right page
in the top 5 results 86.7% of the time, versus 62.5% for the common "vector search only" approach.

## 2. Business context and problem

| Problem | Impact |
|---|---|
| Regulatory PDFs are long, bilingual (English/Hindi), table-heavy and frequently amended | Finding one rule or figure takes minutes to hours of manual searching |
| Keyword search (Ctrl+F) needs the exact wording; users don't know it | Paraphrased questions fail |
| General-purpose chatbots answer from memory | Answers can be outdated or invented, with no way to verify; unacceptable for compliance |
| Typical "chat with PDF" tools use semantic (vector) search only | They miss exact identifiers such as "section 148A", "Rule 114B", "Form 26AS", which dominate legal text |

**Opportunity:** a question-answering assistant that combines keyword and semantic search, cites
its sources at page level, and refuses rather than guesses.

**Secondary objective (portfolio):** demonstrate production-grade GenAI engineering (hybrid
retrieval, reranking, evaluation discipline, deployment) with real, defensible numbers for a
career move into GenAI engineering.

## 3. Business objectives

| ID | Objective | Measure |
|---|---|---|
| BO-1 | Answer regulatory questions faster than manual search | Answer in seconds instead of minutes |
| BO-2 | Make every answer verifiable | Every factual sentence cites document + page |
| BO-3 | Never present invented answers as fact | Refuse when the documents don't support an answer |
| BO-4 | Beat the standard vector-only approach, with evidence | Recall@5 and MRR measured against a vector-only baseline on a fixed test set |
| BO-5 | Be accessible without installation | Public web demo |
| BO-6 | Keep running cost at zero | Free hosting and free-tier/local LLMs |

## 4. Stakeholders and users

| Stakeholder | Interest |
|---|---|
| Compliance and legal analysts (primary user) | Fast, citable answers on SEBI/Budget/IRDAI rules |
| Financial advisors, CA/CS/CFA students | Look up rules, rates and figures with a page to verify |
| Recruiters and hiring managers (portfolio audience) | A working demo plus evidence of engineering judgment |
| Owner / developer | Maintainability, low cost, ability to add documents |

### User personas
- **Priya, compliance analyst:** needs the exact clause and page to quote in an internal note.
  Success: correct answer with a page reference she can open.
- **Rahul, CA student:** asks in his own words ("how much will the government borrow next year?").
  Success: the system understands paraphrases, not just exact wording.
- **Hiring manager:** has two minutes. Success: opens the link, asks a question, sees a cited
  answer and a results table that shows measured improvement.

## 5. Scope

### In scope (v1.0)
- English text of 27 public PDFs: 14 SEBI master circulars, 8 Union Budget 2026-27 documents,
  4 IRDAI regulations, 1 PRS summary of the Income-tax Bill 2025 (see `data/sources.csv`).
- Natural-language questions; answers with page-level citations; refusal when unsupported.
- Four retrieval modes for comparison: vector, BM25, hybrid, hybrid + rerank.
- Evaluation suite with a golden question set and published results.
- Web UI, REST API, Docker image, public deployment, operations documentation.
- Adding new PDFs through a documented, repeatable pipeline.

### Out of scope (v1.0)
- Legal or financial advice; answers are information retrieval from source documents.
- Hindi or other non-English content (filtered out on purpose).
- Documents other than PDF (Word/HTML must be converted to PDF).
- User accounts, chat history, multi-turn conversation.
- Automatic monitoring of regulator websites for new circulars.
- RBI documents (listed in sources but blocked to automated download; not indexed in v1.0).
- Guaranteed uptime or service levels for the free public demo.

## 6. Business requirements

| ID | Requirement | Priority | Status |
|---|---|---|---|
| BR-1 | Users can ask a question in plain English and receive an answer drawn only from the indexed documents | Must | Met |
| BR-2 | Every answer identifies the source document and page number for its claims | Must | Met |
| BR-3 | When the documents do not contain the answer, the system says so instead of guessing | Must | Met (refusal accuracy 0.90) |
| BR-4 | The system finds answers for both exact identifiers (section/rule/form numbers) and paraphrased questions | Must | Met (see section 8) |
| BR-5 | Retrieval quality is measured on a fixed test set and compared with a vector-only baseline | Must | Met |
| BR-6 | A public web link lets anyone try the system without installing anything | Must | Met |
| BR-7 | The system can be integrated by other applications through an API | Should | Met (REST API) |
| BR-8 | New documents can be added without code changes | Should | Met (PDF + one CSV row + 2 commands) |
| BR-9 | Running cost is zero for the demo | Should | Met (free hosting, free LLM tier or local LLM) |
| BR-10 | The system shows why it returned each result (transparency for reviewers) | Could | Met (retrieval debug panel) |
| BR-11 | Answers are produced within an acceptable wait for an interactive tool | Should | Met on deployment (~6-11 s); local CPU LLM is slow (~1-2 min) |

## 7. Constraints

- **Budget:** zero. Free hosting (Streamlit Community Cloud), free Gemini tier or local Ollama.
- **Free-tier limits:** the demo's LLM (`gemini-3.6-flash`) allows 20 requests/day; higher
  limits need a Flash-Lite model or paid key. Hugging Face Docker Spaces now require a paid plan.
- **Hardware:** CPU only (no GPU at runtime); limits reranker speed.
- **Data:** public documents only; some regulator sites block automated downloads.
- **Team:** single developer, ~3 weeks.

## 8. Success criteria (targets vs. results)

Measured on a 74-question golden set (64 answerable), full set, current defaults.

| Criterion | Target | Result | Met |
|---|---|---|---|
| Recall@5 better than vector-only | > 0.625 | 0.867 (hybrid + rerank) | Yes |
| Ranking quality (MRR) better than vector-only | > 0.499 | 0.707 | Yes |
| Tuning does not overfit the test set | Tune on dev half, report on test half | Done; test half hybrid + rerank 0.875 | Yes |
| Unanswerable questions refused | >= 0.80 | 0.90 (hybrid) | Yes |
| Citations point to a correct page | >= 0.75 | 0.828 (hybrid) | Yes |
| Public demo available | Live URL | hybrid-rag-finance.streamlit.app | Yes |
| Interactive response time on the demo | < 15 s | ~6-11 s observed | Yes |
| Running cost | 0 | 0 | Yes |
| Faithfulness measured by an independent judge | Reported | Self-graded only (0.92, not discriminating) | **Open** |
| Golden set human-verified | Spot-check done | Reviewed with an AI assistant; human spot-check pending | **Open** |

Honest finding: plain BM25 keyword search reached 0.859 Recall@5, close to the final system. The
benefit of the full pipeline over BM25 is mainly ranking quality (MRR 0.707 vs 0.665) and
robustness to paraphrase; over vector-only it is large (+24 points Recall@5).

## 9. Assumptions

- Users ask in English about content that exists in the indexed documents.
- The indexed document versions are the ones users care about (no automatic updates).
- Citation to a page is sufficient for verification; users will open the source to confirm.
- A free-tier LLM is acceptable for a demo; production use would use a paid key.

## 10. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM states a claim the cited page doesn't support | Medium | High | Prompt restricts to sources; invalid citation numbers dropped; faithfulness judge available; users verify via page citation |
| Regulations amended after indexing | High | High | Documented re-ingestion pipeline; date and source URL kept per document |
| Free LLM quota exhausted during a demo | Medium | Medium | Switch `LLM_MODEL` to a Flash-Lite model (tested); local Ollama fallback |
| Users treat answers as legal advice | Medium | High | Scope statement: retrieval of source text, not advice; every claim cited |
| Free hosting changes terms (as Hugging Face did) | Medium | Medium | Same code runs in Docker anywhere; deployment options documented |
| API keys leaked | Low | High | Keys only in `.env` / platform secrets; `.env` gitignored; rotation procedure in runbook |
| Evaluation set biased toward the system's own chunking | Medium | Medium | Page-level labels, ~30% hand-written questions, dev/test split |

## 11. Benefits

- **Time:** seconds per answer instead of manual searching of 100-500 page PDFs.
- **Trust:** every claim is traceable to a page; refusals instead of guesses.
- **Evidence:** a reproducible evaluation shows where each retrieval method wins and loses.
- **Cost:** zero to run as a demo; scales to paid LLMs/rerankers by configuration only.

## 12. Future phases (not committed)

| Phase | Scope |
|---|---|
| 1.1 | Independent faithfulness judge; human-verified golden set; demo video |
| 1.2 | RBI documents; OCR for image-only pages; higher-limit LLM for the demo |
| 2.0 | Hindi/multilingual documents (bge-m3), query rewriting, metadata filters (publisher, year), scheduled re-ingestion of new circulars, user feedback on answers |

## 13. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Product owner | Vishal Kondaparthy | | |
