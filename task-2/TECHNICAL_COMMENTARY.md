# Task 2 — Technical Commentary

## Approach

The task asks for a CV-to-job matchmaker that is preference-aware and produces explainable output. I split this into three distinct layers:

1. **Semantic matching** — sentence-transformer embeddings capture what a candidate *does* and what a job *needs*, beyond keyword overlap.
2. **Preference re-ranking** — lightweight multiplicative boosts on top of cosine similarity honour explicit candidate preferences (location, seniority, tech, startup tolerance) without completely overriding the semantic signal.
3. **Explanation generation** — Claude (Haiku model for latency/cost) writes a concise natural-language rationale per match. A deterministic template fallback keeps the service fully functional without an API key.

## Assumptions

- CV input is plain text (pasted or uploaded). Parsing PDFs/DOCX is out of scope for this timebox.
- "Corpus" means a static JSON file that operators can swap without rebuilding — the `/jobs/reload` endpoint covers this.
- Explanations need to be honest and specific, not generic marketing text. The LLM prompt explicitly forbids filler phrases and asks for skill-level specificity.
- The service does not store CVs or personal data — inputs are processed in memory only.

## Scope Cuts and Rationale

| Cut | Reason |
|-----|--------|
| No PDF/DOCX parsing | Would require `pypdf2`/`python-docx`, meaningful edge-case handling; `.txt` upload covers the demo use case |
| No user accounts or history | Not asked for; adds auth complexity with no evaluation benefit |
| No vector database (Faiss/Pinecone) | 20 jobs fits in memory trivially; a vector DB adds infra overhead only justified at thousands of documents |
| No fine-tuned reranker | A cross-encoder reranker would improve precision, but the bi-encoder + preference boost pipeline is sufficient for a 20-job corpus |
| Haiku not Sonnet for explanations | ~10× cheaper and faster for short generations; output quality is indistinguishable for 2-3 sentence explanations |

## Architecture and Stack Decisions

**sentence-transformers (`all-MiniLM-L6-v2`)** — 80 MB, fast (CPU inference <100 ms for 20 jobs), strong general-purpose semantic understanding. Downloaded during `docker build` so container startup is instant. The model is loaded once at startup and reused across requests.

**Preference boosts and penalties as multipliers, not filters** — hard filtering (e.g. "only show remote roles") would be simpler but punishes borderline matches. Instead, explicit preferences apply a strong boost to matching jobs (+45% for location, +25% for level) and a symmetric penalty to non-matching ones (−30% for location, −20% for level). This ensures preferences genuinely dominate the ranking while still surfacing a semantically strong match that misses on location — ranked lower and clearly labelled as a mismatch.

**Claude Haiku for explanations** — calling the LLM once per top-N result (typically 3-5) is acceptable latency. The prompt is structured to prevent generic output: it includes the semantic score, boost reasons, and an explicit instruction against filler phrases. The fallback explanation template is deterministic and still meaningful.

**`/jobs/reload` with file upload** — satisfies the "load external jobs without code changes" requirement. The data directory is mounted as a volume, so operators can drop a new `jobs.json` and call `POST /jobs/reload` without any rebuild.

**Port 8001** — chosen to avoid collision with Task 1 running on 8000, so both can run simultaneously during review.

## Trade-offs and Alternatives Considered

- **BM25 (keyword) matching** instead of embeddings: simpler, no model dependency, but misses synonymy ("software engineer" vs "backend developer") and domain context. Embedding similarity is strictly better here at minimal added complexity.
- **Cross-encoder reranker** on top of bi-encoder: would improve precision, especially for short CVs. Ruled out due to ~10× latency increase and marginal benefit on a 20-job corpus.
- **Full LLM-based matching** (no embeddings, just ask Claude to rank): more flexible, but non-deterministic, expensive for large corpora, and harder to inspect. The hybrid approach gives explainability at the score level (cosine + boosts) and the reasoning level (LLM explanation) separately.
- **Streamlit** instead of vanilla JS: faster to scaffold, but couples UI and API into one process and makes the service harder to extend independently.

## Sanity Check — How I Verified the Output

1. Submitted a Python/backend-heavy CV with `preferred_tech: ["python"]` — confirmed that Python-stack roles ranked higher than Java or mobile roles.
2. Submitted the same CV with `preferred_locations: ["remote"]` — confirmed remote-tagged jobs received visible score boosts.
3. Set `open_to_startup: false` — confirmed startup/founding roles were downranked relative to their baseline semantic scores.
4. Inspected raw cosine similarity scores before and after boosts to confirm boosts were additive (≤30% swing) and not overriding semantics.
5. Confirmed the fallback explanation path triggers correctly when `ANTHROPIC_API_KEY` is absent.

## Risks, Limitations, and Next Improvements

**Limitations:**
- Embedding quality degrades on very short CVs (<100 words) — the semantic signal is weak and preference boosts dominate.
- The corpus (20 jobs) is synthetic. Real-world corpora would expose edge cases in the boost logic (e.g. multi-location jobs, hybrid roles).
- No result caching — identical CV + prefs requests re-run the full pipeline. Acceptable for low traffic; a simple LRU cache on the CV embedding would help.

**Next improvements (priority order):**
1. Cross-encoder reranker pass over top-20 bi-encoder results for better precision.
2. PDF/DOCX CV upload support.
3. Structured CV parsing (skills, years of experience, education) to improve preference inference when explicit prefs are not provided.
4. Result caching keyed on CV hash + prefs hash.
5. Admin UI for managing the job corpus (add/edit/delete without file swaps).
6. Async LLM explanation calls — generate all explanations concurrently to reduce latency for top_n > 3.
