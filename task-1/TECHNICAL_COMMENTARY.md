# Task 1 — Technical Commentary

## Approach

The task asks for a trained MNIST classifier with a feedback loop, served via a simple UI, all containerised. I kept the scope tight: one Python service, one SQLite file, one static HTML page — no external dependencies beyond PyTorch.

## Assumptions

- "Deploy" means runnable locally via Docker Compose; a publicly hosted version is optional (not attempted within the timebox).
- The feedback loop is a correction store, not an online learning loop. Collected labels are persisted and could be used for fine-tuning later, but re-training on each correction would be over-engineered for the scope.
- "Simple UI" means functional, not polished. A canvas + vanilla JS is sufficient and avoids a JS build pipeline entirely.

## Scope Cuts and Rationale

| Cut | Reason |
|-----|--------|
| No online fine-tuning on feedback | Out of scope for 4 hrs; persisted labels are the foundation for it |
| No authentication | Not required for a demo/review context |
| No GPU support in Docker | CPU inference is fast enough for 28×28 input; keeps the image lean |
| No image-upload endpoint | Canvas drawing covers the core UX; easy to add later |
| SQLite instead of Postgres | Zero-config, no extra container; more than adequate for this workload |

## Architecture and Stack Decisions

**FastAPI** — async, automatic OpenAPI docs, Pydantic validation. The obvious choice for a Python ML API in 2024+.

**PyTorch CNN** — a 2-conv-layer network (32→64 filters, ReLU, MaxPool, Dropout) is the standard baseline for MNIST. Achieves ~99% test accuracy in 5 epochs on CPU. Weights are ~800 KB.

**Training-on-first-boot** — the entrypoint script checks for `mnist_cnn.pt` and trains only when absent. The named Docker volume keeps weights across restarts, so the 2-minute cost is paid once.

**Vanilla JS / HTML Canvas** — no build step, no npm, no framework overhead. The canvas captures 280×280 pixel drawings which are base64-encoded and sent to the API. The server resizes to 28×28 and normalises to MNIST statistics.

**White-background inversion** — MNIST training data has white digits on a black background. The browser canvas defaults to black-on-black (transparent = black). The preprocessing pipeline detects background brightness and inverts if needed, so the model receives the expected input distribution regardless of drawing style.

**SQLite** — predictions and feedback are logged to a single table. The schema is deliberately minimal: `id`, `created_at`, `digit`, `confidence`, `image_prefix`, `correct_label`. `image_prefix` stores only the first 64 characters of the base64 string to keep the DB small.

## Trade-offs and Alternatives Considered

- **Streamlit** instead of vanilla JS: faster to build, but adds a Python dependency and makes the API/UI coupling less clean. Chose separation of concerns.
- **ONNX export** instead of raw PyTorch: slightly faster inference, but adds complexity and `onnxruntime` dependency. PyTorch CPU inference for a 28×28 input is already <10 ms.
- **Postgres + Alembic**: better for production, overkill for a demo. SQLite with direct `sqlite3` calls keeps the codebase minimal.

## Sanity Check — How I Verified the Output

1. Ran `train.py` locally and confirmed test accuracy ≥ 99% after 5 epochs.
2. Loaded the model and ran inference on 10 sample MNIST images — all correct.
3. Drew digits 0–9 in the browser canvas and confirmed predictions were reasonable.
4. Verified the inversion logic by drawing on a white background and confirming the model still predicted correctly.
5. Submitted corrections via the feedback UI and confirmed `correct_label` was stored in SQLite.

## Risks, Limitations, and Next Improvements

**Limitations:**
- Feedback labels are stored but not used for re-training. A fine-tuning pipeline (e.g., nightly batch update) would close this loop.
- The model is trained once at container boot. If the volume is lost, it retrains from scratch.
- No input validation beyond basic image decoding. A malicious large image could cause OOM.

**Next improvements (priority order):**
1. Feedback-driven fine-tuning endpoint or scheduled job.
2. Image upload (drag-and-drop) in addition to canvas drawing.
3. Model versioning — store multiple checkpoints and allow rollback.
4. Metrics dashboard: accuracy over time, feedback rate, class-level error rates.
5. GPU-enabled Docker image for faster training in production.
