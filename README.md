# Fake Image Detector

Upload an image and the system classifies it as **Real** or **Fake** with a
confidence score. A Streamlit frontend calls a FastAPI backend, which runs the
AI team's trained ConvNeXt model and returns a binary verdict.

## Project Structure

- `frontend/` — Streamlit application. `app.py` is the entry point; `components/`
  holds the UI state machine (uploader, empty, analysing, result and error
  states), `core/` holds configuration, the `Analyzer` contract, validation and
  session state, and `services/` holds the two analyzers — `api_analyzer.py`
  (real HTTP calls) and `mock_analyzer.py` (built-in demo). `styles/` and
  `assets/` carry presentation only. `.streamlit/config.toml` sets the theme and
  the 15 MB upload limit.
- `backend/` — FastAPI application. `app/api/routes.py` defines the endpoints,
  `app/core/` configuration and upload validation, `app/schemas/` the response
  contract, and `app/services/` the model loading and inference seam plus the
  model-class-to-verdict mapping. `tools/setup_model.py` downloads the runtime
  checkpoint from a release asset; `tools/prepare_checkpoint.py` reconstructs a
  loadable `.pth` from the AI team's delivered archive (fallback path). `tests/`
  contains the unit suite and the model integration suite.
- `ai_model/` — Supplied by the AI team. `predict.py` is their reference
  inference script, `best_accuracy_model/` is the delivered checkpoint archive,
  and `images/` holds sample images used for verification.

## Architecture / Data Flow

```
Streamlit frontend
  -> POST /analyze            (multipart/form-data, field "file")
FastAPI backend
  -> validate upload          (size, extension, MIME, decodability)
PyTorch ConvNeXt model
  -> model class -> verdict + confidence
  -> {"verdict": "real"|"fake", "confidence": 0.0-1.0}
frontend result card
```

`frontend/services/api_analyzer.py` performs the request with the `requests`
library. The backend validates the upload independently and does not trust the
client. `confidence` is a fraction in the range `0.0`–`1.0`; the frontend
formats it as a percentage for display.

There is no database, authentication, analysis history, or storage. Images are
held in Streamlit session state for the current session only and are not
persisted.

## Requirements

Python is required. The project was developed against Python 3.12.

- Backend dependencies: `backend/requirements.txt` (plus
  `backend/requirements-dev.txt` for pytest and httpx)
- Frontend dependencies: `frontend/requirements.txt`
- The AI team's inference dependencies: `ai_model/requirements.txt`

The backend and frontend use separate environments in the current setup. The
commands below assume the backend virtual environment at `backend/.venv`.

## Model Setup

The model is **not** committed to this repository. It is a ~544 MiB binary, and
putting it in normal Git history would bloat every clone permanently. Two
artifacts are therefore deliberately absent from the source repository:

- `backend/artifacts/best_accuracy_model.pth` — the runtime checkpoint the
  backend loads. Intentionally not committed.
- `ai_model/best_accuracy_model/` — the AI team's delivered checkpoint, an
  *extracted* PyTorch serialization archive. Intentionally not committed.

**The model cannot be regenerated from source code alone.** The trained weights
are themselves the artefact; there is no training pipeline, dataset or seed in
this repository. The two things versioned and distributed here are separate:

- **Source code** — versioned in Git, cloned normally.
- **Model artefact** — distributed as a GitHub **Release asset**, fetched by
  `backend/tools/setup_model.py`.

### Normal path: download the released checkpoint

This is the documented setup path for a new developer.

1. **Clone the repository.**

   ```
   git clone <repository-url>
   cd Fake_Image_Detector
   ```

2. **Obtain the released runtime model.** The intended distribution mechanism is
   a GitHub Release asset named `best_accuracy_model.pth`, together with its
   SHA-256 digest. The exact asset URL and digest are **not yet configured** —
   the Release has not been created. Once it exists, record both in this
   section and in the release notes.

3. **Run the setup script**, which downloads and verifies the asset into
   `backend/artifacts/best_accuracy_model.pth`, the path the backend already
   loads by default:

   ```
   cd backend
   .venv\Scripts\python.exe tools\setup_model.py --url "<release asset url>" --sha256 "<digest>"
   ```

   The URL and digest can be supplied through the environment instead, which is
   the form to record once the Release exists:

   ```
   set FID_MODEL_URL=<release asset url>
   set FID_MODEL_SHA256=<digest>
   .venv\Scripts\python.exe tools\setup_model.py
   ```

   The script streams the file to disk rather than loading it into memory,
   verifies it against the SHA-256 digest, and only then moves it into place
   atomically. A failed or corrupt download is discarded and never becomes the
   checkpoint, and an existing verified checkpoint is left alone instead of
   being re-downloaded. Re-running is safe; use `--force` to replace an existing
   checkpoint deliberately.

   `MODEL_PATH` in `backend/.env` can point at a checkpoint held elsewhere; see
   `backend/.env.example`.

### Fallback: use the AI team's extracted archive

Use this if the Release asset is unavailable but you have been given the
original archive by the AI team.

1. Obtain the original model artefact from the AI team and place it at
   `ai_model/best_accuracy_model/`.
2. Then run:

   ```
   cd backend
   .venv\Scripts\python.exe tools\prepare_checkpoint.py
   ```

`torch.load` cannot open an extracted directory, and PyTorch's zip reader
requires members to sit under a single top-level prefix, so the delivered
archive is not loadable as-is. `tools/prepare_checkpoint.py` resolves this
mechanically: it re-zips the directory byte for byte under the expected
prefix. No tensor is converted, re-ordered, re-typed or re-normalised, and the
AI team's folder is only ever read, never modified.

The two paths are independent. `setup_model.py` only fetches an already-built
`.pth` and never calls `prepare_checkpoint.py`.

## Complete Setup

From a fresh clone to a running system:

1. Clone the repository.
2. Install the backend dependencies (`pip install -r requirements-dev.txt` from
   `backend/`) and the frontend dependencies (`pip install -r requirements.txt`
   from `frontend/`).
3. Obtain the released runtime model and run
   `backend/tools/setup_model.py` as described above.
4. Start the backend.
5. Start the frontend.

Steps 4 and 5 are the commands in the next two sections.

## Running the Backend

```
cd F:\Fake_Image_Detector\backend
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Endpoints:

- `GET /health` — liveness probe. Reports that the API process is serving. It
  deliberately does not report model readiness, since loading the checkpoint to
  answer a probe would make the probe the slow part.
- `POST /analyze` — accepts a `multipart/form-data` upload in the field `file`
  and returns `{"verdict": "real" | "fake", "confidence": 0.0-1.0}`. Responds
  `400` if the upload is not a decodable image, `413` if it exceeds the size
  limit, `422` if there is no file part, `500` if inference fails, and `503` if
  the model could not be loaded.

Interactive API documentation is served at `/docs` while the backend is running.

## Running the Frontend

```
cd F:\Fake_Image_Detector\frontend
python -m streamlit run app.py
```

Then open <http://localhost:8501>.

Streamlit must be launched from `frontend/` because the `.streamlit`
configuration — the theme and the 15 MB upload limit — lives in
`frontend/.streamlit/config.toml`, which Streamlit resolves relative to the
working directory. Launching from the repository root will not pick it up.

## Configuration

The frontend reads these environment variables, defined in
`frontend/core/config.py`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FID_ANALYZER_BACKEND` | `api` | `api` uses the real backend; `mock` uses the built-in demo analyzer. |
| `FID_API_BASE_URL` | `http://localhost:8000` | Backend origin. No trailing slash required. |
| `FID_API_TIMEOUT_SECONDS` | `60` | Seconds to wait for the backend. |

**API mode is the default.** Mock mode remains available for frontend-only
development and testing: set `FID_ANALYZER_BACKEND=mock` before starting
Streamlit and the backend can be left stopped. Results produced in mock mode are
labelled as such, so they cannot be mistaken for real analysis.

The backend has its own optional configuration, read from `backend/.env`; see
`backend/.env.example` for the full list.

## Testing

Frontend:

```
cd F:\Fake_Image_Detector\frontend
python -m unittest discover -s tests -t .
```

Backend:

```
cd F:\Fake_Image_Detector\backend
.venv\Scripts\python.exe -m pytest                    # fast suite, integration excluded
.venv\Scripts\python.exe -m pytest -m integration     # loads the real checkpoint
.venv\Scripts\python.exe -m pytest tests\test_setup_model.py   # model download script only
```

`backend/pytest.ini` sets `addopts = -q -m "not integration"`, so the default
run excludes the tests that load the ~544 MiB checkpoint. Those tests are
marked `integration` and must be requested explicitly.

The model download tests (`tests/test_setup_model.py`) run entirely offline: the
release asset is replaced with in-process doubles of `urllib.request`, so no
network access and no 544 MiB fixture is required.

## Current Model Classification

The model is a 3-class classifier. The class names are read from the checkpoint
itself rather than hardcoded. The API contract is binary, so the three classes
are collapsed onto two in exactly one place,
`backend/app/services/model_class_map.py`:

- `full_synthetic` → `fake`
- `real` → `real`
- `tampered` → `fake` — **provisional**

The `tampered` → `fake` mapping is a **provisional binary collapse based on the
current project integration, not independently verified model semantics.** The
model distinguishes "never existed" from "was edited", and folding `tampered`
into `fake` conflates the two — an edited photograph of a real scene is
authentic in origin but altered. This is a product decision rather than a model
fact, and it is pending confirmation from the AI team. A checkpoint reporting a
class name outside this set causes a hard error rather than a guessed verdict.

## Important Notes

- The first real inference may take around **15–20 seconds**, because the
  model is loaded lazily on first use. The load happens once per process;
  subsequent requests do not repeat it. The frontend's default 60-second
  timeout accommodates this.
- **Model accuracy has not been independently benchmarked.** There is no
  labelled evaluation dataset in this repository, so no accuracy or precision
  figure is claimed here. The mapping in
  `backend/app/services/model_class_map.py` is also unconfirmed by the AI team.
- The API returns `confidence` in the range `0.0`–`1.0`.
