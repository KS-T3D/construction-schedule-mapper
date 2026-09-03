# T3D Schedule Mapper — Deploy Guide (current)

Two parts, both free-tier capable:
- **Backend** (Python/FastAPI) → **Render**  — does parsing, embeddings, matching, Supabase I/O
- **Frontend** (static HTML) → **GitHub Pages** — the UI; calls the backend

Database: your existing **Supabase**.

---

## FILES

Backend repo (root):
```
main.py              # the app
requirements.txt     # dependencies
render.yaml          # Render config
schema-additions.sql # run ONCE in Supabase (app_config + cell_vectors tables)
```
Frontend:
```
docs/index.html      # the UI (served by GitHub Pages)
```

---

## ONE-TIME SETUP

### 1. Supabase schema
Run BOTH SQL files once in Supabase → SQL Editor (safe to re-run):
- the original schema (organizations, projects, schedule_versions, learned_mappings)
- **schema-additions.sql** (app_config, cell_vectors)

### 2. Backend on Render
1. Put the four backend files in a GitHub repo.
2. render.com → New + → Web Service → pick the repo → it reads render.yaml.
3. Set **Environment variables**:
   ```
   SUPABASE_URL = https://svewwfvmfgjoysmenbwd.supabase.co
   SUPABASE_KEY = <LEGACY anon JWT key, starts eyJ...  NOT the sb_publishable_ one>
   OPENAI_API_KEY = sk-...        <-- REQUIRED for large files (see note)
   SEM_THRESHOLD = 0.42           (optional)
   ```
4. Create Web Service → wait for build → you get a URL like
   `https://construction-schedule-mapper.onrender.com`
5. Open that URL → should show `{"ok":true,"embedder":"openai","supabase":true}`.
   - `embedder` MUST say **openai**. If it says `local`, the OpenAI key isn't being read.

### 3. Frontend on GitHub Pages
1. In the SAME repo, put the UI at `docs/index.html`.
2. Edit the top of index.html's script and set `DEFAULT_BACKEND` to your Render URL
   (already set to construction-schedule-mapper.onrender.com).
3. Repo → Settings → Pages → Deploy from branch → main → /docs → Save.
4. Your UI URL: `https://<youruser>.github.io/<repo>/`

---

## IMPORTANT: why OPENAI_API_KEY is required

Render's free tier has **512MB RAM**. The free local embedding model loads PyTorch
(~350MB) and, on large files, the whole job exceeds 512MB → the service crashes
("Ran out of memory"). Using **OpenAI embeddings** moves that compute off your server
(embeddings run on OpenAI's side), so the box stays light. Embeddings are cheap
(pennies for thousands of activities). The key is stored either as a Render env var
(reliable) or set once via the app's Settings (saved to Supabase, shared by all users).

The backend also now **streams cells in chunks (float16)** so peak memory stays ~250MB
even for ~10k-cell files — this plus OpenAI keeps it under 512MB. If you ever process
much larger files or many concurrent users and still hit limits, upgrade Render to the
$7/mo Starter tier (2GB) — no code change needed.

---

## SUPPORTED FILES
Client schedule: **.xer .csv .xls .xlsx .mpp**
(.mpp needs Java + MPXJ on the server; if Java isn't present it tells you to export
the MPP as XER/XML instead. XER/CSV/XLSX work everywhere.)
Base T3D file: **.csv** (your standard export with id, structure, zone, category, stage, dates).

---

## HOW IT WORKS (quick)
- Semantic matching via embeddings (separate location + work vectors).
- Location normalization (L2→Level 2, Z4→Zone 4, etc.) makes location matching reliable.
- **Learning**: approved mappings are stored per org (work vocab) + project (location);
  repeat runs apply them directly, embeddings handle only new activities.
- **Cached cell vectors**: base-file vectors are cached (keyed by a hash of
  structure/zone/category/stage — ignores dates), so repeat runs skip re-embedding cells.
- **Methods**: One-shot (full) or Sequential (confirm structures → then stages).
- **Save**: writes a version + learned mappings to Supabase, and downloads a CSV in your
  exact format (activity id first, MM/DD/YYYY). CSV always downloads even if the DB write
  fails.
- **Manage**: delete a project or a whole company (cascades all its data).

---

## LOCAL RUN (for testing on your own machine, no memory limit)
```
python3 -m venv venv && source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
export SUPABASE_URL=...   export SUPABASE_KEY=...   export OPENAI_API_KEY=...
uvicorn main:app --host 0.0.0.0 --port 8000
```
Then serve the frontend over http (not file://) so it can call localhost:
```
python3 -m http.server 5500      # in the folder with index.html
```
Open http://localhost:5500/index.html → Settings → Backend URL = http://localhost:8000
