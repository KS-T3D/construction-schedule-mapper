# Deploying the T3D Schedule Mapper backend to Render

## What this is
A Python (FastAPI) backend that:
- parses a client schedule (XER / CSV / XLSX) and your base T3D CSV
- matches activities to cells using **semantic embeddings** (free local model),
  with **learned mappings** layered on top for repeat runs
- flags out-of-range dates against the project window
- reads/writes your existing **Supabase** (orgs, projects, versions, learned_mappings)
- returns the mapping + a ready CSV in your exact format

Semantics run **every time**. Learning just makes later runs more accurate/cheaper —
approved mappings are applied directly, embeddings handle anything new.

---

## Step 1 — Put these files in a GitHub repo
Create a repo (e.g. `t3d-mapper-backend`) with:
```
main.py
requirements.txt
render.yaml
```
Push to GitHub.

## Step 2 — Create the Render service
1. Go to https://render.com, sign up (free), connect your GitHub.
2. **New +  →  Web Service  →** pick your repo.
3. Render auto-detects `render.yaml`. Confirm:
   - Runtime: Python
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Plan: **Free**
4. Click **Create Web Service**.

## Step 3 — Set environment variables (Render dashboard → your service → Environment)
```
SUPABASE_URL = https://svewwfvmfgjoysmenbwd.supabase.co
SUPABASE_KEY = <your Supabase publishable/anon key>
SEM_THRESHOLD = 0.42        (optional; lower = looser matches, higher = stricter)
```
Later, to switch from the free local model to OpenAI embeddings, add:
```
OPENAI_API_KEY = sk-...     (embeddings are pennies)
```
No code change needed — the backend auto-detects it.

## Step 4 — Deploy
Render builds and deploys. First build takes a few minutes (it downloads the
embedding model). When live you get a URL like:
```
https://t3d-schedule-mapper.onrender.com
```
Test it: open that URL — you should see `{"ok": true, ...}`.

## Step 5 — Host the frontend on GitHub Pages (free, always-on)
1. Put the frontend HTML in the same repo under /docs/index.html (or a /frontend folder).
2. Repo → Settings → Pages → Source: Deploy from branch → main → /docs → Save.
3. Your frontend URL: https://<youruser>.github.io/<repo>/
4. Open it → Settings → paste your Render backend URL → Save.

Frontend (GitHub Pages, always instant) calls Backend (Render, wakes on demand).

---

## Notes / honest caveats
- **Free tier sleeps** after ~15 min idle; the first request after waking takes
  ~30–60s (cold start). Fine for occasional use; upgrade to keep it always-on.
- **First build is slow** because it downloads the embedding model (~90MB) once.
- **Big files**: 9,844 cells × 5,000 activities embeds in seconds on the server —
  no browser freeze, no per-row API calls, no credit burn. Embeddings are the whole
  point: semantic matching at near-zero cost.
- **Supabase**: uses the same project you already set up. The backend connects to it
  directly; keep the schema you already ran.
