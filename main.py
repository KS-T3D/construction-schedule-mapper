"""
T3D Schedule Mapper — backend (FastAPI, for Render).

Flow:
  /map   : parse client schedule + base cells -> semantic match (embeddings)
           + learned-mapping overrides -> return proposed mappings + diff + flags
  /save  : persist approved version + learned mappings to Supabase

Embeddings: free local sentence-transformers by default.
            Swap to OpenAI text-embedding-3-small by setting OPENAI_API_KEY (hook below).

Semantics run EVERY time. Learning (approved mappings) is layered on top as an
accelerator: a previously-approved activity->stage is applied directly; embeddings
handle everything new.
"""
from __future__ import annotations
import os, io, csv, re, json
from datetime import datetime, date
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# ---- Supabase ----
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()   # use the legacy anon JWT (eyJ...), not sb_publishable_
sb: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as _e:
        print("Supabase init failed:", _e)   # don't crash the app; /map still works, /save won't
        sb = None

# ---- embeddings (free local by default) ----
_EMB = None
def get_embedder():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer("all-MiniLM-L6-v2")   # small, fast, free
    return _EMB

def embed(texts: list[str]) -> np.ndarray:
    # OpenAI hook: if OPENAI_API_KEY set, use text-embedding-3-small instead.
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        import requests
        out = []
        for i in range(0, len(texts), 256):
            batch = texts[i:i+256]
            r = requests.post("https://api.openai.com/v1/embeddings",
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": "text-embedding-3-small", "input": batch}, timeout=60)
            r.raise_for_status()
            out.extend([d["embedding"] for d in r.json()["data"]])
        return np.array(out, dtype=np.float32)
    m = get_embedder()
    # smaller batch to keep peak memory low on Render free tier (512MB)
    return np.array(m.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True), dtype=np.float32)

def cosine_topk(query_vecs: np.ndarray, target_vecs: np.ndarray, k=3):
    # normalize
    q = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-9)
    t = target_vecs / (np.linalg.norm(target_vecs, axis=1, keepdims=True) + 1e-9)
    # chunk the query rows so we never allocate a full (n_query x n_target) matrix at once
    n = q.shape[0]
    idx_all = np.empty((n, k), dtype=np.int32)
    sims_all = np.empty((n, k), dtype=np.float32)
    CH = 512
    for i in range(0, n, CH):
        block = q[i:i+CH] @ t.T                       # (<=512, n_target)
        part = np.argsort(-block, axis=1)[:, :k]
        idx_all[i:i+block.shape[0]] = part
        for r in range(block.shape[0]):
            sims_all[i+r] = block[r, part[r]]
    return idx_all, sims_all

# ---- date helpers ----
def parse_date(s):
    if not s: return None
    s = str(s).strip().split(" ")[0].split("T")[0]
    for f in ("%Y-%m-%d","%m/%d/%Y","%d-%m-%Y","%d/%m/%Y","%m/%d/%y"):
        try: return datetime.strptime(s, f).date()
        except ValueError: continue
    return None
def fmt_csv(d): return d.strftime("%m/%d/%Y") if d else ""
def iso(d): return d.strftime("%Y-%m-%d") if d else ""

# ---- XER parser (zero-dep) ----
def parse_xer(text: str):
    tables, cur, fields = {}, None, []
    for line in text.split("\n"):
        c = line.rstrip("\r").split("\t")
        if c[0] == "%T": cur = c[1]; tables[cur] = {"rows": []}
        elif c[0] == "%F": fields = c[1:]; tables[cur]["fields"] = fields
        elif c[0] == "%R" and cur:
            tables[cur]["rows"].append(dict(zip(fields, c[1:])))
    T = tables.get("TASK", {}).get("rows", [])
    wbs = {w["wbs_id"]: (w.get("wbs_name",""), w.get("parent_wbs_id"))
           for w in tables.get("PROJWBS", {}).get("rows", [])}
    def wpath(wid):
        p, guard = [], 0
        while wid in wbs and guard < 40:
            name, parent = wbs[wid]; p.insert(0, name); wid = parent; guard += 1
        return [x for x in p if x]
    acts = []
    for i, t in enumerate(T):
        acts.append({
            "activity_id": t.get("task_code") or t.get("task_id"),
            "row_no": i + 1,
            "name": t.get("task_name", ""),
            "wbs": " > ".join(wpath(t.get("wbs_id"))),
            "start": parse_date(t.get("target_start_date") or t.get("early_start_date") or t.get("act_start_date")),
            "finish": parse_date(t.get("target_end_date") or t.get("early_end_date") or t.get("act_end_date")),
        })
    return acts

# ---- tabular parser (csv/xlsx) ----
def guess(cols, *cands):
    low = {c.lower().strip(): c for c in cols}
    for cand in cands:
        if cand in low: return low[cand]
    for cand in cands:
        for k, orig in low.items():
            if cand in k: return orig
    return None

def parse_tabular(rows: list[dict]):
    if not rows: return []
    cols = list(rows[0].keys())
    cId = guess(cols, "activity id","activity_id","id","task id","code")
    cName = guess(cols, "activity name","name","task name","description","activity")
    cS = guess(cols, "start","planned start","start date","early start")
    cF = guess(cols, "finish","planned finish","end","end date","early finish")
    cW = guess(cols, "wbs","wbs path","wbs name")
    out = []
    for i, r in enumerate(rows):
        out.append({
            "activity_id": str(r.get(cId, f"row_{i+1}")).strip() if cId else f"row_{i+1}",
            "row_no": i + 1,
            "name": str(r.get(cName, "")).strip() if cName else "",
            "wbs": str(r.get(cW, "")).strip() if cW else "",
            "start": parse_date(r.get(cS)) if cS else None,
            "finish": parse_date(r.get(cF)) if cF else None,
        })
    return out

def load_base_cells(text: str):
    rows = list(csv.DictReader(io.StringIO(text)))
    cells = []
    for r in rows:
        sk = next((k for k in r if k.lower().startswith("startdate")), None)
        ek = next((k for k in r if k.lower().startswith("enddate")), None)
        wa = next((k for k in r if re.search(r"wbs.?activity.?id", k, re.I)), None)
        if not r.get("category") or not r.get("stage"): continue
        cells.append({
            "id": (r.get("id") or "").strip(),
            "scope": (r.get("scope") or "").strip(),
            "structure": (r.get("structure") or "").strip(),
            "zone": (r.get("zone") or "").strip(),
            "category": (r.get("category") or "").strip(),
            "stage": (r.get("stage") or "").strip(),
            "start": parse_date(r.get(sk)) if sk else None,
            "end": parse_date(r.get(ek)) if ek else None,
            "wbs_activity_id": (r.get(wa) or "").strip() or None if wa else None,
        })
    return cells

def level_of(structure): return structure.split(">")[-1].strip()

# ---- location normalization (deterministic; handles L2 -> Level 2, Z4 -> Zone 4) ----
# Embeddings are unreliable at abbreviations; this fixes the location axis explicitly.
def normalize_location(text: str) -> str:
    t = " " + (text or "").lower() + " "
    # levels: L2, Lvl2, L-02, Level2, LV2, 2nd floor  ->  "level 2"
    t = re.sub(r"\b(?:l|lv|lvl|level)[\s\-]?0*(\d{1,2})\b", r" level \1 ", t)
    t = re.sub(r"\b0*(\d{1,2})(?:st|nd|rd|th)?\s+floor\b", r" level \1 ", t)
    t = re.sub(r"\bbasement\b|\bb1\b|\blower level\b", " basement ", t)
    t = re.sub(r"\bpenthouse\b|\bph\b", " penthouse ", t)
    t = re.sub(r"\bmezzanine\b|\bmezz\b", " mezzanine ", t)
    # zones: Z4, Zn4, Zone-4, Area A -> "zone 4" / "area a"
    t = re.sub(r"\b(?:z|zn|zone)[\s\-]?0*(\d{1,2})\b", r" zone \1 ", t)
    t = re.sub(r"\barea[\s\-]?([a-z0-9]{1,3})\b", r" area \1 ", t)
    return re.sub(r"\s+", " ", t).strip()

# ---- app ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.options("/{rest_of_path:path}")
async def preflight(rest_of_path: str):
    from fastapi.responses import Response
    return Response(status_code=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    })

from fastapi.responses import JSONResponse
from starlette.requests import Request as _Req
@app.exception_handler(Exception)
async def any_error(request: _Req, exc: Exception):
    # always return CORS headers even on error, so the browser shows the real message
    return JSONResponse(status_code=500, content={"error": str(exc)},
                        headers={"Access-Control-Allow-Origin": "*"})

@app.get("/")
def health(): return {"ok": True, "embedder": "openai" if os.environ.get("OPENAI_API_KEY") else "local", "supabase": bool(sb)}

def read_upload_text(f: UploadFile) -> str:
    raw = f.file.read()
    try: return raw.decode("utf-8-sig")
    except UnicodeDecodeError: return raw.decode("latin-1", errors="ignore")

def parse_client_file(f: UploadFile):
    name = (f.filename or "").lower()
    if name.endswith(".xer"):
        return parse_xer(read_upload_text(f))
    if name.endswith(".csv"):
        return parse_tabular(list(csv.DictReader(io.StringIO(read_upload_text(f)))))
    if name.endswith(".xls") or name.endswith(".xlsx"):
        import openpyxl
        f.file.seek(0)
        wb = openpyxl.load_workbook(io.BytesIO(f.file.read()), read_only=True)
        ws = wb.active
        rows = list(ws.values); header = [str(h) for h in rows[0]]
        dicts = [dict(zip(header, r)) for r in rows[1:]]
        return parse_tabular(dicts)
    raise HTTPException(400, "Unsupported file type (xer/csv/xlsx only)")

def project_window(project_id: Optional[str]):
    if not (sb and project_id): return None, None
    try:
        d = sb.table("projects").select("*").eq("id", project_id).single().execute().data
        ps = parse_date(d.get("start_date")) if d else None
        pe = None
        if ps and d.get("duration_months"):
            m = int(d["duration_months"]); y = ps.year + (ps.month - 1 + m)//12; mo = (ps.month - 1 + m)%12 + 1
            pe = date(y, mo, min(ps.day, 28))
        return ps, pe
    except Exception:
        return None, None

def load_learned(org_id, project_id):
    work = {}
    if not sb or not org_id: return work
    try:
        rows = sb.table("learned_mappings").select("*").eq("org_id", org_id).eq("scope","org").eq("target_kind","work").execute().data
        for r in rows: work[r["pattern"]] = (r["category"], r["stage"])
    except Exception: pass
    return work

def pattern_of(name):
    toks = [w for w in re.sub(r"[^a-z0-9 ]"," ",(name or "").lower()).split() if len(w) > 2 and not w.isdigit()]
    return " ".join(toks[:5])

SEM_THRESHOLD = float(os.environ.get("SEM_THRESHOLD", "0.42"))

def _rollup(a_list):
    starts = [a["start"] for a in a_list if a["start"]]
    finishes = [a["finish"] for a in a_list if a["finish"]]
    return (min(starts) if starts else None, max(finishes) if finishes else None)

def _prep(client: UploadFile, base: UploadFile):
    acts = parse_client_file(client)
    base.file.seek(0)
    cells = load_base_cells(read_upload_text(base))
    if not acts: raise HTTPException(400, "No activities parsed from client file")
    if not cells: raise HTTPException(400, "No cells parsed from base file")
    return acts, cells

def _oor(ps, pe, s, e):
    for d in (s, e):
        if not d: continue
        if ps and d < ps: return True
        if pe and d > pe: return True
    return False

# structure/location match: which activities belong to a cell's structure+zone
def structure_score(cells, acts, threshold):
    # embed the LOCATION text (normalized) of cells and activities
    cell_loc = [normalize_location(f'{level_of(c["structure"])} {c["zone"]}') for c in cells]
    act_loc  = [normalize_location(f'{a["name"]} {a["wbs"]}') for a in acts]
    cv = embed(cell_loc); av = embed(act_loc)
    idx, sims = cosine_topk(cv, av, k=12)
    return idx, sims

# work match: within candidate activities, which match a cell's category+stage
def work_match(cell, cand_acts, threshold):
    if not cand_acts: return [], 0.0
    cell_txt = f'{cell["category"]} {cell["stage"]}'
    cv = embed([cell_txt]); av = embed([f'{a["name"]}' for a in cand_acts])
    cvn = cv/(np.linalg.norm(cv,axis=1,keepdims=True)+1e-9)
    avn = av/(np.linalg.norm(av,axis=1,keepdims=True)+1e-9)
    sims = (cvn @ avn.T)[0]
    order = np.argsort(-sims)
    top = float(sims[order[0]]) if len(order) else 0.0
    if top < threshold: return [], top
    keep = [cand_acts[j] for j in order if sims[j] >= top - 0.05]
    return keep, top

def build_result(cell, matched, rule, conf, ps, pe):
    ns, ne = _rollup(matched) if matched else (None, None)
    changed = bool((ns or ne) and (ns != cell["start"] or ne != cell["end"]))
    state = "changed" if (matched and changed) else ("unchanged" if matched else ("orphaned" if (cell["start"] or cell["end"]) else "unmapped"))
    return {
        "cell": {**cell, "start": iso(cell["start"]), "end": iso(cell["end"]), "level": level_of(cell["structure"])},
        "new_start": iso(ns), "new_end": iso(ne),
        "matched_ids": [a["activity_id"] for a in matched],
        "matched_names": [a["name"] for a in matched],
        "matched_rows": [a["row_no"] for a in matched],
        "rule": rule if matched else "none", "confidence": round(conf, 3),
        "state": state, "out_of_range": _oor(ps, pe, ns, ne),
    }

def acts_payload(acts):
    return [{"id":a["activity_id"],"name":a["name"],"row":a["row_no"],
             "start":iso(a["start"]),"finish":iso(a["finish"])} for a in acts]

# ---------- ONE-SHOT ----------
@app.post("/map")
async def map_endpoint(client: UploadFile = File(...), base: UploadFile = File(...),
                       org_id: str = Form(""), project_id: str = Form("")):
    acts, cells = _prep(client, base)
    ps, pe = project_window(project_id); learned = load_learned(org_id, project_id)
    by_id = {a["activity_id"]: a for a in acts}
    # combined location+work embedding (fast single pass)
    cell_texts = [f'{c["category"]} {c["stage"]} {normalize_location(level_of(c["structure"])+" "+c["zone"])}' for c in cells]
    act_texts  = [f'{a["name"]} {normalize_location(a["wbs"])}' for a in acts]
    cv = embed(cell_texts); av = embed(act_texts)
    idx, sims = cosine_topk(cv, av, k=6)
    results = []
    for ci, cell in enumerate(cells):
        matched, rule, conf = [], "none", 0.0
        if cell["wbs_activity_id"] and cell["wbs_activity_id"] in by_id:
            matched = [by_id[cell["wbs_activity_id"]]]; rule, conf = "id_match", 1.0
        else:
            lh = [a for a in acts if learned.get(pattern_of(a["name"])) == (cell["category"], cell["stage"])]
            if lh: matched, rule, conf = lh, "learned", 0.98
            else:
                cand = [(acts[int(idx[ci][r])], float(sims[ci][r])) for r in range(idx.shape[1]) if sims[ci][r] >= SEM_THRESHOLD]
                if cand:
                    top = cand[0][1]; matched = [a for a, s in cand if s >= top - 0.05]; rule, conf = "semantic", top
        results.append(build_result(cell, matched, rule, conf, ps, pe))
    return {"mappings": results, "stats": _stats(results, cells, acts, learned), "activities": acts_payload(acts)}

# ---------- SEQUENTIAL: STEP 1 (structure/location) ----------
@app.post("/map-structure")
async def map_structure(client: UploadFile = File(...), base: UploadFile = File(...),
                        org_id: str = Form(""), project_id: str = Form("")):
    acts, cells = _prep(client, base)
    # group cells by structure+zone; assign candidate activities per location group
    idx, sims = structure_score(cells, acts, SEM_THRESHOLD)
    groups = {}
    for ci, cell in enumerate(cells):
        key = f'{cell["structure"]}||{cell["zone"]}'
        if key not in groups:
            cand = [(acts[int(idx[ci][r])], float(sims[ci][r])) for r in range(idx.shape[1]) if sims[ci][r] >= SEM_THRESHOLD]
            groups[key] = {"structure": cell["structure"], "zone": cell["zone"], "level": level_of(cell["structure"]),
                           "candidate_activity_ids": [a["activity_id"] for a, s in cand[:40]],
                           "confidence": round(cand[0][1],3) if cand else 0.0}
    return {"structures": list(groups.values()), "activities": acts_payload(acts),
            "stats": {"structures": len(groups), "activities": len(acts)}}

# ---------- SEQUENTIAL: STEP 2 (stages within confirmed locations) ----------
class StageBody(BaseModel):
    org_id: str = ""
    project_id: str = ""
    # confirmed location -> list of activity_ids the user approved for that structure+zone
    location_assignments: dict   # {"structure||zone": [activity_id,...]}
    cells: list                  # base cells (from step 1 response, passed back)
    activities: list             # activities (from step 1 response, passed back)

@app.post("/map-stages")
async def map_stages(body: StageBody):
    ps, pe = project_window(body.project_id); learned = load_learned(body.org_id, body.project_id)
    by_id = {a["id"]: {"activity_id":a["id"],"name":a["name"],"row_no":a["row"],
                       "start":parse_date(a.get("start")),"finish":parse_date(a.get("finish"))} for a in body.activities}
    results = []
    for cell in body.cells:
        key = f'{cell["structure"]}||{cell["zone"]}'
        cand_ids = body.location_assignments.get(key, [])
        cand = [by_id[i] for i in cand_ids if i in by_id]
        matched, rule, conf = [], "none", 0.0
        lh = [a for a in cand if learned.get(pattern_of(a["name"])) == (cell["category"], cell["stage"])]
        if lh: matched, rule, conf = lh, "learned", 0.98
        else:
            matched, conf = work_match(cell, cand, SEM_THRESHOLD)
            rule = "semantic" if matched else "none"
        cell_norm = {**cell, "start": parse_date(cell.get("start")), "end": parse_date(cell.get("end")),
                     "wbs_activity_id": cell.get("wbs_activity_id")}
        results.append(build_result(cell_norm, matched, rule, conf, ps, pe))
    return {"mappings": results, "stats": _stats(results, body.cells, body.activities, learned)}

def _stats(results, cells, acts, learned):
    return {"cells": len(cells), "activities": len(acts),
            "matched": sum(1 for r in results if r["matched_ids"]),
            "changed": sum(1 for r in results if r["state"] == "changed"),
            "orphaned": sum(1 for r in results if r["state"] == "orphaned"),
            "out_of_range": sum(1 for r in results if r["out_of_range"]),
            "first_mapping": len(learned) == 0,
            "embedder": "openai" if os.environ.get("OPENAI_API_KEY") else "local"}

class SaveBody(BaseModel):
    org_id: str
    project_id: str
    source_name: str
    mappings: list   # each: {cell{...}, new_start, new_end, accepted, matched_names}

@app.post("/save")
async def save_endpoint(body: SaveBody):
    if not sb: raise HTTPException(500, "Supabase not configured on server")
    # version
    cells = []
    for m in body.mappings:
        use_new = m.get("accepted") and (m.get("new_start") or m.get("new_end"))
        c = m["cell"]
        cells.append({**{k: c.get(k) for k in ("id","scope","structure","zone","category","stage")},
                      "start": m["new_start"] if use_new else c.get("start"),
                      "end": m["new_end"] if use_new else c.get("end")})
    label = f'{datetime.utcnow().date().isoformat()} · {body.source_name}'
    sb.table("schedule_versions").insert({"project_id": body.project_id, "label": label,
                                          "cells": cells, "source_name": body.source_name}).execute()
    # learn (org work vocab + project location)
    learn = []
    for m in body.mappings:
        if not (m.get("accepted") and m.get("matched_names")): continue
        if m.get("rule") == "id_match": continue
        c = m["cell"]
        for nm in m["matched_names"]:
            p = pattern_of(nm)
            if not p: continue
            learn.append({"org_id": body.org_id, "project_id": None, "scope":"org","pattern":p,
                          "target_kind":"work","category":c["category"],"stage":c["stage"]})
            learn.append({"org_id": body.org_id, "project_id": body.project_id, "scope":"project","pattern":p,
                          "target_kind":"location","structure":c["structure"],"zone":c["zone"]})
    if learn:
        sb.table("learned_mappings").insert(learn).execute()
    # build CSV (activity id first)
    header = ["activity id","scope","structure","zone","category","stage","startDate(MM/DD/YYYY)","endDate(MM/DD/YYYY)"]
    lines = [",".join(header)]
    for m in body.mappings:
        c = m["cell"]; use_new = m.get("accepted") and (m.get("new_start") or m.get("new_end"))
        s = fmt_csv(parse_date(m["new_start"])) if use_new else fmt_csv(parse_date(c.get("start")))
        e = fmt_csv(parse_date(m["new_end"])) if use_new else fmt_csv(parse_date(c.get("end")))
        vals = [c.get("wbs_activity_id") or "", c.get("scope",""), c.get("structure",""), c.get("zone",""),
                c.get("category",""), c.get("stage",""), s, e]
        vals = ['"'+str(v).replace('"','""')+'"' if re.search(r'[",\n]', str(v)) else str(v) for v in vals]
        lines.append(",".join(vals))
    return {"saved": True, "learned": len(learn), "csv": "\ufeff" + "\r\n".join(lines)}
