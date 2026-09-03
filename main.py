"""T3D Schedule Mapper — backend v2 (FastAPI)."""
from __future__ import annotations
import os, io, csv, re, json, hashlib
from datetime import datetime, date
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.requests import Request as _Req
from pydantic import BaseModel
import numpy as np
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
sb: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try: sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e: print("Supabase init failed:", e); sb = None

def get_openai_key():
    if sb:
        try:
            r = sb.table("app_config").select("value").eq("key","openai_api_key").execute()
            if r.data and r.data[0].get("value"): return r.data[0]["value"]
        except Exception: pass
    return os.environ.get("OPENAI_API_KEY") or None

_EMB = None
def get_embedder():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB
def embedder_name(): return "openai" if get_openai_key() else "local"
def embed(texts):
    if not texts: return np.zeros((0,8), np.float32)
    key = get_openai_key()
    if key:
        import requests; out=[]
        for i in range(0,len(texts),2000):
            r=requests.post("https://api.openai.com/v1/embeddings",headers={"Authorization":f"Bearer {key}"},
                json={"model":"text-embedding-3-small","input":texts[i:i+2000]},timeout=90); r.raise_for_status()
            out.extend([d["embedding"] for d in r.json()["data"]])
        return np.array(out,np.float16)
    m=get_embedder()
    return np.array(m.encode(texts,batch_size=16,show_progress_bar=False,convert_to_numpy=True),np.float16)
def _norm(v):
    v=v.astype(np.float32); return (v/(np.linalg.norm(v,axis=1,keepdims=True)+1e-9)).astype(np.float16)
def cosine_topk(q,t,k=6):
    if q.shape[0]==0 or t.shape[0]==0: return np.zeros((q.shape[0],k),np.int32),np.zeros((q.shape[0],k),np.float32)
    q=_norm(q).astype(np.float32); t=_norm(t).astype(np.float32); n=q.shape[0]; k=min(k,t.shape[0])
    idx=np.empty((n,k),np.int32); sims=np.empty((n,k),np.float32); CH=512
    for i in range(0,n,CH):
        block=q[i:i+CH]@t.T; part=np.argsort(-block,axis=1)[:,:k]; idx[i:i+block.shape[0]]=part
        for r in range(block.shape[0]): sims[i+r]=block[r,part[r]]
    return idx,sims

def parse_date(s):
    if not s: return None
    s=str(s).strip().split(" ")[0].split("T")[0]
    for f in ("%Y-%m-%d","%m/%d/%Y","%d-%m-%Y","%d/%m/%Y","%m/%d/%y"):
        try: return datetime.strptime(s,f).date()
        except ValueError: continue
    return None
def fmt_csv(d): return d.strftime("%m/%d/%Y") if d else ""
def iso(d): return d.strftime("%Y-%m-%d") if d else ""

def normalize_location(text):
    t=" "+(text or "").lower()+" "
    t=re.sub(r"\b(?:l|lv|lvl|level)[\s\-]?0*(\d{1,2})\b",r" level \1 ",t)
    t=re.sub(r"\b0*(\d{1,2})(?:st|nd|rd|th)?\s+floor\b",r" level \1 ",t)
    t=re.sub(r"\bbasement\b|\bb1\b|\blower level\b"," basement ",t)
    t=re.sub(r"\bpenthouse\b|\bph\b"," penthouse ",t)
    t=re.sub(r"\bmezzanine\b|\bmezz\b"," mezzanine ",t)
    t=re.sub(r"\b(?:z|zn|zone)[\s\-]?0*(\d{1,2})\b",r" zone \1 ",t)
    t=re.sub(r"\barea[\s\-]?([a-z0-9]{1,3})\b",r" area \1 ",t)
    return re.sub(r"\s+"," ",t).strip()
def level_of(s): return s.split(">")[-1].strip()

def parse_xer(text):
    tables,cur,fields={},None,[]
    for line in text.split("\n"):
        c=line.rstrip("\r").split("\t")
        if c[0]=="%T": cur=c[1]; tables[cur]={"rows":[]}
        elif c[0]=="%F": fields=c[1:]; tables[cur]["fields"]=fields
        elif c[0]=="%R" and cur: tables[cur]["rows"].append(dict(zip(fields,c[1:])))
    T=tables.get("TASK",{}).get("rows",[])
    wbs={w["wbs_id"]:(w.get("wbs_name",""),w.get("parent_wbs_id")) for w in tables.get("PROJWBS",{}).get("rows",[])}
    def wp(wid):
        p,g=[],0
        while wid in wbs and g<40: n,par=wbs[wid]; p.insert(0,n); wid=par; g+=1
        return [x for x in p if x]
    return [{"activity_id":t.get("task_code") or t.get("task_id"),"row_no":i+1,"name":t.get("task_name",""),
             "wbs":" > ".join(wp(t.get("wbs_id"))),
             "start":parse_date(t.get("target_start_date") or t.get("early_start_date") or t.get("act_start_date")),
             "finish":parse_date(t.get("target_end_date") or t.get("early_end_date") or t.get("act_end_date"))} for i,t in enumerate(T)]

def parse_mpp(raw):
    try:
        import jpype, jpype.imports
        if not jpype.isJVMStarted(): jpype.startJVM()
        from net.sf.mpxj.reader import UniversalProjectReader
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mpp",delete=False) as f: f.write(raw); path=f.name
        proj=UniversalProjectReader().read(path); out=[]
        for i,task in enumerate(proj.getTasks()):
            if task is None or task.getName() is None: continue
            wp=[]; p=task.getParentTask()
            while p is not None and p.getName() is not None: wp.insert(0,str(p.getName())); p=p.getParentTask()
            s=task.getStart(); e=task.getFinish()
            out.append({"activity_id":str(task.getUniqueID()),"row_no":i+1,"name":str(task.getName()),"wbs":" > ".join(wp),
                        "start":parse_date(str(s)[:10]) if s else None,"finish":parse_date(str(e)[:10]) if e else None})
        return out
    except Exception as e:
        raise HTTPException(400,f"MPP needs Java+MPXJ on server. Error: {e}. Export MPP as XER/XML instead.")

def guess(cols,*cands):
    low={c.lower().strip():c for c in cols}
    for cand in cands:
        if cand in low: return low[cand]
    for cand in cands:
        for k,orig in low.items():
            if cand in k: return orig
    return None
def parse_tabular(rows):
    if not rows: return []
    cols=list(rows[0].keys())
    cId=guess(cols,"activity id","activity_id","id","task id","code"); cName=guess(cols,"activity name","name","task name","description","activity")
    cS=guess(cols,"start","planned start","start date","early start"); cF=guess(cols,"finish","planned finish","end","end date","early finish")
    cW=guess(cols,"wbs","wbs path","wbs name")
    return [{"activity_id":str(r.get(cId,f"row_{i+1}")).strip() if cId else f"row_{i+1}","row_no":i+1,
             "name":str(r.get(cName,"")).strip() if cName else "","wbs":str(r.get(cW,"")).strip() if cW else "",
             "start":parse_date(r.get(cS)) if cS else None,"finish":parse_date(r.get(cF)) if cF else None} for i,r in enumerate(rows)]
def load_base_cells(text):
    rows=list(csv.DictReader(io.StringIO(text))); cells=[]
    for r in rows:
        sk=next((k for k in r if k.lower().startswith("startdate")),None); ek=next((k for k in r if k.lower().startswith("enddate")),None)
        wa=next((k for k in r if re.search(r"wbs.?activity.?id",k,re.I)),None)
        if not r.get("category") or not r.get("stage"): continue
        cells.append({"id":(r.get("id") or "").strip(),"scope":(r.get("scope") or "").strip(),"structure":(r.get("structure") or "").strip(),
                      "zone":(r.get("zone") or "").strip(),"category":(r.get("category") or "").strip(),"stage":(r.get("stage") or "").strip(),
                      "start":parse_date(r.get(sk)) if sk else None,"end":parse_date(r.get(ek)) if ek else None,
                      "wbs_activity_id":(r.get(wa) or "").strip() or None if wa else None})
    return cells

def base_hash(cells):
    ident="|".join(f'{c["structure"]}#{c["zone"]}#{c["category"]}#{c["stage"]}' for c in cells)
    return hashlib.sha256(ident.encode()).hexdigest()
def get_cached_vectors(pid,chash,emb):
    if not (sb and pid): return None
    try:
        r=sb.table("cell_vectors").select("*").eq("project_id",pid).eq("content_hash",chash).eq("embedder",emb).execute()
        if r.data:
            row=r.data[0]; return (np.array(row["loc_vectors"],np.float32),np.array(row["work_vectors"],np.float32),row["cell_keys"])
    except Exception as e: print("vec read fail:",e)
    return None
def save_cached_vectors(pid,chash,emb,loc,work,keys):
    if not (sb and pid): return
    try:
        sb.table("cell_vectors").delete().eq("project_id",pid).eq("embedder",emb).execute()
        sb.table("cell_vectors").insert({"project_id":pid,"content_hash":chash,"embedder":emb,
            "loc_vectors":loc.tolist(),"work_vectors":work.tolist(),"cell_keys":keys}).execute()
    except Exception as e: print("vec write fail:",e)
def compute_cell_vectors(cells,pid):
    chash=base_hash(cells); emb=embedder_name(); keys=[c["id"] for c in cells]
    cached=get_cached_vectors(pid,chash,emb)
    if cached and cached[2]==keys: return cached[0],cached[1]
    loc=embed([normalize_location(f'{level_of(c["structure"])} {c["zone"]}') for c in cells])
    work=embed([f'{c["category"]} {c["stage"]}' for c in cells])
    save_cached_vectors(pid,chash,emb,loc,work,keys)
    return loc,work

def load_learned(org_id,pid):
    work={}
    if not (sb and org_id): return work
    try:
        for r in sb.table("learned_mappings").select("*").eq("org_id",org_id).eq("scope","org").eq("target_kind","work").execute().data:
            work[r["pattern"]]=(r["category"],r["stage"])
    except Exception: pass
    return work
def pattern_of(name):
    toks=[w for w in re.sub(r"[^a-z0-9 ]"," ",(name or "").lower()).split() if len(w)>2 and not w.isdigit()]
    return " ".join(toks[:5])

app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=False,allow_methods=["*"],allow_headers=["*"],expose_headers=["*"])
@app.options("/{p:path}")
async def preflight(p:str):
    return Response(status_code=204,headers={"Access-Control-Allow-Origin":"*","Access-Control-Allow-Methods":"*","Access-Control-Allow-Headers":"*"})
@app.exception_handler(Exception)
async def any_err(request:_Req,exc:Exception):
    return JSONResponse(status_code=500,content={"error":str(exc)},headers={"Access-Control-Allow-Origin":"*"})
import threading, uuid, traceback
JOBS = {}   # job_id -> {status, progress, stage, result, error}

def _run_map_job(job_id, acts, cells, org_id, project_id, mode, location_assignments=None):
    try:
        JOBS[job_id].update(status="running", stage="embedding activities", progress=0.05)
        ps, pe = project_window(project_id); learned = load_learned(org_id, project_id)
        by_id = {a["activity_id"]: a for a in acts}
        al = _norm(embed([normalize_location(f'{a["name"]} {a["wbs"]}') for a in acts])).astype(np.float32)
        aw = _norm(embed([a["name"] for a in acts])).astype(np.float32)
        JOBS[job_id].update(stage="embedding cells", progress=0.35)
        cell_loc, cell_work = compute_cell_vectors(cells, project_id)
        cl_all = _norm(cell_loc).astype(np.float32); cw_all = _norm(cell_work).astype(np.float32)
        learn_index = {}
        for a in acts:
            key = learned.get(pattern_of(a["name"]))
            if key: learn_index.setdefault(key, []).append(a)
        JOBS[job_id].update(stage="matching", progress=0.5)
        results = []; CH = 400; n = len(cells)
        for i in range(0, n, CH):
            chunk = cells[i:i+CH]
            comb = (0.5*(cl_all[i:i+CH]@al.T) + 0.5*(cw_all[i:i+CH]@aw.T))
            for r in range(len(chunk)):
                cell = chunk[r]; matched, rule, conf = [], "none", 0.0
                if cell["wbs_activity_id"] and cell["wbs_activity_id"] in by_id:
                    matched = [by_id[cell["wbs_activity_id"]]]; rule, conf = "id_match", 1.0
                elif (cell["category"], cell["stage"]) in learn_index:
                    matched, rule, conf = learn_index[(cell["category"], cell["stage"])], "learned", 0.98
                else:
                    row = comb[r]; order = np.argsort(-row)[:6]
                    cand = [(acts[int(j)], float(row[int(j)])) for j in order if row[int(j)] >= SEM]
                    if cand: top = cand[0][1]; matched = [a for a, sc in cand if sc >= top-0.05]; rule, conf = "semantic", top
                results.append(build_result(cell, matched, rule, conf, ps, pe))
            del comb
            JOBS[job_id].update(progress=0.5 + 0.45*min(1.0, (i+CH)/max(1,n)))
        JOBS[job_id].update(status="done", progress=1.0, stage="done",
                            result={"mappings": results, "stats": _stats(results, cells, acts, learned),
                                    "activities": acts_payload(acts)})
    except Exception as e:
        JOBS[job_id].update(status="error", error=f"{e}\n{traceback.format_exc()[:500]}")

@app.post("/map-async")
async def map_async(client: UploadFile = File(...), base: UploadFile = File(...),
                    org_id: str = Form(""), project_id: str = Form("")):
    acts, cells = _prep(client, base)
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "progress": 0.0, "stage": "queued", "result": None, "error": None}
    threading.Thread(target=_run_map_job, args=(job_id, acts, cells, org_id, project_id, "full"), daemon=True).start()
    return {"job_id": job_id}

@app.get("/job/{job_id}")
async def get_job(job_id: str):
    j = JOBS.get(job_id)
    if not j: raise HTTPException(404, "job not found")
    out = {"status": j["status"], "progress": round(j["progress"], 3), "stage": j["stage"]}
    if j["status"] == "done": out["result"] = j["result"]; JOBS.pop(job_id, None)  # free memory after fetch
    if j["status"] == "error": out["error"] = j["error"]; JOBS.pop(job_id, None)
    return out

@app.get("/")
def health(): return {"ok":True,"embedder":embedder_name(),"supabase":bool(sb)}

def read_text(f):
    raw=f.file.read()
    try: return raw.decode("utf-8-sig")
    except UnicodeDecodeError: return raw.decode("latin-1",errors="ignore")
def parse_client_file(f):
    name=(f.filename or "").lower()
    if name.endswith(".xer"): return parse_xer(read_text(f))
    if name.endswith(".csv"): return parse_tabular(list(csv.DictReader(io.StringIO(read_text(f)))))
    if name.endswith(".xls") or name.endswith(".xlsx"):
        import openpyxl; f.file.seek(0); wb=openpyxl.load_workbook(io.BytesIO(f.file.read()),read_only=True); ws=wb.active
        rows=list(ws.values); header=[str(h) for h in rows[0]]; return parse_tabular([dict(zip(header,r)) for r in rows[1:]])
    if name.endswith(".mpp"): f.file.seek(0); return parse_mpp(f.file.read())
    raise HTTPException(400,"Unsupported file (xer/csv/xlsx/mpp)")
def project_window(pid):
    if not (sb and pid): return None,None
    try:
        d=sb.table("projects").select("*").eq("id",pid).single().execute().data
        ps=parse_date(d.get("start_date")) if d else None; pe=None
        if ps and d.get("duration_months"):
            m=int(d["duration_months"]); y=ps.year+(ps.month-1+m)//12; mo=(ps.month-1+m)%12+1; pe=date(y,mo,min(ps.day,28))
        return ps,pe
    except Exception: return None,None
def _rollup(a):
    s=[x["start"] for x in a if x["start"]]; f=[x["finish"] for x in a if x["finish"]]
    return (min(s) if s else None,max(f) if f else None)
def _oor(ps,pe,s,e):
    for d in (s,e):
        if not d: continue
        if ps and d<ps: return True
        if pe and d>pe: return True
    return False
def build_result(cell,matched,rule,conf,ps,pe):
    ns,ne=_rollup(matched) if matched else (None,None)
    changed=bool((ns or ne) and (ns!=cell["start"] or ne!=cell["end"]))
    state="changed" if (matched and changed) else ("unchanged" if matched else ("orphaned" if (cell["start"] or cell["end"]) else "unmapped"))
    return {"cell":{**cell,"start":iso(cell["start"]),"end":iso(cell["end"]),"level":level_of(cell["structure"])},
            "new_start":iso(ns),"new_end":iso(ne),"matched_ids":[a["activity_id"] for a in matched],
            "matched_names":[a["name"] for a in matched],"matched_rows":[a["row_no"] for a in matched],
            "rule":rule if matched else "none","confidence":round(conf,3),"state":state,"out_of_range":_oor(ps,pe,ns,ne)}
def acts_payload(acts):
    return [{"id":a["activity_id"],"name":a["name"],"row":a["row_no"],"start":iso(a["start"]),"finish":iso(a["finish"])} for a in acts]
def _prep(client,base):
    acts=parse_client_file(client); base.file.seek(0); cells=load_base_cells(read_text(base))
    if not acts: raise HTTPException(400,"No activities parsed")
    if not cells: raise HTTPException(400,"No cells parsed")
    return acts,cells
def _stats(results,cells,acts,learned):
    return {"cells":len(cells),"activities":len(acts),"matched":sum(1 for r in results if r["matched_ids"]),
            "changed":sum(1 for r in results if r["state"]=="changed"),"orphaned":sum(1 for r in results if r["state"]=="orphaned"),
            "out_of_range":sum(1 for r in results if r["out_of_range"]),"first_mapping":len(learned)==0,"embedder":embedder_name()}
SEM=float(os.environ.get("SEM_THRESHOLD","0.42"))

@app.post("/map")
async def map_endpoint(client:UploadFile=File(...),base:UploadFile=File(...),org_id:str=Form(""),project_id:str=Form("")):
    acts,cells=_prep(client,base); ps,pe=project_window(project_id); learned=load_learned(org_id,project_id)
    by_id={a["activity_id"]:a for a in acts}
    # activity vectors (small set) — embedded fresh each run
    al=_norm(embed([normalize_location(f'{a["name"]} {a["wbs"]}') for a in acts])).astype(np.float32)
    aw=_norm(embed([a["name"] for a in acts])).astype(np.float32)
    # CELL vectors — from cache when base unchanged (big speedup on repeat runs)
    cell_loc,cell_work=compute_cell_vectors(cells,project_id)
    cl_all=_norm(cell_loc).astype(np.float32); cw_all=_norm(cell_work).astype(np.float32)
    learn_index={}
    for a in acts:
        key=learned.get(pattern_of(a["name"]))
        if key: learn_index.setdefault(key,[]).append(a)
    results=[]; CH=400; n=len(cells)
    for i in range(0,n,CH):
        chunk=cells[i:i+CH]
        comb=(0.5*(cl_all[i:i+CH]@al.T)+0.5*(cw_all[i:i+CH]@aw.T))
        for r in range(len(chunk)):
            cell=chunk[r]; matched,rule,conf=[],"none",0.0
            if cell["wbs_activity_id"] and cell["wbs_activity_id"] in by_id:
                matched=[by_id[cell["wbs_activity_id"]]]; rule,conf="id_match",1.0
            elif (cell["category"],cell["stage"]) in learn_index:
                matched,rule,conf=learn_index[(cell["category"],cell["stage"])],"learned",0.98
            else:
                row=comb[r]; order=np.argsort(-row)[:6]
                cand=[(acts[int(j)],float(row[int(j)])) for j in order if row[int(j)]>=SEM]
                if cand: top=cand[0][1]; matched=[a for a,s in cand if s>=top-0.05]; rule,conf="semantic",top
            results.append(build_result(cell,matched,rule,conf,ps,pe))
        del comb
    return {"mappings":results,"stats":_stats(results,cells,acts,learned),"activities":acts_payload(acts)}

@app.post("/map-structure")
async def map_structure(client:UploadFile=File(...),base:UploadFile=File(...),org_id:str=Form(""),project_id:str=Form("")):
    acts,cells=_prep(client,base)
    al=_norm(embed([normalize_location(f'{a["name"]} {a["wbs"]}') for a in acts])).astype(np.float32)
    groups={}; CH=400; n=len(cells)
    # only need one cell per unique structure||zone; dedupe first to save work
    seen={}; uniq=[]
    for c in cells:
        k=f'{c["structure"]}||{c["zone"]}'
        if k not in seen: seen[k]=True; uniq.append(c)
    for i in range(0,len(uniq),CH):
        chunk=uniq[i:i+CH]
        cl=_norm(embed([normalize_location(f'{level_of(c["structure"])} {c["zone"]}') for c in chunk])).astype(np.float32)
        sims=cl@al.T
        for r,cell in enumerate(chunk):
            row=sims[r]; order=np.argsort(-row)[:40]
            cand=[(acts[int(j)],float(row[int(j)])) for j in order if row[int(j)]>=SEM]
            key=f'{cell["structure"]}||{cell["zone"]}'
            groups[key]={"structure":cell["structure"],"zone":cell["zone"],"level":level_of(cell["structure"]),
                         "candidate_activity_ids":[a["activity_id"] for a,s in cand[:40]],"confidence":round(cand[0][1],3) if cand else 0.0}
        del cl,sims
    return {"structures":list(groups.values()),"activities":acts_payload(acts),"stats":{"structures":len(groups),"activities":len(acts)}}

class StageBody(BaseModel):
    org_id:str=""; project_id:str=""; location_assignments:dict; cells:list; activities:list
@app.post("/map-stages")
async def map_stages(body:StageBody):
    ps,pe=project_window(body.project_id); learned=load_learned(body.org_id,body.project_id)
    by_id={a["id"]:{"activity_id":a["id"],"name":a["name"],"row_no":a["row"],"start":parse_date(a.get("start")),"finish":parse_date(a.get("finish"))} for a in body.activities}
    results=[]
    for cell in body.cells:
        key=f'{cell["structure"]}||{cell["zone"]}'; cand=[by_id[i] for i in body.location_assignments.get(key,[]) if i in by_id]
        matched,rule,conf=[],"none",0.0
        lh=[a for a in cand if learned.get(pattern_of(a["name"]))==(cell["category"],cell["stage"])]
        if lh: matched,rule,conf=lh,"learned",0.98
        elif cand:
            cw=_norm(embed([f'{cell["category"]} {cell["stage"]}'])); aw=_norm(embed([a["name"] for a in cand]))
            sims=(cw@aw.T)[0]; order=np.argsort(-sims)
            if len(order) and sims[order[0]]>=SEM: top=float(sims[order[0]]); matched=[cand[int(j)] for j in order if sims[j]>=top-0.05]; rule,conf="semantic",top
        cn={**cell,"start":parse_date(cell.get("start")),"end":parse_date(cell.get("end")),"wbs_activity_id":cell.get("wbs_activity_id")}
        results.append(build_result(cn,matched,rule,conf,ps,pe))
    return {"mappings":results,"stats":_stats(results,body.cells,body.activities,learned)}

class SaveBody(BaseModel):
    org_id:str; project_id:str; source_name:str; mappings:list
@app.post("/save")
async def save_endpoint(body:SaveBody):
    warnings=[]; header=["activity id","scope","structure","zone","category","stage","startDate(MM/DD/YYYY)","endDate(MM/DD/YYYY)"]; lines=[",".join(header)]
    for m in body.mappings:
        c=m.get("cell",{}); use_new=m.get("accepted") and (m.get("new_start") or m.get("new_end"))
        s=fmt_csv(parse_date(m.get("new_start"))) if use_new else fmt_csv(parse_date(c.get("start")))
        e=fmt_csv(parse_date(m.get("new_end"))) if use_new else fmt_csv(parse_date(c.get("end")))
        vals=[c.get("wbs_activity_id") or "",c.get("scope",""),c.get("structure",""),c.get("zone",""),c.get("category",""),c.get("stage",""),s,e]
        vals=['"'+str(v).replace('"','""')+'"' if re.search(r'[",\n]',str(v)) else str(v) for v in vals]; lines.append(",".join(vals))
    csv_text="\ufeff"+"\r\n".join(lines); learned_count=0
    if sb:
        try:
            cells=[]
            for m in body.mappings:
                c=m.get("cell",{}); use_new=m.get("accepted") and (m.get("new_start") or m.get("new_end"))
                cells.append({**{k:c.get(k) for k in ("id","scope","structure","zone","category","stage")},
                              "start":m.get("new_start") if use_new else c.get("start"),"end":m.get("new_end") if use_new else c.get("end")})
            sb.table("schedule_versions").insert({"project_id":body.project_id,"label":f'{datetime.utcnow().date().isoformat()} · {body.source_name}',"cells":cells,"source_name":body.source_name}).execute()
        except Exception as e: warnings.append(f"version save failed: {e}")
        try:
            learn=[]
            for m in body.mappings:
                if not (m.get("accepted") and m.get("matched_names")) or m.get("rule")=="id_match": continue
                c=m.get("cell",{})
                for nm in m["matched_names"]:
                    p=pattern_of(nm)
                    if not p: continue
                    learn.append({"org_id":body.org_id,"project_id":None,"scope":"org","pattern":p,"target_kind":"work","category":c.get("category"),"stage":c.get("stage")})
                    learn.append({"org_id":body.org_id,"project_id":body.project_id,"scope":"project","pattern":p,"target_kind":"location","structure":c.get("structure"),"zone":c.get("zone")})
            if learn: sb.table("learned_mappings").insert(learn).execute(); learned_count=len(learn)
        except Exception as e: warnings.append(f"learning save failed: {e}")
    else: warnings.append("Supabase not configured — CSV only")
    return {"saved":not warnings,"learned":learned_count,"csv":csv_text,"warnings":warnings}

class ConfigBody(BaseModel):
    openai_api_key:Optional[str]=None
@app.post("/config")
async def set_config(body:ConfigBody):
    if not sb: raise HTTPException(500,"Supabase not configured")
    if body.openai_api_key is not None:
        sb.table("app_config").upsert({"key":"openai_api_key","value":body.openai_api_key or "","updated_at":datetime.utcnow().isoformat()}).execute()
    return {"ok":True,"embedder":embedder_name()}
@app.get("/config")
async def get_config(): return {"has_openai_key":bool(get_openai_key()),"embedder":embedder_name()}

@app.delete("/project/{project_id}")
async def delete_project(project_id:str):
    if not sb: raise HTTPException(500,"Supabase not configured")
    sb.table("projects").delete().eq("id",project_id).execute(); return {"deleted":"project","id":project_id}
@app.delete("/org/{org_id}")
async def delete_org(org_id:str):
    if not sb: raise HTTPException(500,"Supabase not configured")
    sb.table("organizations").delete().eq("id",org_id).execute(); return {"deleted":"org","id":org_id}
