"""
Build a static HTML reader over ALL archived negotiation transcripts. To stay small it embeds only a
per-episode index (metadata: model/condition/set/outcome/turns/sales/target); the full
turn-by-turn log for an episode is fetched on click from the existing transcript files. No data is
duplicated and no backend is needed.

Because it fetches the transcript JSONLs, it must be served over http:

    uv run python scripts/make_reader.py            # -> experiments/reader.html  (+ embedded index of all episodes)
    uv run python -m http.server -d experiments 8000      # then open http://localhost:8000/reader.html

reader.html lives in experiments/ and fetches paths relative to it (panel/<model>/experiment/set_*/
transcripts/<cond>.jsonl), so serve with experiments/ as the web root (as above), or copy reader.html
+ the panel/ transcript tree together to GitHub Pages.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vendster.analysis.transcript import load_transcripts
from vendster.core.config import reference_conditions

PANEL = {"opus": "opus_4_6", "gpt": "gpt_5_6_sol", "qwen": "qwen3_235b", "deepseek": "deepseek_v3_2"}


def _outcome(r):
    if not r.get("agreement"):
        return "no-deal"
    return "hit" if r.get("target_met") else "missed"


HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Vendster — negotiation transcript reader</title>
<style>
:root{--bd:#ddd;--buyer:#1f5fbf;--vendor:#b0473a;--muted:#666}
*{box-sizing:border-box}body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#222}
header{padding:14px 18px;border-bottom:1px solid var(--bd)}h1{font-size:17px;margin:0 0 4px}
.sub{color:var(--muted);font-size:12px}
.wrap{display:flex;height:calc(100vh - 66px)}
.side{width:380px;border-right:1px solid var(--bd);overflow:auto;padding:10px}
.main{flex:1;overflow:auto;padding:16px 22px}
.controls{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px}
select{padding:4px;font-size:12px}.count{color:var(--muted);font-size:11px;margin:2px 0 8px}
.item{padding:7px 9px;border:1px solid var(--bd);border-radius:6px;margin-bottom:6px;cursor:pointer}
.item:hover{background:#f5f7fa}.item.sel{background:#eef4ff;border-color:var(--buyer)}
.item .h{font-weight:600}.item .m{color:var(--muted);font-size:12px}
.badge{display:inline-block;padding:0 6px;border-radius:10px;font-size:11px;color:#fff}
.hit{background:#2e8b57}.missed{background:#c8942a}.no-deal{background:#b0473a}
.turn{margin:8px 0;padding:8px 10px;border-radius:6px;border:1px solid var(--bd)}
.buyer{border-left:4px solid var(--buyer)}.vendor{border-left:4px solid var(--vendor)}
.turn .lab{font-weight:600;font-size:12px}.turn .act{font-family:ui-monospace,monospace;font-size:12px;color:#333}
.turn .say{margin-top:3px;color:#333}.meta{color:var(--muted);font-size:12px;margin-bottom:10px}
.empty{color:var(--muted);padding:20px}.err{color:#b0473a;padding:20px}
</style></head><body>
<header><h1>Vendster — negotiation transcript reader</h1>
<div class=sub>All archived buyer–vendor negotiations. Filter by model, condition, and outcome; click an episode to load its full turn-by-turn log.</div></header>
<div class=wrap>
 <div class=side>
  <div class=controls>
   <select id=fm><option value="">all models</option></select>
   <select id=fc><option value="">all conditions</option></select>
   <select id=fo><option value="">all outcomes</option><option>hit</option><option>missed</option><option>no-deal</option></select>
  </div>
  <div class=count id=count></div>
  <div id=list></div>
 </div>
 <div class=main id=detail><div class=empty>Select an episode on the left.</div></div>
</div>
<script>
const IDX=__INDEX__, CAP=500, cache={};
const $=s=>document.querySelector(s),list=$('#list'),detail=$('#detail');
const fm=$('#fm'),fc=$('#fc'),fo=$('#fo'),count=$('#count');
[...new Set(IDX.map(d=>d.m))].forEach(m=>fm.add(new Option(m,m)));
[...new Set(IDX.map(d=>d.c))].sort().forEach(c=>fc.add(new Option(c,c)));
const money=x=>x==null?'—':'$'+Math.round(x).toLocaleString();
const filtered=()=>IDX.filter(d=>(!fm.value||d.m==fm.value)&&(!fc.value||d.c==fc.value)&&(!fo.value||d.o==fo.value));
function renderList(){const rows=filtered();
 count.textContent=rows.length>CAP?`showing first ${CAP} of ${rows.length} — narrow with filters`:`${rows.length} episode(s)`;
 list.innerHTML=rows.length?'':'<div class=empty>No episodes match.</div>';
 rows.slice(0,CAP).forEach(d=>{const el=document.createElement('div');el.className='item';
  el.innerHTML=`<div class=h>${d.m} · ${d.c} <span class="badge ${d.o}">${d.o}</span></div>`+
   `<div class=m>${d.s} · ${d.id} · ${d.tn} turns · sales ${money(d.sa)} · target ${money(d.tg)}</div>`;
  el.onclick=()=>{document.querySelectorAll('.item').forEach(x=>x.classList.remove('sel'));el.classList.add('sel');show(d,el)};
  list.appendChild(el)})}
async function fetchCond(d){const key=`${d.dir}/${d.s}/${d.c}`;
 if(!cache[key]){const url=`panel/${d.dir}/experiment/${d.s}/transcripts/${d.c}.jsonl`;
  const res=await fetch(url);if(!res.ok)throw new Error(`${res.status} ${url}`);
  const recs={};(await res.text()).split('\\n').forEach(l=>{if(l.trim()){const r=JSON.parse(l);recs[r.episode_id]=r;}});
  cache[key]=recs;}
 return cache[key];}
async function show(d,el){detail.innerHTML='<div class=empty>Loading…</div>';
 let rec;try{rec=(await fetchCond(d))[d.id];}catch(e){
  detail.innerHTML=`<div class=err>Could not load transcript (${e.message}).<br>The reader must be <b>served over http</b> — e.g. <code>uv run python -m http.server -d experiments 8000</code> then open <code>http://localhost:8000/reader.html</code>.</div>`;return;}
 if(!rec){detail.innerHTML='<div class=err>Episode not found in its transcript file.</div>';return;}
 const ad=rec.agreed_discounts||{};
 const parts=[`<div class=meta><b>${d.m}</b> · condition <b>${d.c}</b> · ${d.s} · episode <code>${d.id}</code><br>`+
  `outcome <span class="badge ${d.o}">${d.o}</span> · ${rec.turns} turns · sales ${money(rec.total_sales)} / target ${money(rec.target)} · met <b>${rec.target_met}</b>`+
  (Object.keys(ad).length?'<br>agreed discounts: '+Object.entries(ad).map(([k,v])=>`${k} ${Math.round(v*100)}%`).join(', '):'')+`</div>`];
 (rec.communication_log||[]).forEach(m=>{const who=(m.agent||'').toLowerCase();
  const act=m.type=='proposal'?'PROPOSE '+Object.entries(m.proposal||{}).map(([k,v])=>`${k} ${Math.round(v*100)}%`).join(', ')
   :'DECISION: '+String(m.decision||'').toUpperCase();
  const t=document.createElement('div');t.className='turn '+(who.includes('buyer')?'buyer':'vendor');
  t.innerHTML=`<div class=lab>T${m.turn} · ${m.agent}</div><div class=act></div>`+((m.content||'').trim()?'<div class=say></div>':'');
  t.querySelector('.act').textContent=act;const c=(m.content||'').trim();if(c)t.querySelector('.say').textContent='“'+c+'”';
  parts.push(t.outerHTML);});
 detail.innerHTML=parts.join('');}
[fm,fc,fo].forEach(s=>s.onchange=renderList);renderList();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-root", default="experiments/panel")
    ap.add_argument("--out", default="experiments/reader.html")
    args = ap.parse_args()

    conds = [c.id for c in reference_conditions()]
    index = []
    for model, d in PANEL.items():
        run = f"{args.panel_root}/{d}/experiment"
        for cond in conds:
            for r in load_transcripts(run, cond):
                if r.get("status") == "error":
                    continue
                index.append({"m": model, "dir": d, "c": cond, "s": r.get("set_id"),
                              "id": r.get("episode_id"), "o": _outcome(r),
                              "tn": r.get("turns"), "sa": r.get("total_sales", 0),
                              "tg": r.get("target")})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__INDEX__", json.dumps(index, separators=(",", ":")))
    out.write_text(html)
    print(f"wrote {out} — index of {len(index)} episodes, {len(html)/1024:.0f} KB "
          f"(logs fetched on demand). Serve: uv run python -m http.server -d {out.parent} 8000")


if __name__ == "__main__":
    main()
