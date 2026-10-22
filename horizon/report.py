"""Render results/summary.json into a static page at docs/index.html (GitHub Pages)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .axes import AXES
from .runner import ROOT

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AGI Horizon</title>
<style>
:root{color-scheme:light;--surface:#fcfcfb;--panel:#f5f4f1;--text:#0b0b0b;--text2:#52514e;--muted:#8a8984;--grid:#e6e5e0;
--s1:#2a78d6;--front:#52514e;--ring:#fcfcfb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;--panel:#222220;--text:#f0efec;
--text2:#c3c2b7;--muted:#8f8e86;--grid:#33332f;--s1:#3987e5;--front:#c3c2b7;--ring:#1a1a19}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--panel:#222220;--text:#f0efec;--text2:#c3c2b7;--muted:#8f8e86;
--grid:#33332f;--s1:#3987e5;--front:#c3c2b7;--ring:#1a1a19}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px 64px}h1{font-size:28px;margin:0 0 4px}h2{font-size:18px;margin:40px 0 8px}
p.lede{color:var(--text2);margin:0 0 24px;max-width:760px}.note{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
.card{background:var(--panel);border-radius:10px;padding:14px 14px 6px}.card h3{margin:0;font-size:14px}.card .sub{color:var(--muted);font-size:12px}
.hero{background:var(--panel);border-radius:10px;padding:16px}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{fill:var(--muted);font-size:11px}
.dot{fill:var(--s1);stroke:var(--ring);stroke-width:2}.dot:hover,.dot.on{r:6}.front{fill:none;stroke:var(--front);stroke-width:2;stroke-dasharray:4 3}
.gl{stroke:var(--grid);stroke-width:1}.lbl{fill:var(--text2)!important;font-size:11px}
#tip{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);padding:6px 9px;border-radius:6px;font-size:12px;opacity:0;transition:opacity .1s;max-width:260px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--text2);font-weight:600;cursor:pointer}.tw{overflow-x:auto}
.legend{display:flex;gap:16px;color:var(--text2);font-size:12px;margin:6px 0 0}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--s1);margin-right:5px;vertical-align:-1px}
.legend b{display:inline-block;width:16px;border-top:2px dashed var(--front);margin-right:5px;vertical-align:3px}
</style></head><body><main>
<h1>AGI Horizon</h1>
<p class="lede">How far up open-ended, auto-graded difficulty ladders each model climbs, across seven abilities plus calibration.
Each score θ is the level where a model succeeds half the time. Higher is better and there is no ceiling.
No model grades anything; all items are freshly generated from a secret seed. <a href="https://github.com/raiden-02/agi-horizon/blob/main/DESIGN.md">Method and bias controls</a>.</p>
<p class="note" id="meta"></p>
<div class="hero"><h3 style="margin:0">Horizon Index over time</h3><div class="sub note">Mean θ over the 7 axes, by OpenRouter listing date. Dashed: best so far. No human baseline yet, so no AGI line is drawn.</div>
<div id="hero"></div><div class="legend"><span><i></i>model</span><span><b></b>frontier (best so far)</span></div></div>
<h2>Per ability</h2><div class="grid" id="axes"></div>
<h2>All results</h2><p class="note">θ with 90% bootstrap interval in the tooltip. ≥ marks a lower bound (never failed); ≤ an upper bound (never succeeded). Calibration: Brier skill, 1.0 = perfect.</p>
<div class="tw"><table id="tbl"></table></div>
</main><div id="tip"></div>
<script>
const D=__DATA__, AX=__AXES__;
const ms=Object.entries(D.models).map(([id,s])=>({id,s,t:(s.meta.created||0)*1000})).filter(m=>m.t);
document.getElementById('meta').textContent=`${ms.length} models · generated ${D.generated} · pre-registration ${D.lock?D.lock.slice(0,12):'unlocked'}`;
const tip=document.getElementById('tip');
function showTip(e,h){tip.innerHTML=h;tip.style.opacity=1;const x=Math.min(e.clientX+12,innerWidth-270);tip.style.left=x+'px';tip.style.top=(e.clientY+12)+'px'}
function hide(){tip.style.opacity=0}
function chart(el,pts,h,W){ // pts: {t,v,id,label}
  const H=h,P={l:34,r:12,t:10,b:22};if(!pts.length){el.innerHTML='<p class="note">no data</p>';return}
  const t0=Math.min(...pts.map(p=>p.t)),t1=Math.max(...pts.map(p=>p.t)),vs=pts.map(p=>p.v);
  const v0=Math.min(0,Math.floor(Math.min(...vs))),v1=Math.ceil(Math.max(...vs))+1;
  const X=t=>P.l+(t-t0)/((t1-t0)||1)*(W-P.l-P.r),Y=v=>H-P.b-(v-v0)/(v1-v0)*(H-P.t-P.b);
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img">`;const step=Math.max(1,Math.ceil((v1-v0)/5));
  for(let v=v0;v<=v1;v+=step)s+=`<line class="gl" x1="${P.l}" x2="${W-P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${P.l-6}" y="${Y(v)+4}" text-anchor="end">${v}</text>`;
  for(let y=new Date(t0).getFullYear();y<=new Date(t1).getFullYear()+1;y++){const t=Date.UTC(y,0,1);if(t>=t0&&t<=t1)s+=`<text x="${X(t)}" y="${H-4}" text-anchor="middle">${y}</text>`}
  const so=[...pts].sort((a,b)=>a.t-b.t);
  let path='';let b=-1e9;so.forEach((p,i)=>{const nb=Math.max(b,p.v);path+=(i?`L${X(p.t)},${Y(b)}L${X(p.t)},${Y(nb)}`:`M${X(p.t)},${Y(nb)}`);b=nb});
  path+=`L${W-P.r},${Y(b)}`;s+=`<path class="front" d="${path}"/>`;
  so.forEach((p,i)=>s+=`<circle class="dot" r="4.5" cx="${X(p.t)}" cy="${Y(p.v)}" data-i="${pts.indexOf(p)}"/>`);
  const top=[...pts].sort((a,b)=>b.v-a.v).slice(0,W>500?3:1);
  top.forEach(p=>s+=`<text class="lbl" x="${X(p.t)+(X(p.t)>W/2?-8:8)}" y="${Y(p.v)-8}" text-anchor="${X(p.t)>W/2?'end':'start'}">${p.id.split('/')[1]}</text>`);
  el.innerHTML=s+'</svg>';
  el.querySelectorAll('circle').forEach(c=>{const p=pts[c.dataset.i];
    c.addEventListener('mousemove',e=>showTip(e,`<b>${p.id}</b><br>${p.label}<br>${new Date(p.t).toISOString().slice(0,10)}`));c.addEventListener('mouseleave',hide)});
}
const fmt=a=>a&&a.theta!=null?(a.censored==='high'?'≥':a.censored==='low'?'≤':'')+a.theta.toFixed(1):'–';
chart(document.getElementById('hero'),ms.filter(m=>m.s.horizon_index).map(m=>({t:m.t,v:m.s.horizon_index.value,id:m.id,
  label:`Index ${m.s.horizon_index.value.toFixed(2)} (90% ${m.s.horizon_index.ci90.join('–')})`})),300,960);
const ax=document.getElementById('axes');
AX.concat(['calibration']).forEach(a=>{const c=document.createElement('div');c.className='card';
  c.innerHTML=`<h3>${a.replace('_',' ')}</h3><div class="sub">${a==='calibration'?'Brier skill (1 = perfect)':'θ: frontier level'}</div><div></div>`;ax.appendChild(c);
  const pts=a==='calibration'?ms.filter(m=>m.s.calibration.skill!=null).map(m=>({t:m.t,v:m.s.calibration.skill*10,id:m.id,label:`Brier skill ${m.s.calibration.skill} · overconfidence ${m.s.calibration.overconfidence}`}))
   :ms.filter(m=>m.s.axes[a].theta!=null).map(m=>({t:m.t,v:m.s.axes[a].theta,id:m.id,label:`θ ${fmt(m.s.axes[a])} (90% ${m.s.axes[a].ci90.join('–')}) · ${m.s.axes[a].n} items`}));
  chart(c.lastChild,pts,190,380);if(a==='calibration')c.querySelector('.sub').textContent='Brier skill ×10 (10 = perfect)'});
const cols=['model','lab','listed','index',...AX,'calib','cost $'];let sortK='index',dir=-1;
function row(m){return{model:m.id,lab:m.s.meta.lab||'',listed:new Date(m.t).toISOString().slice(0,10),index:m.s.horizon_index?m.s.horizon_index.value:-99,
 ...Object.fromEntries(AX.map(a=>[a,m.s.axes[a]])),calib:m.s.calibration.skill??-1,'cost $':m.s.cost}}
function draw(){const rows=ms.map(row).sort((a,b)=>{const va=a[sortK],vb=b[sortK];const x=v=>v&&v.theta!==undefined?(v.theta??-99):v;return (x(va)>x(vb)?1:-1)*dir});
 document.getElementById('tbl').innerHTML='<tr>'+cols.map(c=>`<th data-k="${c}">${c.replace('_',' ')}</th>`).join('')+'</tr>'+rows.map(r=>'<tr>'+cols.map(c=>{const v=r[c];
  return `<td>${v&&v.theta!==undefined?fmt(v):c==='index'?(v===-99?'–':v.toFixed(2)):c==='calib'?(v<0?'–':v.toFixed(3)):c==='cost $'?v.toFixed(2):v}</td>`}).join('')+'</tr>').join('');
 document.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;dir=sortK===k?-dir:-1;sortK=k;draw()})}
draw();
</script></body></html>"""


def build():
    s = json.load(open(os.path.join(ROOT, "results", "summary.json")))
    pricing = json.load(open(os.path.join(ROOT, "results", "pricing.json"), encoding="utf-8"))
    for mid, m in s["models"].items():
        m["meta"] = {**m.get("meta", {}), "created": (pricing.get(mid) or {}).get("created")}
    html = TEMPLATE.replace("__DATA__", json.dumps(s)).replace("__AXES__", json.dumps(list(AXES)))
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote docs/index.html", datetime.now(timezone.utc).isoformat(timespec="seconds"))
