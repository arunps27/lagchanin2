"""Minimal local web UI (stdlib only): python web.py  ->  http://127.0.0.1:8765 (override with PORT)"""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import get_settings
from generator import GroundedGenerator
from retriever import HybridRetriever

S = get_settings()
RETRIEVER = HybridRetriever(S)
GENERATOR = GroundedGenerator(S)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>KYC Hybrid RAG</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 16px;color:#1a1a1a}
textarea{width:100%;font:inherit;padding:8px;box-sizing:border-box}
button{padding:8px 18px;margin-top:8px;font:inherit;cursor:pointer}
#answer{white-space:pre-wrap;background:#f4f6f8;padding:12px;border-radius:6px;margin-top:16px}
.src{font-size:.9rem;color:#444;margin:.3rem 0}.meta{color:#777;font-size:.85rem}
</style></head><body>
<h2>Banking KYC Hybrid RAG</h2>
<p class="meta">BM25 + dense &rarr; RRF &rarr; cross-encoder rerank &rarr; gpt-oss-120b (Groq)</p>
<textarea id="q" rows="3" placeholder="e.g. When is enhanced due diligence required for PEPs?"></textarea>
<button id="go">Ask</button>
<div id="out"></div>
<script>
const out=document.getElementById('out');
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  out.innerHTML='<p>Thinking...</p>';
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const d=await r.json();
  if(d.error){out.textContent='Error: '+d.error;return;}
  const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  out.innerHTML='<div id="answer">'+esc(d.answer)+'</div><h4>Sources</h4>'+
    d.citations.map(c=>'<div class="src">['+c.n+'] '+esc(c.source)+' &ndash; '+esc(c.section)+'</div>').join('')+
    '<p class="meta">mode: '+d.mode+' &middot; retrieval '+d.retrieval_s+'s, generation '+d.generation_s+'s</p>';
}
document.getElementById('go').onclick=ask;
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/ask":
            return self._send(404, "{}", "application/json")
        try:
            n = int(self.headers.get("Content-Length", 0))
            q = json.loads(self.rfile.read(n))["question"].strip()
            t0 = time.perf_counter()
            chunks = RETRIEVER.retrieve(q)
            t1 = time.perf_counter()
            res = GENERATOR.generate(q, chunks)
            t2 = time.perf_counter()
            res["retrieval_s"], res["generation_s"] = round(t1 - t0, 2), round(t2 - t1, 2)
            self._send(200, json.dumps(res), "application/json")
        except Exception as e:  # surface errors to the UI
            self._send(500, json.dumps({"error": str(e)}), "application/json")


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8765"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving on http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()
