#!/usr/bin/env python3
"""Stack de mocks pour tester laplace-research hors-ligne et sans modèle.

Trois serveurs HTTP locaux :
  - :8810  llama-server simulé (POST /completion) — un « LLM » scripté qui
           reconnaît chaque type de prompt du moteur et répond un JSON
           conforme à la grammaire attendue ;
  - :8811  mini-web : trois pages HTML (dont une avec nav/footer/script à
           nettoyer par le lecteur) ;
  - :8812  SearXNG simulé (GET /search?format=json).

Usage :
  python3 tests/mock_research_stack.py &          # lance les 3 mocks
  LAPLACE_RESEARCH_LLM_URL=http://127.0.0.1:8810 \
  LAPLACE_SEARXNG_URL=http://127.0.0.1:8812 \
  LAPLACE_RESEARCH_PROVIDERS=searxng \
  ./build/linux/x86_64/release/laplace-research "sujet de test"
"""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PAGES = {
    "/vulkan-sync": """<!doctype html><html><head><title>Vulkan Synchronization Guide</title>
<script>var noise = 'should never appear';</script></head><body>
<nav><a href="/">Menu bruit</a></nav>
<h1>Vulkan Synchronization</h1>
<p>Pipeline barriers control execution order on the GPU. Since Vulkan 1.3,
dynamic rendering removes render pass boilerplate.</p>
<h2>Timeline semaphores</h2>
<p>Timeline semaphores (Vulkan 1.2, 2020) replace fences for host-device sync
and support out-of-order signaling with 64-bit counters.</p>
<footer>Copyright bruit 2026</footer></body></html>""",
    "/vulkan-arc": """<!doctype html><html><head><title>Intel Arc Driver Notes</title></head><body>
<h1>Intel Arc &amp; Vulkan</h1>
<p>Intel ANV driver exposes Vulkan 1.3 on Arc GPUs. Mesa 24.1 fixed a
synchronization bug that stalled compute queues at 100&#37; utilisation.</p>
</body></html>""",
    "/plain.txt": "Plain text source: barriers cost ~5us on Arc A770 (measured 2025).",
}

SEARCH_RESULTS = [
    {"url": "http://127.0.0.1:8811/vulkan-sync", "title": "Vulkan Synchronization Guide",
     "content": "Pipeline barriers, timeline semaphores"},
    {"url": "http://127.0.0.1:8811/vulkan-arc", "title": "Intel Arc Driver Notes",
     "content": "ANV driver, Mesa fixes"},
    {"url": "http://127.0.0.1:8811/plain.txt", "title": "Barrier microbenchmark",
     "content": "measured costs"},
]


class LlamaMock(BaseHTTPRequestHandler):
    reads = 0  # nombre d'extractions déjà servies (état du scénario)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body.get("prompt", "")
        out = self.route(prompt)
        payload = json.dumps({"content": out, "tokens_evaluated": len(prompt) // 4,
                              "tokens_predicted": len(out) // 4}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    @classmethod
    def route(cls, prompt):
        if "Break this research topic" in prompt:
            return json.dumps({"subquestions": [
                "How do Vulkan pipeline barriers work?",
                "What are known Intel Arc driver issues?"]})
        if "Extract up to" in prompt:
            cls.reads += 1
            return json.dumps({
                "learnings": [f"Learning {cls.reads}: fact extracted from source"],
                "followups": ["What changed in Mesa 24.1?"] if cls.reads == 1 else []})
        if "Choose ONE action" in prompt:
            # Scénario : chercher, lire 2 sources, puis répondre.
            m = re.findall(r"^(\d+): ", prompt, re.M)
            if "SEARCH QUERIES ALREADY DONE" not in prompt:
                return json.dumps({"action": "search", "arg": "vulkan barriers intel arc"})
            if m and cls.reads < 2 and "read" in prompt:
                return json.dumps({"action": "read", "arg": m[0]})
            if "answer" in prompt:
                return json.dumps({"action": "answer", "arg": ""})
            return json.dumps({"action": "search", "arg": "fallback query"})
        if "Judge this answer" in prompt:
            return json.dumps({"pass": True, "reason": "grounded and complete"})
        if "Propose 3 to 5 section titles" in prompt:
            return json.dumps({"sections": ["Barrières de pipeline", "Pilotes Intel Arc"]})
        if "Write ONLY the section" in prompt:
            return "Contenu de section adossé aux sources [S1] et [S2].\n"
        if "Write the final answer" in prompt:
            return ("Les barrières de pipeline ordonnent l'exécution GPU [S1] ; "
                    "le pilote ANV expose Vulkan 1.3 sur Arc [S2].")
        return json.dumps({"error": "prompt non reconnu"})

    def log_message(self, *a):
        pass


class WebMock(BaseHTTPRequestHandler):
    def _serve(self, send_body):
        page = PAGES.get(self.path)
        if page is None:
            self.send_response(404)
            self.end_headers()
            return
        data = page.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def do_GET(self):
        self._serve(True)

    def do_HEAD(self):
        self._serve(False)

    def log_message(self, *a):
        pass


class SearxMock(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps({"results": SEARCH_RESULTS}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


def serve(port, handler):
    HTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    for port, handler in [(8810, LlamaMock), (8811, WebMock), (8812, SearxMock)]:
        threading.Thread(target=serve, args=(port, handler), daemon=True).start()
    print("mocks prêts : llama :8810, web :8811, searxng :8812")
    threading.Event().wait()
