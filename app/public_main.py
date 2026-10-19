from __future__ import annotations
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from app.interactive_demo.api import router

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
app = FastAPI(title="Resonant AI Public Pilot", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.include_router(router)

# One-instance, best-effort traffic controls for a free-host pilot, not a production WAF.
_recent = defaultdict(deque)
@app.middleware("http")
async def pilot_guard(request: Request, call_next):
    path = request.url.path
    if path.startswith('/api/') and request.method not in ('GET', 'POST'):
        return JSONResponse({'error':'Method not allowed'}, status_code=405)
    if path.startswith('/api/'):
        if path.endswith('/m13-semantic-analyze') or path.endswith('/m13-vision-status'):
            return JSONResponse({'error':'Local Ollama is unavailable in the public edition'}, status_code=503)
        client = request.client.host if request.client else 'unknown'
        # Header-derived IPs are deliberately not trusted for identity.
        now = time.monotonic()
        hits = _recent[client]
        while hits and hits[0] < now - 60:
            hits.popleft()
        limit = int(os.getenv('PILOT_REQUESTS_PER_MINUTE', '30'))
        if len(hits) >= limit:
            return JSONResponse({'error':'Please wait before making another request'},status_code=429)
        hits.append(now)
        # Protect expensive uploads, in addition to endpoint's own 12 MiB cap.
        if request.headers.get('content-length'):
            try:
                if int(request.headers['content-length']) > 13 * 1024 * 1024:
                    return JSONResponse({'error':'Request too large'},status_code=413)
            except ValueError:
                return JSONResponse({'error':'Invalid request size'},status_code=400)
    resp = await call_next(request)
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['Referrer-Policy'] = 'no-referrer'
    resp.headers['X-Frame-Options'] = 'DENY'
    return resp

@app.get('/health')
def health():
    return {'status':'ok','edition':'public-pilot'}

@app.get('/')
def landing():
    return RedirectResponse('/companion-experience', status_code=302)

@app.get('/companion-experience')
def companion_experience():
    return FileResponse(STATIC/'companion_experience.html')
