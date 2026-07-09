import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AcademicForge Vercel Proxy")

# Allow Streamlit to call this proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


async def _proxy_request(request: Request, path: str):
    url = f"{BACKEND_URL}/{path}"
    
    # Read the body from the incoming request
    body = await request.body()
    
    # Use httpx to forward the request
    # Vercel serverless has execution limits, but we stream the response for SSE
    client = httpx.AsyncClient(timeout=300.0)
    
    req = client.build_request(
        method=request.method,
        url=url,
        headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")},
        content=body
    )
    
    resp = await client.send(req, stream=True)
    
    # Return as StreamingResponse to pipe back to the frontend
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers={k: v for k, v in resp.headers.items() if k.lower() not in ("content-encoding", "content-length")},
        background=client.aclose
    )

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_all(request: Request, path: str):
    return await _proxy_request(request, path)
