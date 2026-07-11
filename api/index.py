import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path_name: str):
    client = httpx.AsyncClient()
    url = f"{BACKEND_URL}/{path_name}"
    
    # Pass query params and add ngrok warning bypass header
    params = dict(request.query_params)
    headers = dict(request.headers)
    headers["ngrok-skip-browser-warning"] = "true"
    headers.pop("host", None)  # Avoid host mismatch
    
    body = await request.body()
    
    # Handle Streaming responses (e.g. Research Plan generation)
    if "stream" in path_name:
        async def stream_generator():
            async with client.stream(
                request.method, url, params=params, headers=headers, content=body, timeout=240.0
            ) as r:
                async for chunk in r.aiter_raw():
                    yield chunk
        return StreamingResponse(stream_generator(), media_type="text/plain")
    
    # Handle Standard JSON responses
    response = await client.request(
        request.method, url, params=params, headers=headers, content=body, timeout=60.0
    )
    return StreamingResponse(
        response.iter_bytes(),
        status_code=response.status_code,
        headers=dict(response.headers)
    )
