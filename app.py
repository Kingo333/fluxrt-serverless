import os
import time
import platform
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

PORT = int(os.getenv("PORT", "8765"))
STARTED_AT = time.time()

app = FastAPI()

def state():
    return {
        "ok": True,
        "service": "azyah-runpod-doc-native-worker",
        "status": "healthy",
        "port": PORT,
        "uptime_seconds": round(time.time() - STARTED_AT, 2),
        "python": platform.python_version(),
    }

@app.get("/")
async def root():
    return JSONResponse(state())

@app.get("/ping")
async def ping():
    return JSONResponse(state())

@app.get("/health")
async def health():
    return JSONResponse(state())

@app.get("/debug/startup")
async def debug_startup():
    return JSONResponse({
        **state(),
        "env_names": sorted(os.environ.keys()),
        "cwd": os.getcwd(),
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
