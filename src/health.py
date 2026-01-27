"""Health check server for container orchestration."""

import asyncio
from enum import Enum

import aiohttp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response


class ServerState(Enum):
    """Server state for health checks."""
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


class HealthServer:
    """Health check server that runs alongside the main vLLM server."""
    
    def __init__(self):
        self.state = ServerState.INITIALIZING
    
    def set_ready(self):
        """Mark server as ready."""
        self.state = ServerState.READY
    
    def set_error(self):
        """Mark server as error."""
        self.state = ServerState.ERROR
    
    def set_initializing(self):
        """Mark server as initializing."""
        self.state = ServerState.INITIALIZING
    
    def create_app(self) -> FastAPI:
        """Create FastAPI app for health checks."""
        app = FastAPI()
        
        @app.get("/ping")
        async def ping():
            if self.state is ServerState.INITIALIZING:
                return Response(status_code=204)
            elif self.state is ServerState.READY:
                return Response(status_code=200)
            else:  # ERROR state
                return Response(status_code=500)
        
        return app
    
    def run(self, host: str, port: int):
        """Run the health check server (blocking)."""
        app = self.create_app()
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
    
    async def monitor_vllm_health(self, host: str, port: int, max_attempts: int = 60):
        """
        Monitor vLLM server health and update state.
        
        Args:
            host: vLLM server host
            port: vLLM server port
            max_attempts: Maximum attempts (5 seconds each = 5 minutes total)
        """
        url = f"http://{host}:{port}/health"
        
        for _ in range(max_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            self.set_ready()
                            print("vLLM server is ready!")
                            return
            except Exception:
                pass
            
            await asyncio.sleep(5)
        
        # Server didn't become ready in time
        print("vLLM server failed to become ready within timeout")
        self.set_error()
