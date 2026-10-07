"""FastAPI app: REST API under /api, live events on /ws, the dashboard files at /.

Security: the server only listens on 127.0.0.1, and every /api and /ws request must carry the random
token created at launch. That stops other websites open in a browser from poking the trading API.
"""

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .. import __version__, paths
from ..context import AppContext
from ..orchestrator import Orchestrator
from .routes import ALL_ROUTERS

log = logging.getLogger(__name__)

TOKEN_HEADER = "X-NT-Token"


def _token_ok(ctx: AppContext, supplied: str | None) -> bool:
    return bool(supplied) and hmac.compare_digest(str(supplied), ctx.token)


def format_validation_error(exc: ValidationError | RequestValidationError) -> list[str]:
    messages = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p not in ("body",))
        msg = err.get("msg", "invalid value")
        messages.append(f"{loc}: {msg}" if loc else msg)
    return messages


def create_app(ctx: AppContext, start_engine: bool = True) -> FastAPI:
    orchestrator = Orchestrator(ctx)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_engine:
            await orchestrator.start()
        else:
            import asyncio

            ctx.loop = asyncio.get_running_loop()
            ctx.bus.bind_loop(ctx.loop)
        try:
            yield
        finally:
            if start_engine:
                await orchestrator.stop()

    app = FastAPI(title="NewsTrader", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ctx = ctx
    app.state.orchestrator = orchestrator

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            supplied = request.headers.get(TOKEN_HEADER) or request.query_params.get("token")
            if not _token_ok(ctx, supplied):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValidationError)
    async def pydantic_error(_: Request, exc: ValidationError):
        return JSONResponse({"error": "invalid settings", "details": format_validation_error(exc)}, status_code=422)

    @app.exception_handler(RequestValidationError)
    async def request_error(_: Request, exc: RequestValidationError):
        return JSONResponse({"error": "invalid request", "details": format_validation_error(exc)}, status_code=422)

    for router in ALL_ROUTERS:
        app.include_router(router, prefix="/api")

    @app.websocket("/ws")
    async def ws_events(websocket: WebSocket):
        if not _token_ok(ctx, websocket.query_params.get("token")):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        queue = ctx.bus.subscribe()
        try:
            await websocket.send_json({"type": "hello", "data": {"version": __version__, "mode": ctx.state.mode}})
            while True:
                msg = await queue.get()
                await websocket.send_json(msg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:
            log.debug("websocket closed", exc_info=True)
        finally:
            ctx.bus.unsubscribe(queue)

    web = paths.web_dir()

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(web / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse(web / "favicon.svg", media_type="image/svg+xml")

    app.mount("/static", StaticFiles(directory=web), name="static")
    return app
