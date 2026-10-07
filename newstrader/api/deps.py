from __future__ import annotations

from fastapi import HTTPException, Request

from ..context import AppContext


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def bad_request(message: str, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail=message)
