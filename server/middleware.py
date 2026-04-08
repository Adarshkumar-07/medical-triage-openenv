"""
server/middleware.py

ASGI middleware stack: RequestID → Logging → ErrorNormalization.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.monotonic()
        request_id = getattr(request.state, "request_id", "-")
        response = await call_next(request)
        duration_ms = round((time.monotonic() - start) * 1000, 2)
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )
        logger.info(json.dumps({
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": client_ip,
        }))
        return response


class ErrorNormalizationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = getattr(request.state, "request_id", "-")
        try:
            return await call_next(request)
        except Exception as exc:
            logger.error(
                "Unhandled exception on %s %s [request_id=%s]: %s\n%s",
                request.method, request.url.path, request_id, exc,
                traceback.format_exc(),
            )
            return JSONResponse(
                status_code=500,
                content={
                    "type": "about:blank",
                    "title": type(exc).__name__,
                    "status": 500,
                    "detail": str(exc),
                    "request_id": request_id,
                },
            )