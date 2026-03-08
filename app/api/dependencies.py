from __future__ import annotations

from fastapi import Request

from app.services.container import AppServices


def get_services(request: Request) -> AppServices:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise RuntimeError("Application services are not initialized.")
    return services

