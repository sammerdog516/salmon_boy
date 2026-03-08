from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_health,
    routes_ingest,
    routes_migration,
    routes_process,
    routes_risk,
    routes_train,
)
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.services.container import build_services


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    app = FastAPI(title=app_settings.app_name, version=app_settings.app_version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def startup_event() -> None:
        app_settings.ensure_directories()
        app.state.services = build_services(app_settings)

    app.include_router(routes_health.router)
    app.include_router(routes_migration.router)
    app.include_router(routes_ingest.router)
    app.include_router(routes_process.router)
    app.include_router(routes_risk.router)
    app.include_router(routes_train.router)
    return app


app = create_app()

