from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import app.models  # noqa: F401  (registers every model on Base.metadata before any request runs)
from app.auth.routes import router as auth_router
from app.config import settings
from app.observability import (
    instrument_fastapi,
    instrument_static,
    setup_logging,
    setup_metrics,
    setup_sentry,
    setup_tracing,
)
from app.routes.courses import router as courses_router
from app.routes.me import router as me_router

setup_tracing()
setup_logging()
setup_sentry()

app = FastAPI(title="Knovolve")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # No allow_credentials: auth is a bearer token in the Authorization header,
    # not a cookie, so the browser never needs to send credentials cross-origin.
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(courses_router)
app.include_router(me_router)

instrument_static()
instrument_fastapi(app)
setup_metrics(app)


@app.get("/health")
def health():
    return {"status": "ok"}
