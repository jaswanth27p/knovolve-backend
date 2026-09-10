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
from app.routes.course_extensions import router as course_extensions_router
from app.routes.courses import router as courses_router
from app.routes.me import router as me_router

setup_tracing()
setup_logging()
setup_sentry()

app = FastAPI(title="Knovolve")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # Auth cookies require allow_credentials=True; the browser refuses to
    # send/store credentialed cookies for a wildcard origin, which is why
    # cors_origins must stay an explicit allowlist rather than "*".
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(courses_router)
app.include_router(course_extensions_router)
app.include_router(me_router)

instrument_static()
instrument_fastapi(app)
setup_metrics(app)


@app.get("/health")
def health():
    return {"status": "ok"}
