from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.auth.routes import router as auth_router
from app.config import settings
from app.routes.courses import router as courses_router

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


@app.get("/health")
def health():
    return {"status": "ok"}
