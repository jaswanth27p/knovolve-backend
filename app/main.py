from fastapi import FastAPI
from app.auth.routes import router as auth_router
from app.routes.courses import router as courses_router

app = FastAPI(title="Knovolve")
app.include_router(auth_router)
app.include_router(courses_router)


@app.get("/health")
def health():
    return {"status": "ok"}
