FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime system dependencies:
#   graphviz     -> `dot` binary used by app/diagrams/render.py
#   pango/cairo  -> WeasyPrint PDF/HTML export (app/documents/render.py)
#   fonts        -> text layout for WeasyPrint
RUN apt-get update && apt-get install -y --no-install-recommends \
        graphviz \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libharfbuzz-subset0 \
        libgdk-pixbuf-2.0-0 \
        shared-mime-info \
        fonts-dejavu-core \
        fontconfig \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer) - the package is found via app/.
COPY pyproject.toml ./
COPY app ./app
RUN pip install --upgrade pip setuptools wheel && pip install .

# Then the rest of the source (tests, alembic, observability configs, ...).
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
