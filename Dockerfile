# SpartanScout image: one image, two programs (the API and the Streamlit UI).
# compose.yaml starts each with its own command.
#
# Stage 1 (builder) uses uv to install the exact Python and the exact locked
# dependencies. Stage 2 (runtime) is a slim Debian image that receives only the
# finished interpreter, virtual environment, and code: no uv, no build tools.
#
# Versions: uv 0.12.17 matches the uv that created uv.lock (see .venv/pyvenv.cfg);
# Python 3.12.14 comes from .python-version, installed by uv.

ARG UV_VERSION=0.12.17

FROM ghcr.io/astral-sh/uv:${UV_VERSION}-debian-slim AS builder

# Byte-compile for faster start-up; copy files (cache mounts can't hard-link);
# keep the managed Python in /python so the runtime stage can copy it.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed

WORKDIR /app

# 1. The interpreter named in .python-version (cached until that file changes).
COPY .python-version ./
RUN uv python install

# 2. Dependencies only, from the lock file. This slow layer is reused as long
#    as uv.lock and pyproject.toml are unchanged, even when the code changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# 3. The code, then the project itself. --locked fails the build if uv.lock is
#    out of date, so the image always matches the committed lock file.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


FROM debian:trixie-slim AS runtime

# Run as an ordinary user, never as root.
RUN groupadd --system app \
 && useradd --system --gid app --create-home --home-dir /home/app app

COPY --from=builder /python /python
COPY --from=builder --chown=app:app /app /app

# The tracker folder is a volume in compose.yaml; it must exist and belong to app.
RUN mkdir -p /app/data/tracker && chown app:app /app/data/tracker

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

WORKDIR /app
USER app
EXPOSE 8000 8501

# Default program: the API. compose.yaml overrides this for the UI.
CMD ["uvicorn", "job_search_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
