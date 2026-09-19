# The Streamlit dashboard. Build: docker build -t alpacatrader .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" \
    STREAMLIT_SERVER_HEADLESS=true

# Dependencies before sources: the heavy layer survives every code edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# The build fails here rather than at the first request: `--no-dev` has to carry everything the
# page imports at module scope, and CI builds the image without ever starting it.
RUN python -c "import alpacatrader.app.dashboard"

# Stateless: the page downloads from Alpaca and writes nothing, so no disk is mounted. The two
# credentials are injected at run time and never baked in — see .env.example.
EXPOSE 8501
# Shell form on purpose: a host that injects the port at run time (Render's PORT) would not be
# expanded by the exec form. The fallback keeps `docker run -p 8501:8501` working locally.
CMD streamlit run src/alpacatrader/app/dashboard.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
