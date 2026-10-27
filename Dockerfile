# One image for the API and the UI. CPU-only PyTorch (the Linux index in pyproject.toml);
# the LLM runs in Ollama outside this image.
FROM python:3.12.7-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# dependencies first, so code changes don't reinstall them
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY .streamlit ./.streamlit
RUN uv sync --locked --no-dev

# run as a normal user; the model cache volume is mounted on its home
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "--factory", "eurolex.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
