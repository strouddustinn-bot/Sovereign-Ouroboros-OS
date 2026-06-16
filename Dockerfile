FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml .
COPY ouroboros/ ./ouroboros/

RUN pip install --no-cache-dir -e ".[serve]"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "ouroboros", "serve", "--host", "0.0.0.0", "--port", "8000"]
