FROM python:3.12-slim AS builder
WORKDIR /opt/app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- runtime stage ----------
FROM python:3.12-slim
RUN addgroup --system app && adduser  --system --ingroup app --disabled-password app

ENV PYTHONUNBUFFERED=1 \
    MCP_PORT=5055 \
    PYTHONPATH=/home/app/.local/lib/python3.12/site-packages \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=3 \
    PYTHONNOUSERSIT=1

WORKDIR /opt/app

COPY --from=builder /install /usr/local
COPY --chown=app:app app.py tools.py ./

USER app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]