FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8765 \
    PORT_HEALTH=8765

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn

COPY app.py /app/app.py

EXPOSE 8765

CMD ["python", "-u", "/app/app.py"]
