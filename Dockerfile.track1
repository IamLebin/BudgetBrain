FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY app app
COPY fireworks fireworks
COPY router router
COPY solvers solvers

CMD ["python", "-m", "app.main"]
