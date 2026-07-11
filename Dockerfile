FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY app app
COPY fireworks fireworks
COPY router router
COPY solvers solvers
COPY api api
COPY index.html index.html
COPY server.py server.py

EXPOSE 8000

CMD ["python", "server.py"]
