FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts/novaspine-server.py /app/scripts/novaspine-server.py

RUN pip install --no-cache-dir -e .

EXPOSE 8420

CMD ["novaspine", "serve", "--host", "0.0.0.0", "--port", "8420"]
