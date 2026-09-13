#!/bin/sh
# Runs Ollama alongside the web app in the same container.
set -e

ollama serve > /tmp/ollama.log 2>&1 &

echo "Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is ready."
    break
  fi
  sleep 1
done

exec gunicorn -w 1 -b 0.0.0.0:${PORT} --timeout 240 webapp.app:app
