FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl zstd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ollama runs as a second process in this container (see start.sh).
RUN curl -fsSL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst -o /tmp/ollama.tar.zst \
    && tar --use-compress-program=unzstd -xf /tmp/ollama.tar.zst -C /usr/local \
    && rm /tmp/ollama.tar.zst

# Baked in at build time (before COPY . .) so code changes don't bust this layer's cache.
ENV WHISPER_MODEL=base
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

ENV OLLAMA_MODEL=gemma3:4b
ENV OLLAMA_MODEL_LLAMA=llama3.2:3b
# Cap to 1: two models resident at once (Gemma + Llama) OOM'd the container.
ENV OLLAMA_MAX_LOADED_MODELS=1
RUN ollama serve > /tmp/ollama-build.log 2>&1 & \
    OLLAMA_PID=$! && \
    for i in $(seq 1 30); do curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 && break; sleep 1; done && \
    ollama pull gemma3:4b && \
    ollama pull llama3.2:3b && \
    kill $OLLAMA_PID

COPY . .

ENV PORT=8080
EXPOSE 8080
RUN chmod +x start.sh

CMD ["./start.sh"]
