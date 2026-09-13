# Dictation

Turns spoken audio into clean, ready-to-paste text: removes "um"s, fixes self-corrections, adds punctuation.

## Requirements

- Python 3.10 or newer
- `pip`

## Step 1 — Get the code

```bash
git clone https://github.com/sb895/glot-dictation-demo.git
cd glot-dictation-demo
python3 -m venv .venv
source .venv/bin/activate
```

## Step 2 — Install (pick one)

The app needs an AI model to "polish" the text. You can either use one over the internet (simpler), or run one on your own computer (no API key needed). If you're not sure, pick **Option A**.

### Option A — Basic install (recommended): use an external AI over the internet

```bash
pip install -e ".[web,providers]"
```

Then get a free API key and set it as an environment variable. You only need **one** of these:

**Google Gemini** (has a free tier) — get a key at https://aistudio.google.com/apikey
```bash
export GEMINI_API_KEY=paste-your-key-here
```

**OpenAI / ChatGPT** (paid) — get a key at https://platform.openai.com/api-keys
```bash
export OPENAI_API_KEY=paste-your-key-here
```

### Option B — Full install: also run AI models locally on your computer

Do everything in Option A above, then:

1. Install [Ollama](https://ollama.com/download) — this runs AI models on your own machine.
2. Download a model:
   ```bash
   ollama pull gemma3
   ```
3. That's it — no API key needed for this part. Ollama runs in the background automatically once installed.

## Step 3 — Run it

```bash
python -m webapp.app
```

Open http://127.0.0.1:5000 in your browser. Hold the button, speak, release — the cleaned-up text appears.

## If you skip the API key / Ollama part

The app still runs — it just cleans up filler words with simple rules instead of using AI. You'll see this called out on screen so it's never a silent downgrade.
