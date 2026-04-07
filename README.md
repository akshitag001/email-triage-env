---
title: OpenEnv Email Triage
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# OpenEnv Email Triage Environment

I built this environment as a real-world email triage task for reinforcement-learning style agents.
The idea is straightforward: the agent reads an incoming email, classifies the message, assigns a
priority, and decides whether a response is needed. That is a common workplace workflow, and it gives
the model a meaningful decision loop to learn from.

## What this project does

The environment exposes the standard OpenEnv-style workflow:

- `reset()` starts a new batch of emails and returns the first observation.
- `step(action)` evaluates one classification decision and returns observation, reward, done, and info.
- `state()` exposes the current episode state for debugging and grading.

There are three graded tasks:

- **Easy**: obvious spam, promo, and urgent support cases.
- **Medium**: mixed internal, support, and sales emails.
- **Hard**: ambiguous emails that force the agent to use context carefully.

The reward is dense, not binary. Each step can earn partial credit for:

- category accuracy
- priority accuracy
- response requirement accuracy
- a small efficiency bonus

That makes the feedback useful during learning instead of waiting until the end of the episode.

## How I approached it

I kept the implementation focused on the submission requirements:

- typed Pydantic models for actions, observations, rewards, and step/reset results
- a deterministic environment so grading is repeatable
- a baseline `inference.py` that uses the OpenAI client and prints the required structured logs
- a Docker setup so the same code can run in Hugging Face Spaces
- a FastAPI service so the space responds to `/reset`, `/step`, `/state`, and `/health`

The code is intentionally plain and readable. The goal was not to build a flashy demo.
The goal was to build something a grader can evaluate cleanly and a human can understand quickly.

## Output format

The baseline script prints exactly these line types:

- `[START]`
- `[STEP]`
- `[END]`

That matters because the evaluator depends on consistent stdout formatting.

## Session note

This submission was shaped during the **OpenEnv Round 1 Bootcamp: Build Your First RL Environment** session.
Thanks to **Ben Burtenshaw** and **Pulkit Aneja** for the walkthrough and guidance.

Bootcamp session reference:

- https://huggingface.co/spaces/akshitag001/email-triage-env

If you want to include the bootcamp screenshot in the repository later, add the image as a file and
replace this link with a local markdown image.

## Files included in this submission

- `env.py`
- `inference.py`
- `grader.py`
- `app.py`
- `openenv.yaml`
- `requirements.txt`
- `Dockerfile`

## Local run

If you want to run it locally, set the required environment variables and start the baseline:

```powershell
$env:HF_TOKEN="your-token-here"
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
python inference.py
```

## Submission note

No credentials are stored in the repository. Tokens are loaded only from environment variables.
