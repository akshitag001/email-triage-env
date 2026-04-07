"""Pre-submission validator for the Email Triage OpenEnv project."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from fastapi.testclient import TestClient

from app import app
from env import EmailTriageEnv, Action, Category, Priority


ROOT = Path(__file__).resolve().parent


def ok(message: str) -> None:
    print(f"[PASS] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def check_yaml() -> bool:
    yaml_path = ROOT / "openenv.yaml"
    if not yaml_path.exists():
        fail("openenv.yaml missing")
        return False

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    required_fields = ["name", "version", "tasks", "action_space", "observation_space"]
    for field in required_fields:
        if field not in config:
            fail(f"openenv.yaml missing field: {field}")
            return False

    if len(config["tasks"]) < 3:
        fail("openenv.yaml must define at least 3 tasks")
        return False

    ok("openenv.yaml format and task count")
    return True


def check_env_api() -> bool:
    env = EmailTriageEnv(task="easy")
    reset_result = env.reset()
    if not hasattr(reset_result, "observation") or not hasattr(reset_result, "info"):
        fail("reset() result shape invalid")
        return False

    step_result = env.step(
        Action(category=Category.PROMOTIONAL, priority=Priority.LOW, response_required=False)
    )
    if not (0.0 <= step_result.reward <= 1.0):
        fail("step() reward out of [0,1]")
        return False

    state = env.state()
    for field in ["task", "current_step", "batch_size", "total_reward"]:
        if field not in state:
            fail(f"state() missing field: {field}")
            return False

    ok("typed models + reset()/step()/state() API")
    return True


def check_http_endpoints() -> bool:
    client = TestClient(app)

    health = client.get("/health")
    if health.status_code != 200:
        fail("/health endpoint failed")
        return False

    reset = client.post("/reset", json={"task": "easy"})
    if reset.status_code != 200:
        fail("/reset endpoint failed")
        return False

    step = client.post(
        "/step",
        json={
            "category": "promotional",
            "priority": "low",
            "response_required": False,
        },
    )
    if step.status_code != 200:
        fail("/step endpoint failed")
        return False

    state = client.get("/state")
    if state.status_code != 200:
        fail("/state endpoint failed")
        return False

    ok("HTTP endpoints /reset /step /state")
    return True


def check_inference_runs() -> bool:
    env_vars = os.environ.copy()
    env_vars.pop("OPENAI_API_KEY", None)
    env_vars.pop("HF_TOKEN", None)
    env_vars["LOCAL_ONLY"] = "1"

    proc = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=ROOT,
        env=env_vars,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )

    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        fail("inference.py exited non-zero")
        print(output)
        return False

    if "[START]" not in output or "[STEP]" not in output or "[END]" not in output:
        fail("inference.py log format missing [START]/[STEP]/[END]")
        return False

    scores = [float(match.group(1)) for match in re.finditer(r"score=([0-9]+\.[0-9]+)", output)]
    if not scores:
        fail("No scores found in inference output")
        return False

    if not all(0.0 <= s <= 1.0 for s in scores):
        fail("One or more scores are outside [0, 1]")
        return False

    ok("inference.py runs and emits valid structured scores")
    return True


def check_baseline_contract() -> bool:
    inference_path = ROOT / "inference.py"
    content = inference_path.read_text(encoding="utf-8")

    if "from openai import OpenAI" not in content:
        fail("inference.py must use OpenAI client")
        return False

    if "HF_TOKEN" not in content:
        fail("inference.py must read HF_TOKEN from environment")
        return False

    if "API_BASE_URL" not in content or "MODEL_NAME" not in content:
        fail("inference.py must read API_BASE_URL and MODEL_NAME from environment")
        return False

    ok("baseline uses OpenAI client + HF_TOKEN, API_BASE_URL, MODEL_NAME")
    return True


def main() -> int:
    checks = [
        check_yaml,
        check_env_api,
        check_http_endpoints,
        check_baseline_contract,
        check_inference_runs,
    ]

    passed = 0
    for check in checks:
        try:
            if check():
                passed += 1
        except Exception as exc:
            fail(f"{check.__name__} crashed: {exc}")

    print(f"\nSummary: {passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
