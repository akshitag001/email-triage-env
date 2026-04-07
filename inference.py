"""
Inference Script: Email Triage Agent
Evaluates an LLM agent on email classification tasks
"""

import asyncio
import os
import textwrap
from typing import List, Optional
import json

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None
from env import EmailTriageEnv, Action, Category, Priority

# Environment configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")  # Optional for local testing, required for submission
TASK_NAME = os.getenv("MY_ENV_TASK", "email-triage")
BENCHMARK = os.getenv("MY_ENV_BENCHMARK", "email-triage-v1")

# Hyperparameters
MAX_STEPS = 10
TEMPERATURE = 0.3
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.6

# Max possible reward calculation
_MAX_REWARD_PER_STEP = 1.0  # Each step can reward up to 1.0
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert email triage assistant. Your task is to classify incoming emails
    and determine their priority and whether they require a response.
    
    For each email, respond with ONLY a JSON object (no markdown, no explanation):
    {
      "category": "spam|promotional|support|sales|internal|other",
      "priority": "low|medium|high|urgent",
      "response_required": true|false
    }
    
    Guidelines:
    - SPAM: Unsolicited offers, phishing, scams
    - PROMOTIONAL: Marketing emails, newsletters
    - SUPPORT: Technical/account issues, urgent matters
    - SALES: B2B sales pitches
    - INTERNAL: From company, team, or colleagues
    - OTHER: Everything else
    
    Priority:
    - LOW: Can wait, not important
    - MEDIUM: Should address soon
    - HIGH: Important, needs attention
    - URGENT: Critical, must address immediately
    
    Response:
    - true: Email requires an active response
    - false: FYI only, acknowledgment only, or spam/promotional
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    """Log step result"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = str(success).lower()
    print(
        f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(observation, history: List[str]) -> str:
    """Build prompt for model"""
    history_block = "\n".join(history[-2:]) if history else "None"
    return textwrap.dedent(
        f"""
        Email #{observation.current_step} of {observation.total_emails_in_batch}
        
        From: {observation.sender}
        Subject: {observation.subject}
        Attachment: {'Yes' if observation.has_attachment else 'No'}
        
        Preview:
        {observation.body_preview}
        
        Previous classifications:
        {history_block}
        
        Classify this email:
        """
    ).strip()


def get_model_action(
    client, observation, history: List[str]
) -> Optional[Action]:
    """Get model classification"""
    user_prompt = build_user_prompt(observation, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = (completion.choices[0].message.content or "").strip()

        # Parse JSON response
        try:
            data = json.loads(response_text)
            action = Action(
                category=Category(data.get("category", "other")),
                priority=Priority(data.get("priority", "low")),
                response_required=bool(data.get("response_required", False)),
            )
            return action
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[DEBUG] Failed to parse model response: {response_text}", flush=True)
            # Return safe default
            return Action(
                category=Category.OTHER,
                priority=Priority.LOW,
                response_required=False,
            )

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return Action(
            category=Category.OTHER, priority=Priority.LOW, response_required=False
        )


def get_heuristic_action(observation) -> Action:
    """Offline classifier for local runs without any API key."""
    text = (
        f"{observation.sender} {observation.subject} {observation.body_preview}"
    ).lower()

    if any(k in text for k in ["critical", "unusual account activity", "failed login", "payment failed", "action required", "urgent", "eod"]):
        if "@company.com" in observation.sender or "team@" in observation.sender or "hr@" in observation.sender:
            return Action(category=Category.INTERNAL, priority=Priority.URGENT, response_required=True)
        return Action(category=Category.SUPPORT, priority=Priority.URGENT, response_required=True)

    if any(k in text for k in ["newsletter", "deals", "benefits enrollment", "marketing"]):
        return Action(category=Category.PROMOTIONAL, priority=Priority.LOW, response_required=False)

    if any(k in text for k in ["selected", "click here", "limited time offer", "hot leads"]):
        return Action(category=Category.SPAM, priority=Priority.LOW, response_required=False)

    if any(k in text for k in ["sales", "enterprise", "let's talk", "solution"]):
        return Action(category=Category.SALES, priority=Priority.MEDIUM, response_required=False)

    if any(k in text for k in ["github", "workflow", "ci", "project", "boss@company.com", "hr@company.com"]):
        return Action(category=Category.INTERNAL, priority=Priority.MEDIUM, response_required=True)

    if any(k in text for k in ["support", "issue", "failed", "alert"]):
        return Action(category=Category.SUPPORT, priority=Priority.HIGH, response_required=True)

    return Action(category=Category.OTHER, priority=Priority.LOW, response_required=False)


async def run_task(task_name: str, client=None, use_local_heuristic: bool = False) -> tuple:
    """Run single task and return metrics"""
    env = EmailTriageEnv(task=task_name)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False
    last_error = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if use_local_heuristic:
                action = get_heuristic_action(obs)
            else:
                action = get_model_action(client, obs, history)

            if action is None:
                last_error = "Failed to get model action"
                log_step(
                    step=step,
                    action="",
                    reward=0.0,
                    done=True,
                    error=last_error,
                )
                break

            step_result = env.step(action)
            reward = step_result.reward
            done = step_result.done
            obs = step_result.observation
            info = step_result.info

            rewards.append(reward)
            steps_taken = step

            # Format action for logging
            action_str = f"classify(category={action.category.value},priority={action.priority.value})"

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            # Track history
            history.append(
                f"Step {step}: {action.category.value}/{action.priority.value} -> reward {reward:.2f}"
            )

            if done:
                break

        # Calculate score as per-email average for the current task batch.
        # This keeps scores comparable across easy/medium/hard task sizes.
        score = (sum(rewards) / len(rewards)) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        last_error = str(e)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return task_name, score, steps_taken, sum(rewards)


async def main() -> None:
    """Main evaluation loop"""
    force_local = os.getenv("LOCAL_ONLY", "").lower() in {"1", "true", "yes"}
    use_local_heuristic = force_local or not HF_TOKEN or OpenAI is None
    client = None

    if not use_local_heuristic:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    print(
        f"[DEBUG] Using mode: {'local-heuristic' if use_local_heuristic else 'llm-api'}",
        flush=True,
    )
    print(f"[DEBUG] Using model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] API Base: {API_BASE_URL}", flush=True)
    print(f"[DEBUG] Starting evaluation on 3 tasks", flush=True)

    # Run all three tasks
    tasks = ["easy", "medium", "hard"]
    results = []

    for task in tasks:
        print(f"\n[DEBUG] Running task: {task}", flush=True)
        task_name, score, steps, total_reward = await run_task(
            task_name=task,
            client=client,
            use_local_heuristic=use_local_heuristic,
        )
        results.append((task_name, score, steps, total_reward))
        print(f"[DEBUG] Completed {task}: score={score:.3f}", flush=True)

    print("\n[DEBUG] === Final Results ===", flush=True)
    for task_name, score, steps, total_reward in results:
        print(
            f"[DEBUG] {task_name}: score={score:.3f}, steps={steps}, reward={total_reward:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
