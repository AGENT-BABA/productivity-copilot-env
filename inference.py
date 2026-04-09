import asyncio
import contextlib
import io
import os
import textwrap
from types import SimpleNamespace
from typing import Any, List, Optional

from openai import OpenAI

from openenv.core import EnvClient

from productivity_env import ProductivityAction, ProductivityEnv

API_KEY = os.environ["API_KEY"] if "API_KEY" in os.environ else os.getenv("HF_TOKEN")
API_BASE_URL = os.environ["API_BASE_URL"] if "API_BASE_URL" in os.environ else "https://router.huggingface.co/v1"
MODEL_NAME = os.environ["MODEL_NAME"] if "MODEL_NAME" in os.environ else "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("PRODUCTIVITY_TASK", "triage")
BENCHMARK = os.getenv("PRODUCTIVITY_BENCHMARK", "productivity_copilot")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
MAX_STEPS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI productivity coach managing a simulated human worker.
    Each turn you observe the human's condition (stress, focus, distraction, failure probability).
    Your goal is to decrease failure probability while keeping stress below 8.
    Reply with exactly one action in the format: ACTION_TYPE|Message
    Available actions:
    WAIT|
    FORCE_BREAK|Take a break!
    BLOCK_SOCIAL_MEDIA|Blocked
    SEND_NUDGE|You can do this!
    
    Choose wisely based on the observation!
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs_dict: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Observation: {obs_dict}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Consider the human's stress and failure probability. Send your next action.
        """
    ).strip()

def get_model_message(client: OpenAI, step: int, obs_dict: dict, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, obs_dict, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "WAIT|"
    except Exception:
        return "WAIT|"


def create_client() -> OpenAI:
    if "API_BASE_URL" in os.environ and "API_KEY" in os.environ:
        return OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def warmup_llm_proxy(client: OpenAI) -> None:
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Reply with exactly WAIT|warmup"},
                {"role": "user", "content": "warmup"},
            ],
            temperature=0,
            max_tokens=8,
            stream=False,
        )
    except Exception:
        pass


def normalize_result(result: Any) -> SimpleNamespace:
    if hasattr(result, "observation"):
        observation = result.observation
        reward = result.reward
        done = result.done
    else:
        observation = result
        reward = getattr(result, "reward", None)
        done = getattr(result, "done", False)

    return SimpleNamespace(observation=observation, reward=reward, done=done)

async def main() -> None:
    client = None
    env = None

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        client = create_client()
        warmup_llm_proxy(client)

        if IMAGE_NAME:
            env = await EnvClient.from_docker_image(
                IMAGE_NAME,
                env_vars={"PRODUCTIVITY_TASK": TASK_NAME},
            )
            result = await env.reset(task_name=TASK_NAME)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                env = ProductivityEnv(task_name=TASK_NAME)
            result = normalize_result(env.reset(task_name=TASK_NAME))

        result = normalize_result(result)
        last_obs = result.observation.model_dump()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            response = get_model_message(client, step, last_obs, last_reward, history)
            parts = response.split("|", 1)
            action_type = parts[0] if len(parts) > 0 else "WAIT"
            message = parts[1] if len(parts) > 1 else ""

            valid_actions = ["WAIT", "FORCE_BREAK", "BLOCK_SOCIAL_MEDIA", "SEND_NUDGE"]
            if action_type not in valid_actions:
                action_type = "WAIT"

            if IMAGE_NAME:
                result = await env.step(ProductivityAction(action_type=action_type, message=message))
            else:
                result = normalize_result(
                    env.step(ProductivityAction(action_type=action_type, message=message))
                )
            result = normalize_result(result)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = getattr(obs, "last_action_error", None)

            rewards.append(reward)
            steps_taken = step
            last_obs = obs.model_dump()
            last_reward = reward

            log_step(step=step, action=action_type, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_type} -> reward {reward:+.2f}")

            if done:
                break

        score = max(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception:
        success = False

    finally:
        try:
            if env is not None and hasattr(env, "close"):
                await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        log_end(success=False, steps=0, score=0.0, rewards=[])
