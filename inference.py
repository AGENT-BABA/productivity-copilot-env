import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from productivity_env import ProductivityAction, ProductivityEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("PRODUCTIVITY_TASK", "triage")
BENCHMARK = os.getenv("PRODUCTIVITY_BENCHMARK", "productivity_copilot")
MAX_STEPS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1

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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

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
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "WAIT|"

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Check if we should use docker image or local class
    image_name = os.getenv("IMAGE_NAME")
    if image_name:
        env = await ProductivityEnv.from_docker_image(image_name)
    else:
        env = ProductivityEnv(task_name=TASK_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        last_obs = result.observation.dict()
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

            result = await env.step(ProductivityAction(action_type=action_type, message=message))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_obs = obs.dict()
            last_reward = reward

            log_step(step=step, action=action_type, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_type} -> reward {reward:+.2f}")

            if done:
                break

        # Calculate score (normalized)
        # Assume max reward per step is 0.1 from the base reward calculation.
        MAX_TOTAL_REWARD = MAX_STEPS * 0.1
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            # Safely close if required
            if hasattr(env, 'close'):
                await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
