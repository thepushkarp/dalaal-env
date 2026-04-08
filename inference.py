"""
Inference Script — Dalaal Browser-Use Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from dalaal_env import DalaalEnvAction, DalaalEnvEnv

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-27B")

TASK_NAME = os.getenv("DALAAL_TASK", "todo_add")
BENCHMARK = "dalaal_env"
MAX_STEPS = 15
TEMPERATURE = 0.0
MAX_TOKENS = 300

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a browser automation agent. You interact with web pages by reading
    an accessibility tree and issuing actions.

    ACCESSIBILITY TREE FORMAT:
    Each element has an [ID] followed by its role and properties:
      [1] heading "Page Title"
      [2] textbox "Search" value=""
      [3] button "Submit"
      [4] checkbox "Accept terms" checked=false

    AVAILABLE ACTIONS (respond with exactly one JSON object):
    - Click an element:       {"action_type": "click", "element_id": <id>}
    - Type into an element:   {"action_type": "type", "element_id": <id>, "text": "<text>"}
    - Select a dropdown option: {"action_type": "select_option", "element_id": <id>, "text": "<option label>"}
    - Press a key:            {"action_type": "press_key", "key": "<key name>"}
    - Scroll:                 {"action_type": "scroll", "direction": "up" or "down"}
    - Go back:                {"action_type": "go_back"}
    - Signal task complete:   {"action_type": "done"}

    STRATEGY:
    - Think step by step about what action to take next to accomplish the task.
    - Each action changes the page. After typing text into an input, you typically
      need to click a button to submit it.
    - Do NOT repeat the same action if the page hasn't changed.
    - When the task appears complete (e.g., you can see the expected result in the
      accessibility tree), use {"action_type": "done"}.

    RULES:
    - Respond with ONLY a JSON object. No explanation, no markdown, no extra text.
    - Use element IDs from the current accessibility tree.
    - If you see an error, try a different approach.
""")


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
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def parse_action(text: str) -> DalaalEnvAction:
    """Parse LLM response into a DalaalEnvAction."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()

    data = json.loads(text)
    return DalaalEnvAction(**data)


def build_user_prompt(
    task: str,
    tree: str,
    url: str,
    step: int,
    max_steps: int,
    last_error: Optional[str],
) -> str:
    parts = [
        f"TASK: {task}",
        f"STEP: {step}/{max_steps}",
        f"URL: {url}",
    ]
    if last_error:
        parts.append(f"LAST ACTION ERROR: {last_error}")
    parts.append(f"ACCESSIBILITY TREE:\n{tree}")
    parts.append("Respond with your next action as a JSON object.")
    return "\n\n".join(parts)


def get_action_from_llm(
    client: OpenAI,
    messages: list,
    task: str,
    tree: str,
    url: str,
    step: int,
    max_steps: int,
    last_error: Optional[str],
) -> tuple[DalaalEnvAction, str]:
    """Call the LLM and parse the response into an action. Returns (action, raw_text)."""
    user_prompt = build_user_prompt(task, tree, url, step, max_steps, last_error)
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": text})
        print(f"[DEBUG] LLM response: {text}", flush=True)
        return parse_action(text), text
    except Exception as exc:
        print(f"[DEBUG] LLM/parse error: {exc}", flush=True)
        return DalaalEnvAction(action_type="done"), ""


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        from openenv.core.containers.runtime import LocalDockerProvider
        provider = LocalDockerProvider()
        base_url = provider.start_container(IMAGE_NAME)
        provider.wait_for_ready(base_url, timeout_s=60.0)
        env = DalaalEnvEnv(base_url=base_url, provider=provider)
        # Retry WebSocket connect — server may need extra time after health check passes
        for attempt in range(5):
            try:
                await env.connect()
                break
            except (ConnectionError, OSError) as e:
                if attempt == 4:
                    raise
                print(f"[DEBUG] WS connect attempt {attempt + 1} failed: {e}, retrying...", flush=True)
                await asyncio.sleep(3)
    else:
        env = DalaalEnvEnv(base_url=os.getenv("DALAAL_ENV_URL", "http://localhost:8000"))
        await env.connect()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=TASK_NAME)
        obs = result.observation

        # Conversation history for multi-turn reasoning
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action, raw = get_action_from_llm(
                client=client,
                messages=messages,
                task=obs.task_description,
                tree=obs.accessibility_tree,
                url=obs.url,
                step=step,
                max_steps=obs.max_steps,
                last_error=obs.last_action_error,
            )

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = obs.last_action_error

            rewards.append(reward)
            steps_taken = step

            action_str = f"{action.action_type}({action.element_id or action.text or action.key or ''})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Final reward is the last reward (which encodes success)
        if rewards and rewards[-1] > 0:
            score = rewards[-1]
            success = True
        else:
            score = 0.0
            success = False

        score = min(max(score, 0.0), 1.0)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
