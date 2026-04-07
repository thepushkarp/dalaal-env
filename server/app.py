"""
FastAPI application for the Dalaal Browser-Use Environment.

Endpoints:
    - POST /reset: Reset the environment (pass task name in body)
    - POST /step: Execute a browser action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import DalaalEnvAction, DalaalEnvObservation
    from .dalaal_env_environment import DalaalEnvEnvironment
except (ImportError, SystemError):
    try:
        from models import DalaalEnvAction, DalaalEnvObservation
        from server.dalaal_env_environment import DalaalEnvEnvironment
    except ImportError:
        from dalaal_env.models import DalaalEnvAction, DalaalEnvObservation
        from dalaal_env.server.dalaal_env_environment import DalaalEnvEnvironment


app = create_app(
    DalaalEnvEnvironment,
    DalaalEnvAction,
    DalaalEnvObservation,
    env_name="dalaal_env",
    max_concurrent_envs=1,
)


# ── Landing page & info endpoints ────────────────────────────────────

from fastapi.responses import HTMLResponse, JSONResponse

try:
    from server.tasks import TASKS
except ImportError:
    try:
        from .tasks import TASKS
    except (ImportError, SystemError):
        from dalaal_env.server.tasks import TASKS


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    task_rows = ""
    for tid in sorted(TASKS):
        t = TASKS[tid]
        task_rows += f"""
        <tr>
          <td><code>{t.id}</code></td>
          <td>{t.description}</td>
          <td>{t.site_file}</td>
          <td>{t.max_steps}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Dalaal Env — Browser-Use RL Environment</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }}
    .hero {{ text-align: center; padding: 48px 24px 32px; }}
    .hero h1 {{ font-size: 2.5rem; color: #f8fafc; margin-bottom: 8px; }}
    .hero .subtitle {{ font-size: 1.1rem; color: #94a3b8; max-width: 600px; margin: 0 auto; }}
    .badge {{ display: inline-block; padding: 4px 12px; background: #1e40af; color: #93c5fd; border-radius: 16px; font-size: 13px; margin: 16px 4px 0; }}
    .container {{ max-width: 960px; margin: 0 auto; padding: 0 24px 48px; }}
    .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; border: 1px solid #334155; }}
    .card h2 {{ font-size: 1.3rem; color: #f1f5f9; margin-bottom: 12px; }}
    .card p {{ color: #94a3b8; font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 12px; }}
    .stat {{ background: #0f172a; padding: 16px; border-radius: 8px; text-align: center; }}
    .stat .num {{ font-size: 2rem; font-weight: bold; color: #60a5fa; }}
    .stat .lbl {{ font-size: 13px; color: #94a3b8; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th {{ text-align: left; padding: 10px 12px; background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
    td {{ padding: 10px 12px; border-bottom: 1px solid #334155; }}
    code {{ background: #0f172a; padding: 2px 6px; border-radius: 4px; font-size: 13px; color: #60a5fa; }}
    .arch {{ display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; align-items: center; gap: 12px; text-align: center; margin: 16px 0; }}
    .arch-box {{ background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; }}
    .arch-arrow {{ color: #60a5fa; font-size: 24px; }}
    .endpoint {{ display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid #334155; }}
    .method {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; min-width: 50px; text-align: center; }}
    .method.get {{ background: #065f46; color: #6ee7b7; }}
    .method.post {{ background: #92400e; color: #fcd34d; }}
    .method.ws {{ background: #5b21b6; color: #c4b5fd; }}
    a {{ color: #60a5fa; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .footer {{ text-align: center; padding: 24px; color: #475569; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>Dalaal Env</h1>
    <p class="subtitle">A reinforcement learning environment where LLM agents learn to navigate and interact with web pages through accessibility tree observations.</p>
    <span class="badge">OpenEnv Framework</span>
    <span class="badge">Playwright + CDP</span>
    <span class="badge">19 Tasks</span>
  </div>

  <div class="container">
    <div class="card">
      <h2>Overview</h2>
      <div class="grid">
        <div class="stat"><div class="num">{len(TASKS)}</div><div class="lbl">Browser Tasks</div></div>
        <div class="stat"><div class="num">12</div><div class="lbl">Mock Websites</div></div>
        <div class="stat"><div class="num">7</div><div class="lbl">Action Types</div></div>
        <div class="stat"><div class="num">6</div><div class="lbl">Benchmark Sources</div></div>
      </div>
    </div>

    <div class="card">
      <h2>Architecture</h2>
      <div class="arch">
        <div class="arch-box"><strong>LLM Agent</strong><br><small>Qwen / GPT / etc.</small></div>
        <div class="arch-arrow">&rarr;</div>
        <div class="arch-box"><strong>Dalaal Env</strong><br><small>FastAPI + OpenEnv</small></div>
        <div class="arch-arrow">&rarr;</div>
        <div class="arch-box"><strong>Browser</strong><br><small>Playwright + Chromium</small></div>
      </div>
      <p>The agent observes a numbered <strong>accessibility tree</strong> (extracted via CDP) and emits structured actions (click, type, select, scroll, etc.). The environment executes actions in a headless browser and evaluates task-specific JavaScript success criteria.</p>
    </div>

    <div class="card">
      <h2>API Endpoints</h2>
      <div class="endpoint"><span class="method ws">WS</span> <code>/ws</code> <span style="color:#94a3b8">— WebSocket for persistent sessions (primary)</span></div>
      <div class="endpoint"><span class="method post">POST</span> <code>/reset</code> <span style="color:#94a3b8">— Reset environment with a task</span></div>
      <div class="endpoint"><span class="method post">POST</span> <code>/step</code> <span style="color:#94a3b8">— Execute a browser action</span></div>
      <div class="endpoint"><span class="method get">GET</span> <code>/state</code> <span style="color:#94a3b8">— Get current observation</span></div>
      <div class="endpoint"><span class="method get">GET</span> <code>/tasks</code> <span style="color:#94a3b8">— List all available tasks (JSON)</span></div>
      <div class="endpoint"><span class="method get">GET</span> <code>/docs</code> <span style="color:#94a3b8">— Interactive API documentation (Swagger)</span></div>
    </div>

    <div class="card">
      <h2>Action Space</h2>
      <p>Each action is a JSON object with <code>action_type</code> and relevant parameters:</p>
      <table style="margin-top:12px">
        <tr><th>Action</th><th>Parameters</th><th>Description</th></tr>
        <tr><td><code>click</code></td><td><code>element_id</code></td><td>Click an element by its accessibility tree ID</td></tr>
        <tr><td><code>type</code></td><td><code>element_id</code>, <code>text</code></td><td>Type text into an input field</td></tr>
        <tr><td><code>select</code></td><td><code>element_id</code>, <code>text</code></td><td>Select a dropdown option by visible text</td></tr>
        <tr><td><code>press_key</code></td><td><code>key</code></td><td>Press a keyboard key (Enter, Tab, etc.)</td></tr>
        <tr><td><code>scroll</code></td><td><code>direction</code></td><td>Scroll the page (up/down)</td></tr>
        <tr><td><code>wait</code></td><td>—</td><td>Wait for page to settle</td></tr>
        <tr><td><code>done</code></td><td>—</td><td>Signal task completion</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Available Tasks</h2>
      <table>
        <tr><th>Task ID</th><th>Description</th><th>Mock Site</th><th>Max Steps</th></tr>
        {task_rows}
      </table>
    </div>

    <div class="card">
      <h2>Reward Structure</h2>
      <p><strong>+1.0</strong> on task success &nbsp;|&nbsp; <strong>-0.01</strong> per step penalty &nbsp;|&nbsp; Clamped to <strong>[0, 1]</strong></p>
      <p style="margin-top:8px">Example: completing a task in 4 steps &rarr; reward = max(0, 1.0 - 0.04) = <strong>0.96</strong></p>
    </div>

    <div class="card">
      <h2>Quick Start</h2>
      <p style="margin-bottom:8px">Run inference against this environment:</p>
      <code style="display:block;padding:12px;background:#0f172a;border-radius:8px;white-space:pre;line-height:1.8">API_BASE_URL=https://router.huggingface.co/v1 \\
MODEL_NAME=Qwen/Qwen3.5-27B \\
HF_TOKEN=hf_... \\
DALAAL_TASK=todo_add \\
python inference.py</code>
    </div>
  </div>

  <div class="footer">
    Built for the <a href="https://github.com/meta-pytorch/openenv">OpenEnv</a> Round 1 Bootcamp Hackathon
  </div>
</body>
</html>"""


@app.get("/tasks")
async def list_tasks_endpoint():
    return JSONResponse({
        tid: {"description": t.description, "site_file": t.site_file, "max_steps": t.max_steps}
        for tid, t in sorted(TASKS.items())
    })


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
