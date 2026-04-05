# OpenEnv Course: Complete Research Findings

Source repo: https://github.com/huggingface/openenv-course  
Core framework: https://github.com/meta-pytorch/OpenEnv  
Status: Research complete — all findings below are from direct source code inspection.

---

## Course Structure

5 modules, each ~45–60 min, with a README (concepts) + Jupyter notebook (hands-on):

| # | Module | Focus |
|---|--------|-------|
| 1 | Why OpenEnv? | RL loop, Gym limitations, OpenEnv architecture, 3-method interface |
| 2 | Using Existing Environments | Environment Hub, typed models, writing policies, switching games |
| 3 | Deploying Environments | Local uvicorn, Docker, HF Spaces, `openenv push` |
| 4 | **Building Your Own Environment** | 3-component pattern, scaffold → deploy — this is the hackathon module |
| 5 | Training with OpenEnv + TRL | GRPO, reward functions, Wordle fine-tuning, rollout functions |

---

## Core Philosophy

**RL environments are microservices.** Not a Gymnasium subclass that runs in-process — a Docker container exposing a WebSocket/HTTP API. The training code connects over the network.

```
Training Code
    │  WebSocket /ws
    ▼
Docker Container (HF Space / local / cloud)
    └── FastAPI Server
        └── Environment (reset, step, state)
            └── Game/Simulation Logic
```

---

## The 3-Method Interface (Universal)

Every single OpenEnv environment — regardless of what it does — exposes:

```python
env.reset(**kwargs)  → StepResult[ObsT]   # start new episode
env.step(action)     → StepResult[ObsT]   # take action
env.state()          → StateT              # episode metadata
```

`StepResult` is:
```python
@dataclass
class StepResult(Generic[ObsT]):
    observation: ObsT
    reward: Optional[float] = None
    done: bool = False
```

---

## The 3-Component Pattern (File Structure)

```
my_env/
├── models.py              ← Pydantic types: Action, Observation, State
├── client.py              ← WebSocket client (what training code imports)
├── __init__.py            ← exports Action, Observation, Env classes
├── server/
│   ├── environment.py     ← Game logic (subclasses Environment ABC)
│   ├── app.py             ← FastAPI wiring via create_app()
│   └── Dockerfile         ← Container definition
├── openenv.yaml           ← Manifest (name, type, runtime, app, port)
├── pyproject.toml         ← Package metadata + [project.scripts] server entry
└── uv.lock                ← Locked deps (required by validator)
```

---

## Base Classes — Exact Signatures

### `Action` (Pydantic BaseModel)
```python
# from openenv.core.env_server import Action
class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Add your own fields as class attributes
```

### `Observation` (Pydantic BaseModel)
```python
# from openenv.core.env_server import Observation
class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    done: bool = Field(default=False)
    reward: bool | int | float | None = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Add your own fields as class attributes
```

### `State` (Pydantic BaseModel)
```python
# from openenv.core.env_server import State
class State(BaseModel):
    model_config = ConfigDict(extra="allow")  # flexible
    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(default=0, ge=0)
    # Add your own fields as class attributes
```

### `Environment` ABC
```python
# from openenv.core.env_server.interfaces import Environment
class Environment(ABC, Generic[ActT, ObsT, StateT]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False  # set True for concurrent WebSocket

    @abstractmethod
    def reset(self, seed=None, episode_id=None, **kwargs) -> ObsT: ...

    @abstractmethod
    def step(self, action: ActT, timeout_s=None, **kwargs) -> ObsT: ...

    @property
    @abstractmethod
    def state(self) -> StateT: ...

    # Optional overrides:
    async def reset_async(...) -> ObsT: ...    # default calls sync reset
    async def step_async(...) -> ObsT: ...    # default calls sync step
    def get_metadata(self) -> EnvironmentMetadata: ...
    def close(self) -> None: ...
```

### `EnvClient` ABC
```python
# from openenv.core.env_client import EnvClient  (or openenv.core import EnvClient)
class EnvClient(ABC, Generic[ActT, ObsT, StateT]):
    @abstractmethod
    def _step_payload(self, action: ActT) -> Dict[str, Any]: ...

    @abstractmethod
    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ObsT]: ...

    @abstractmethod
    def _parse_state(self, payload: Dict[str, Any]) -> StateT: ...

    # Provided by base class (no override needed):
    async def reset(**kwargs) -> StepResult[ObsT]: ...
    async def step(action) -> StepResult[ObsT]: ...
    async def state() -> StateT: ...
    def sync() -> SyncEnvClient: ...  # synchronous wrapper
```

---

## Exact File Templates (from `openenv init`)

### `models.py`
```python
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class MyEnvAction(Action):
    message: str = Field(..., description="...")

class MyEnvObservation(Observation):
    # done and reward are inherited from Observation
    result: str = Field(default="", description="...")
```

### `server/environment.py`
```python
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import MyEnvAction, MyEnvObservation

class MyEnvEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True  # for multi-session WebSocket

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, seed=None, episode_id=None, **kwargs) -> MyEnvObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return MyEnvObservation(done=False, reward=0.0, result="ready")

    def step(self, action: MyEnvAction, timeout_s=None, **kwargs) -> MyEnvObservation:
        self._state.step_count += 1
        # game logic here
        return MyEnvObservation(done=False, reward=0.0, result=action.message)

    @property
    def state(self) -> State:
        return self._state
```

### `server/app.py`
```python
from openenv.core.env_server.http_server import create_app
from models import MyEnvAction, MyEnvObservation
from server.my_env_environment import MyEnvEnvironment

app = create_app(
    MyEnvEnvironment,          # factory (the class itself)
    MyEnvAction,
    MyEnvObservation,
    env_name="my_env",
    max_concurrent_envs=1,     # raise for concurrent clients
)

def main(host="0.0.0.0", port=8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
```

### `client.py`
```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import MyEnvAction, MyEnvObservation

class MyEnv(EnvClient[MyEnvAction, MyEnvObservation, State]):
    def _step_payload(self, action: MyEnvAction) -> dict:
        return {"message": action.message}

    def _parse_result(self, payload: dict) -> StepResult[MyEnvObservation]:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=MyEnvObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                result=obs_data.get("result", ""),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
```

### `openenv.yaml`
```yaml
spec_version: 1
name: my_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

### `pyproject.toml` (key fields)
```toml
[project]
name = "openenv-my-env"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "openenv-core[core]>=0.2.2",
    "fastapi>=0.115.0",
    "uvicorn>=0.24.0",
    # environment-specific deps here
]

[project.scripts]
server = "my_env.server.app:main"   # REQUIRED by validator

[tool.setuptools]
packages = ["my_env", "my_env.server"]
package-dir = { "my_env" = ".", "my_env.server" = "server" }
```

---

## HTTP Endpoints Exposed by Every Environment

`create_app()` auto-creates all of these:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ws` | WebSocket | Persistent session (primary interface) |
| `/reset` | POST | HTTP reset |
| `/step` | POST | HTTP step |
| `/state` | GET | Episode metadata |
| `/schema` | GET | Action/observation/state JSON schemas |
| `/health` | GET | `{"status": "healthy"}` |
| `/metadata` | GET | name, description, version |
| `/mcp` | POST | JSON-RPC 2.0 (MCP protocol) |
| `/openapi.json` | GET | OpenAPI spec |
| `/web` | GET | Gradio UI |
| `/docs` | GET | Swagger docs |

WebSocket message types:
- `{"type": "reset", "data": {...}}` → `{"type": "observation", "data": {...}}`
- `{"type": "step", "data": {...}}` → `{"type": "observation", "data": {...}}`
- `{"type": "state"}` → `{"type": "state", "data": {...}}`
- `{"type": "close"}` → closes session

---

## Validation Criteria (`openenv validate`)

### Local validation — checks these MUST pass:

| Check | What's verified |
|-------|----------------|
| `pyproject.toml` exists | File present |
| `uv.lock` exists | Run `uv lock` to generate |
| `[project.scripts]` has `server` | Entry point defined |
| Server entry has `:main` | Points to `main()` function |
| `openenv-core>=0.2.0` in deps | Package dependency present |
| `server/app.py` exists | Server module present |
| `server/app.py` has `def main(` | Main function defined |
| `server/app.py` has `__name__` guard | Callable as script |

### Runtime validation — checks a live server:

| Criterion ID | What's checked |
|-------------|----------------|
| `openapi_version_available` | `GET /openapi.json` returns `info.version` |
| `health_endpoint` | `GET /health` returns `{"status": "healthy"}` |
| `metadata_endpoint` | `GET /metadata` returns `name` + `description` strings |
| `schema_endpoint` | `GET /schema` returns `action`, `observation`, `state` dicts |
| `mcp_endpoint` | `POST /mcp` returns JSON-RPC 2.0 payload |
| `mode_endpoint_consistency` | OpenAPI paths match simulation mode (`/reset`, `/step`, `/state` present) |

Run it:
```bash
openenv validate                        # local structure check
openenv validate --url http://localhost:8000  # live server check
```

---

## Available Example Environments (HF Hub)

From `meta-pytorch/OpenEnv/envs/`:

| Environment | Type | URL pattern |
|-------------|------|-------------|
| `echo_env` | MCP echo | `openenv-echo-env.hf.space` |
| `openspiel_env` | Games (Catch, TTT, Blackjack, etc.) | `openenv-openspiel-catch.hf.space` |
| `textarena_env` | Wordle/TextArena | `burtenshaw-textarena.hf.space` |
| `coding_env` | Code execution | — |
| `chess_env` | Chess | — |
| `maze_env` | Grid navigation | — |
| `snake_env` | Snake game | — |
| `grid_world_env` | Grid world | — |
| `reasoning_gym_env` | Math/reasoning | — |
| `browsergym_env` | Browser automation | — |
| `websearch_env` | Web search | — |
| `finqa_env` | Financial QA | — |
| ... | 25+ total | — |

---

## Dependencies Required

### Minimum (server side)
```
openenv-core>=0.2.2   # installs fastapi, uvicorn, pydantic, websockets, fastmcp, gradio
```

### Install
```bash
pip install openenv-core
# or
uv add openenv-core
```

### Client side (training code)
```bash
pip install openenv-core
git clone https://github.com/meta-pytorch/OpenEnv.git
# then add OpenEnv/ and OpenEnv/src to sys.path
```

### Full course deps
```
openenv-core>=0.2.2
fastapi>=0.104.0
uvicorn>=0.24.0
fastmcp>=3.0.0
pydantic>=2.0.0
trl>=0.17.0             # Module 5 only
transformers>=4.40.0    # Module 5 only
datasets>=2.18.0        # Module 5 only
accelerate>=0.28.0      # Module 5 only
huggingface-hub>=0.22.0
trackio                 # experiment tracking, Module 5
```

---

## CLI Commands

```bash
openenv init <env_name>                      # scaffold new environment
openenv validate                             # check local structure
openenv validate --url http://localhost:8000 # check live server
openenv push --repo-id username/my-env       # deploy to HF Spaces
openenv build                                # build Docker image
openenv serve                                # run locally
```

After `openenv push`, available at:
- `https://username-my-env.hf.space` — API
- `https://username-my-env.hf.space/web` — Gradio UI
- `https://username-my-env.hf.space/docs` — Swagger
- `https://username-my-env.hf.space/health` — health check

---

## Two Environment Patterns (Standard vs MCP)

### Pattern A: Standard reset/step (most environments)
- Subclass `Environment` ABC
- Define `Action`, `Observation`, `State` Pydantic models
- Client subclasses `EnvClient`
- Use `create_app(EnvClass, ActionCls, ObsCls)`

### Pattern B: MCP tool-calling (echo_env style)
- Subclass `MCPEnvironment` (wraps FastMCP)
- Define tools with `@mcp.tool` decorator
- Client uses `MCPToolClient` with `list_tools()` / `call_tool(name, **kwargs)`
- Use `create_app(EchoEnv, CallToolAction, CallToolObservation)`

For a hackathon, **Pattern A is simpler and more standard**.

---

## Key Implementation Notes

1. **`SUPPORTS_CONCURRENT_SESSIONS = True`** in your Environment class — required if you want multiple simultaneous WebSocket clients (e.g., during parallel GRPO rollouts).

2. **`done` and `reward` are inherited** from `Observation` base — don't re-declare them in your subclass.

3. **`episode_id` and `step_count` are inherited** from `State` base — don't re-declare them.

4. **`Action` has `extra="forbid"`** — any unknown field in the action payload raises a validation error. Be exact.

5. **Client's `_parse_result` receives the full response dict** — observation fields are nested under `payload["observation"]`, while `done` and `reward` are at the top level alongside it.

6. **`uv.lock` is required** by the validator — run `uv lock` in the env directory before pushing.

7. **`server/app.py` must have `def main(` and `if __name__ == "__main__":`** — the validator checks for both.

8. **`max_concurrent_envs`** defaults to 1 in the template — increase it for GRPO training that runs many parallel rollouts.

9. **Import path pattern**: the environment module supports both in-repo (`from ..models import ...`) and standalone (`from models import ...`) imports via try/except — copy this pattern.

10. **The `.sync()` wrapper** makes the async client synchronous — use it in notebooks and scripts: `with MyEnv(base_url=...).sync() as env:`.

---

## TRL + OpenEnv Integration (Module 5 / GRPO Training)

```python
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[reward_correct, reward_greens],
    rollout_func=rollout_func,    # your function that plays episodes
    train_dataset=dataset,
    args=GRPOConfig(
        num_generations=2,
        max_completion_length=8,
        use_vllm=True,
        vllm_mode="colocate",
        gradient_accumulation_steps=64,
        per_device_train_batch_size=1,
        report_to="trackio",
    ),
)
trainer.train()
```

The rollout function signature:
```python
def rollout_func(trainer, env, tokenizer, prompt, system_prompt, max_turns):
    result = env.reset()
    for turn in range(max_turns):
        if result.done: break
        # generate completion with model
        # send to env.step()
        # collect prompt_ids, completion_ids, logprobs, rewards
    return dict(...)
```

Hardware for Module 5: A100 40GB, ~90 min training time, ~37GB peak memory.

---

## Submission Checklist (Hackathon)

Based on validation criteria, a compliant submission must:

- [ ] `openenv.yaml` at root with `spec_version`, `name`, `type`, `runtime`, `app`, `port`
- [ ] `pyproject.toml` with `openenv-core>=0.2.0` dep and `[project.scripts] server = "...app:main"`
- [ ] `uv.lock` generated (`uv lock`)
- [ ] `server/app.py` with `create_app(...)`, `def main(...)`, `if __name__ == "__main__":` guard
- [ ] `server/Dockerfile` for containerization
- [ ] `models.py` with Pydantic Action/Observation/State subclasses
- [ ] `client.py` with EnvClient subclass implementing `_step_payload`, `_parse_result`, `_parse_state`
- [ ] `__init__.py` exporting Action, Observation, Env classes
- [ ] `GET /health` returns `{"status": "healthy"}`
- [ ] `GET /metadata` returns `{"name": "...", "description": "..."}`
- [ ] `GET /schema` returns `{"action": {...}, "observation": {...}, "state": {...}}`
- [ ] `POST /mcp` returns JSON-RPC 2.0 payload
- [ ] `SUPPORTS_CONCURRENT_SESSIONS = True` if parallel rollouts needed
- [ ] Run `openenv validate` — all checks must pass
- [ ] Run `openenv validate --url <live-url>` — all criteria must pass
- [ ] Deployed to HF Spaces via `openenv push --repo-id username/env-name`
