---
title: Dalaal Env
emoji: "\U0001F310"
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# dalaal-env: Browser-Use RL Environment

An OpenEnv-compatible reinforcement learning environment where agents learn to interact with web pages through an **accessibility tree** interface.

The agent observes a structured text representation of the page (like a screen reader) and takes discrete actions — click, type, scroll, select — to complete browser tasks.

## Architecture

```
Agent (LLM) ←→ OpenEnv Client ←→ WebSocket ←→ OpenEnv Server ←→ Playwright (headless Chromium)
```

## Observation Space

The agent sees the page as a numbered accessibility tree:

```
[1] heading "My Todo List" level=1
[2] textbox "Add a new todo"
[3] button "Add"
[4] checkbox "Complete: Walk the dog" checked=false
[5] button "Delete: Walk the dog"
```

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `click` | `element_id` | Click an element by its tree ID |
| `type` | `element_id`, `text` | Clear and type text into an input |
| `select_option` | `element_id`, `text` | Select a dropdown option by label |
| `press_key` | `key` | Press a keyboard key (Enter, Tab, etc.) |
| `scroll` | `direction` | Scroll up or down |
| `go_back` | — | Browser back |
| `done` | — | Signal task completion |

## Available Tasks

| Task ID | Description | Max Steps |
|---------|-------------|-----------|
| `todo_add` | Add "Buy milk" to a todo list | 10 |
| `todo_add_and_complete` | Add "Buy milk" and mark complete | 15 |
| `login` | Log in with credentials | 10 |
| `search_and_click` | Search and click first result | 10 |
| `add_to_cart` | Add headphones to cart | 10 |
| `add_to_cart_and_checkout` | Add to cart and checkout | 15 |
| `fill_registration` | Fill a multi-field registration form | 15 |

## Reward

- **+1.0** on task success (verified by DOM-based success criteria)
- **-0.01** per step (encourages efficiency)
- **0.0** on failure or timeout
- Final score clamped to [0, 1]

## Quick Start

```bash
# Install
uv sync

# Install Playwright browser
uv run playwright install chromium

# Run server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run inference
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
uv run python inference.py
```

## Docker

```bash
cd server && docker build -t dalaal-env .
```

## Validation

```bash
uv run openenv validate
```
