"""
Dalaal Browser-Use Environment Implementation.

An RL environment where agents interact with web pages through
an accessibility-tree interface, learning to perform browser tasks
like form filling, navigation, and multi-step workflows.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

try:
    from ..models import DalaalEnvAction, DalaalEnvObservation
except (ImportError, SystemError):
    try:
        from models import DalaalEnvAction, DalaalEnvObservation
    except ImportError:
        from dalaal_env.models import DalaalEnvAction, DalaalEnvObservation

from .accessibility import AccessibilityTree
from .tasks import Task, get_task, list_tasks, get_mock_sites_dir

# Step penalty to encourage efficiency
STEP_PENALTY = -0.01


class DalaalEnvEnvironment(Environment):
    """
    Browser-Use RL environment powered by Playwright.

    The agent observes the page through an accessibility tree and takes
    discrete actions (click, type, scroll, etc.) to complete tasks.

    Each episode presents a task on a bundled mock website. The agent
    receives +1.0 reward for task success (minus step penalties) and
    0.0 for failure/timeout.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._a11y = AccessibilityTree()
        self._task: Optional[Task] = None
        self._last_error: Optional[str] = None
        self._total_reward: float = 0.0
        self._mock_sites_dir = get_mock_sites_dir()

    async def _ensure_browser(self):
        """Launch browser if not already running."""
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )

    async def _new_page(self) -> Page:
        """Create a fresh browser context and page."""
        if self._context:
            await self._context.close()
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720},
        )
        self._page = await self._context.new_page()
        return self._page

    async def _build_observation(self, done: bool = False, reward: float = 0.0) -> DalaalEnvObservation:
        """Build observation from current page state."""
        tree_text = await self._a11y.extract(self._page)
        url = self._page.url if self._page else ""
        title = await self._page.title() if self._page else ""

        return DalaalEnvObservation(
            url=url,
            title=title,
            accessibility_tree=tree_text,
            task_description=self._task.description if self._task else "",
            last_action_error=self._last_error,
            step_count=self._state.step_count,
            max_steps=self._task.max_steps if self._task else 20,
            done=done,
            reward=reward,
        )

    async def reset_async(self, seed=None, episode_id=None, **kwargs) -> DalaalEnvObservation:
        """Reset environment with a new task.

        Args:
            seed: Random seed (unused currently)
            episode_id: Custom episode ID
            **kwargs: Must include 'task' with a valid task ID.
                      Defaults to 'todo_add' if not specified.
        """
        task_id = kwargs.get("task", "todo_add")
        self._task = get_task(task_id)
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._last_error = None
        self._total_reward = 0.0

        await self._ensure_browser()
        page = await self._new_page()

        # Load the mock site
        site_path = os.path.join(self._mock_sites_dir, self._task.site_file)
        await page.goto(f"file://{site_path}")
        await page.wait_for_load_state("domcontentloaded")

        return await self._build_observation()

    def reset(self, seed=None, episode_id=None, **kwargs) -> DalaalEnvObservation:
        """Sync reset - delegates to async version."""
        return asyncio.get_event_loop().run_until_complete(
            self.reset_async(seed=seed, episode_id=episode_id, **kwargs)
        )

    async def step_async(self, action: DalaalEnvAction, timeout_s=30, **kwargs) -> DalaalEnvObservation:
        """Execute a browser action and return the new observation."""
        self._state.step_count += 1
        self._last_error = None

        try:
            await self._execute_action(action)
        except Exception as e:
            self._last_error = str(e)

        # Small wait for page to settle after action
        await asyncio.sleep(0.3)

        # Check if agent signaled done
        if action.action_type == "done":
            success = await self._check_success()
            reward = (1.0 + self._total_reward) if success else 0.0
            reward = max(0.0, min(1.0, reward))
            return await self._build_observation(done=True, reward=reward)

        # Check step limit
        if self._state.step_count >= self._task.max_steps:
            success = await self._check_success()
            reward = (1.0 + self._total_reward) if success else 0.0
            reward = max(0.0, min(1.0, reward))
            return await self._build_observation(done=True, reward=reward)

        # Normal step: small penalty
        self._total_reward += STEP_PENALTY
        return await self._build_observation(done=False, reward=STEP_PENALTY)

    def step(self, action: DalaalEnvAction, timeout_s=30, **kwargs) -> DalaalEnvObservation:
        """Sync step - delegates to async version."""
        return asyncio.get_event_loop().run_until_complete(
            self.step_async(action, timeout_s=timeout_s, **kwargs)
        )

    async def _execute_action(self, action: DalaalEnvAction):
        """Execute the given action on the browser page."""
        page = self._page
        if page is None:
            raise RuntimeError("No page open. Call reset() first.")

        if action.action_type == "click":
            await self._action_click(page, action)
        elif action.action_type == "type":
            await self._action_type(page, action)
        elif action.action_type == "select_option":
            await self._action_select(page, action)
        elif action.action_type == "press_key":
            await self._action_press_key(page, action)
        elif action.action_type == "scroll":
            await self._action_scroll(page, action)
        elif action.action_type == "go_back":
            await page.go_back()
        elif action.action_type == "done":
            pass  # handled in step_async

    async def _action_click(self, page: Page, action: DalaalEnvAction):
        if action.element_id is None:
            raise ValueError("click requires element_id")
        node = self._a11y.get_node(action.element_id)
        if node is None:
            raise ValueError(f"No element with id={action.element_id}")
        locator = self._get_locator(page, node)
        await locator.click(timeout=5000)

    async def _action_type(self, page: Page, action: DalaalEnvAction):
        if action.element_id is None:
            raise ValueError("type requires element_id")
        if action.text is None:
            raise ValueError("type requires text")
        node = self._a11y.get_node(action.element_id)
        if node is None:
            raise ValueError(f"No element with id={action.element_id}")
        locator = self._get_locator(page, node)
        await locator.click(timeout=5000)
        await locator.fill(action.text, timeout=5000)

    async def _action_select(self, page: Page, action: DalaalEnvAction):
        if action.element_id is None:
            raise ValueError("select_option requires element_id")
        if action.text is None:
            raise ValueError("select_option requires text")
        node = self._a11y.get_node(action.element_id)
        if node is None:
            raise ValueError(f"No element with id={action.element_id}")
        locator = self._get_locator(page, node)
        await locator.select_option(label=action.text, timeout=5000)

    async def _action_press_key(self, page: Page, action: DalaalEnvAction):
        key = action.key or "Enter"
        await page.keyboard.press(key)

    async def _action_scroll(self, page: Page, action: DalaalEnvAction):
        direction = action.direction or "down"
        delta = -300 if direction == "up" else 300
        await page.mouse.wheel(0, delta)

    def _get_locator(self, page: Page, node):
        """Get a Playwright locator for an accessibility tree node."""
        role = node.role
        name = node.name

        # Map accessibility roles to Playwright's get_by_role
        role_map = {
            "button": "button",
            "link": "link",
            "textbox": "textbox",
            "checkbox": "checkbox",
            "radio": "radio",
            "combobox": "combobox",
            "heading": "heading",
            "listitem": "listitem",
            "option": "option",
            "tab": "tab",
            "menuitem": "menuitem",
            "searchbox": "searchbox",
        }

        playwright_role = role_map.get(role)
        if playwright_role and name:
            return page.get_by_role(playwright_role, name=name)
        elif playwright_role:
            return page.get_by_role(playwright_role)
        elif name:
            return page.get_by_text(name, exact=True)
        else:
            raise ValueError(f"Cannot locate element: role={role}, name={name}")

    async def _check_success(self) -> bool:
        """Check if the current task's success criteria are met."""
        if self._task is None or self._page is None:
            return False
        try:
            result = await self._page.evaluate(self._task.success_check_js)
            return bool(result)
        except Exception:
            return False

    async def close(self):
        """Clean up browser resources."""
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    @property
    def state(self) -> State:
        return self._state
