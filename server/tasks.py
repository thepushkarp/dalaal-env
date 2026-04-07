"""
Task registry for the Dalaal Browser-Use Environment.

Each task defines a goal, a mock site to load, and JavaScript-based
success criteria that are evaluated in the browser context.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Task:
    """A browser task with success criteria."""

    id: str
    description: str
    site_file: str  # relative path from mock_sites/ to HTML file
    max_steps: int
    success_check_js: str  # JS expression returning true/false


# Resolve the mock_sites directory relative to this file
_MOCK_SITES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mock_sites",
)


def get_mock_sites_dir() -> str:
    return _MOCK_SITES_DIR


TASKS: dict[str, Task] = {}


def _register(task: Task):
    TASKS[task.id] = task


# ── Todo App Tasks ──────────────────────────────────────────────────

_register(Task(
    id="todo_add",
    description='Add a new todo item called "Buy milk" to the todo list.',
    site_file="todo_app/index.html",
    max_steps=10,
    success_check_js="""
        (() => {
            const items = document.querySelectorAll('.todo-text');
            return Array.from(items).some(el => el.textContent.trim().toLowerCase() === 'buy milk');
        })()
    """,
))

_register(Task(
    id="todo_add_and_complete",
    description='Add a todo item called "Buy milk" and mark it as completed.',
    site_file="todo_app/index.html",
    max_steps=15,
    success_check_js="""
        (() => {
            const items = document.querySelectorAll('.todo-item.completed');
            return Array.from(items).some(el => el.querySelector('.todo-text')?.textContent.trim().toLowerCase() === 'buy milk');
        })()
    """,
))

# ── Login Form Tasks ────────────────────────────────────────────────

_register(Task(
    id="login",
    description='Log in with username "admin" and password "secret123".',
    site_file="login_form/index.html",
    max_steps=10,
    success_check_js="""
        document.getElementById('success-message') !== null &&
        document.getElementById('success-message').style.display !== 'none'
    """,
))

# ── Search Engine Tasks ─────────────────────────────────────────────

_register(Task(
    id="search_and_click",
    description='Search for "machine learning" and click on the first result link.',
    site_file="search_engine/index.html",
    max_steps=10,
    success_check_js="""
        document.getElementById('result-page') !== null &&
        document.getElementById('result-page').style.display !== 'none'
    """,
))

# ── E-commerce Tasks ────────────────────────────────────────────────

_register(Task(
    id="add_to_cart",
    description='Add the "Wireless Headphones" product to your shopping cart.',
    site_file="ecommerce/index.html",
    max_steps=10,
    success_check_js="""
        (() => {
            const cartCount = document.getElementById('cart-count');
            return cartCount && parseInt(cartCount.textContent) > 0;
        })()
    """,
))

_register(Task(
    id="add_to_cart_and_checkout",
    description='Add the "Wireless Headphones" to your cart and proceed to checkout.',
    site_file="ecommerce/index.html",
    max_steps=15,
    success_check_js="""
        document.getElementById('checkout-page') !== null &&
        document.getElementById('checkout-page').style.display !== 'none'
    """,
))

# ── Registration Form Tasks ─────────────────────────────────────────

_register(Task(
    id="fill_registration",
    description='Fill the registration form with: Name "John Doe", Email "john@example.com", select country "United States", and submit.',
    site_file="registration_form/index.html",
    max_steps=15,
    success_check_js="""
        document.getElementById('success-message') !== null &&
        document.getElementById('success-message').style.display !== 'none'
    """,
))


def get_task(task_id: str) -> Task:
    """Get a task by ID. Raises KeyError if not found."""
    if task_id not in TASKS:
        available = ", ".join(sorted(TASKS.keys()))
        raise KeyError(f"Unknown task '{task_id}'. Available tasks: {available}")
    return TASKS[task_id]


def list_tasks() -> list[str]:
    """Return all available task IDs."""
    return sorted(TASKS.keys())
