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


# ── Email Inbox Tasks (MiniWoB++ inspired) ──────────────────────────

_register(Task(
    id="email_forward",
    description='Find the email from Bob Martinez about "Meeting Notes - Sprint Planning" and forward it to alice@company.com.',
    site_file="email_inbox/index.html",
    max_steps=15,
    success_check_js="""
        (() => {
            const msg = document.getElementById('forward-success');
            return msg && msg.style.display !== 'none' &&
                   msg.dataset.from === 'Bob Martinez' &&
                   msg.dataset.to === 'alice@company.com';
        })()
    """,
))

_register(Task(
    id="email_find_and_forward",
    description='Find the email that mentions an overdue invoice and forward it to finance@company.com.',
    site_file="email_inbox/index.html",
    max_steps=15,
    success_check_js="""
        (() => {
            const msg = document.getElementById('forward-success');
            return msg && msg.style.display !== 'none' &&
                   msg.dataset.from === 'Frank Wilson' &&
                   msg.dataset.to === 'finance@company.com';
        })()
    """,
))

# ── Flight Booking Tasks (MiniWoB++ / Mind2Web inspired) ────────────

_register(Task(
    id="book_cheapest_flight",
    description='Search for flights from San Francisco (SFO) to Tokyo (NRT) and book the cheapest available flight.',
    site_file="flight_booking/index.html",
    max_steps=15,
    success_check_js="""
        (() => {
            const panel = document.getElementById('confirm-panel');
            return panel && panel.style.display !== 'none' &&
                   panel.dataset.from === 'SFO' && panel.dataset.to === 'TYO' &&
                   parseInt(panel.dataset.price) === 489;
        })()
    """,
))

_register(Task(
    id="book_nonstop_flight",
    description='Search for flights from San Francisco (SFO) to Tokyo (NRT), filter to nonstop only, and book the cheapest nonstop flight.',
    site_file="flight_booking/index.html",
    max_steps=20,
    success_check_js="""
        (() => {
            const panel = document.getElementById('confirm-panel');
            return panel && panel.style.display !== 'none' &&
                   panel.dataset.from === 'SFO' && panel.dataset.to === 'TYO' &&
                   parseInt(panel.dataset.price) === 892;
        })()
    """,
))

# ── Date Picker Tasks (MiniWoB++ inspired) ──────────────────────────

_register(Task(
    id="schedule_event",
    description='Schedule an event called "Team Offsite" on December 25, 2025 using the calendar date picker.',
    site_file="date_picker/index.html",
    max_steps=15,
    success_check_js="""
        (() => {
            const msg = document.getElementById('success-message');
            return msg && msg.style.display !== 'none' &&
                   msg.dataset.eventName === 'Team Offsite' &&
                   msg.dataset.date === '2025-12-25';
        })()
    """,
))

# ── Data Table Tasks (WebArena inspired) ─────────────────────────────

_register(Task(
    id="delete_inactive_employees",
    description='In the employee directory, filter by status "Inactive" and delete all inactive employees.',
    site_file="data_table/index.html",
    max_steps=15,
    success_check_js="""
        (() => {
            const msg = document.getElementById('action-result');
            if (!msg || msg.style.display === 'none') return false;
            const names = JSON.parse(msg.dataset.deletedNames || '[]');
            return names.includes('Henry Brown');
        })()
    """,
))

_register(Task(
    id="find_highest_salary",
    description='Sort the employee directory by salary (highest first) and delete the highest-paid employee in the Engineering department.',
    site_file="data_table/index.html",
    max_steps=20,
    success_check_js="""
        (() => {
            const msg = document.getElementById('action-result');
            if (!msg || msg.style.display === 'none') return false;
            const names = JSON.parse(msg.dataset.deletedNames || '[]');
            return names.includes('Carol White');
        })()
    """,
))

# ── Multi-Step Wizard Tasks (Mind2Web DMV inspired) ──────────────────

_register(Task(
    id="renew_registration_2yr",
    description='Renew vehicle registration for plate "ABC1234". Confirm the existing address is correct. Select the 2-year renewal option and submit.',
    site_file="multi_step_wizard/index.html",
    max_steps=20,
    success_check_js="""
        (() => {
            const msg = document.getElementById('success-message');
            return msg && msg.style.display !== 'none' &&
                   msg.dataset.plate === 'ABC1234' &&
                   msg.dataset.years === '2' &&
                   msg.dataset.addressConfirmed === 'true';
        })()
    """,
))

# ── Recipe Site Tasks (WebVoyager Allrecipes inspired) ───────────────

_register(Task(
    id="find_vegan_recipe",
    description='Search for "vegan" recipes, open the highest-rated vegan recipe, and save it to favorites.',
    site_file="recipe_site/index.html",
    max_steps=15,
    success_check_js="""
        (() => {
            const msg = document.getElementById('save-success');
            return msg && msg.style.display !== 'none' &&
                   msg.dataset.recipeName === 'Vegan Buddha Bowl';
        })()
    """,
))

_register(Task(
    id="find_quick_vegetarian",
    description='Filter recipes to "Vegetarian" diet and sort by quickest cooking time. Open the fastest vegetarian recipe and save it to favorites.',
    site_file="recipe_site/index.html",
    max_steps=15,
    success_check_js="""
        (() => {
            const msg = document.getElementById('save-success');
            return msg && msg.style.display !== 'none' &&
                   msg.dataset.recipeName === 'Mushroom Risotto';
        })()
    """,
))

# ── Issue Tracker Tasks (WebArena GitLab inspired) ───────────────────

_register(Task(
    id="label_timeout_issues",
    description='Filter issues to show only those assigned to "maria". Find any open issue that mentions "timeout" in its description and add the "urgent" label to it.',
    site_file="issue_tracker/index.html",
    max_steps=25,
    success_check_js="""
        (() => {
            // Issues assigned to maria that mention "timeout": #101, #107
            // Check if at least one has the urgent label
            const issue101 = document.querySelector('[data-id="101"]') ||
                             (() => { const msg = document.getElementById('label-success');
                                      return msg && msg.dataset.issueId === '101' && msg.dataset.label === 'urgent'; })();
            // Check via the issues array directly
            const script = document.querySelector('script:last-of-type');
            const i101 = typeof issues !== 'undefined' && issues.find(i => i.id === 101);
            const i107 = typeof issues !== 'undefined' && issues.find(i => i.id === 107);
            return (i101 && i101.labels.includes('urgent')) || (i107 && i107.labels.includes('urgent'));
        })()
    """,
))

_register(Task(
    id="label_all_timeout_issues",
    description='Find ALL open issues assigned to "maria" that mention "timeout" in their description, and add the "urgent" label to each one. There may be multiple such issues.',
    site_file="issue_tracker/index.html",
    max_steps=35,
    success_check_js="""
        (() => {
            // maria's open issues mentioning "timeout": #101 (gateway timeout), #107 (timeout-like delay)
            const i101 = typeof issues !== 'undefined' && issues.find(i => i.id === 101);
            const i107 = typeof issues !== 'undefined' && issues.find(i => i.id === 107);
            return i101 && i101.labels.includes('urgent') && i107 && i107.labels.includes('urgent');
        })()
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
