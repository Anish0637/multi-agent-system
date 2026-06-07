"""
CLI entry point — run the multi-agent system interactively.

Usage:
    python main.py
    python main.py --message "ingest lyft_policy.txt for tenant acme"
"""
from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from orchestrator.router import route
from executor.runner import execute

console = Console()


def run_once(message: str, tenant_id: str = "default", user_id: str = "anonymous") -> dict:
    enriched = f"[tenant_id={tenant_id}, user_id={user_id}] {message}"

    with console.status("[bold cyan]Orchestrating…"):
        orch = route(enriched)

    agent_resp = orch.agent_response
    if agent_resp.error:
        console.print(f"[red]Specialist error: {agent_resp.error}")
        return {"error": agent_resp.error}

    action = agent_resp.action
    console.print(Panel(
        f"[bold]Routed to:[/bold] [cyan]{orch.routed_to}[/cyan]\n"
        f"[bold]Action:[/bold]    [yellow]{action.method} {action.path}[/yellow]"
        if action and not action.direct_answer else
        f"[bold]Routed to:[/bold] [cyan]{orch.routed_to}[/cyan]\n"
        f"[bold]Direct answer[/bold]",
        title="Orchestrator Decision",
    ))

    if action.direct_answer:
        console.print(Panel(action.direct_answer, title="Answer"))
        return {"answer": action.direct_answer}

    with console.status("[bold green]Executing action…"):
        result = execute(action, service_name=orch.routed_to)

    status_color = "green" if result.success else "red"
    console.print(Panel(
        json.dumps(result.body, indent=2, default=str),
        title=f"[{status_color}]Result (HTTP {result.status_code})",
    ))

    if result.error:
        console.print(f"[red]Error: {result.error}")

    return {"routed_to": orch.routed_to, "result": result.body, "success": result.success}


def interactive_loop():
    console.print(Panel(
        "[bold cyan]Multi-Agent System[/bold cyan]\n"
        "Type your request in plain English. Type [bold]exit[/bold] to quit.\n\n"
        "[dim]Examples:[/dim]\n"
        '  • "ingest lyft_policy.txt for tenant acme"\n'
        '  • "what are vehicle requirements for Lyft drivers?"\n'
        '  • "erase all data for user alice in tenant acme"\n'
        '  • "add policy: max response 150 words for tenant default"',
        title="Welcome",
    ))

    while True:
        try:
            msg = console.input("\n[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not msg or msg.lower() in ("exit", "quit"):
            break

        run_once(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent System CLI")
    parser.add_argument("--message",   "-m", help="Single message to process")
    parser.add_argument("--tenant-id", "-t", default="default")
    parser.add_argument("--user-id",   "-u", default="anonymous")
    args = parser.parse_args()

    if args.message:
        result = run_once(args.message, args.tenant_id, args.user_id)
        sys.exit(0 if result.get("success", True) else 1)
    else:
        interactive_loop()
