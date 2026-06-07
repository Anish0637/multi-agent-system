"""
Skill loader — reads SKILL.md (and sub-skills) from disk.
Used to inject domain knowledge into specialist agent system prompts.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=32)
def load_skill(skill_path: str) -> str:
    """Load a SKILL.md (or sub-skill) file and return its content."""
    p = Path(skill_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Skill file not found: {p}")
    return p.read_text(encoding="utf-8")


def load_skill_with_subskills(skill_path: str, *subskill_paths: str) -> str:
    """
    Load a main SKILL.md and append any relevant sub-skills.
    Returns a single concatenated string for use as system prompt.
    """
    parts = [load_skill(skill_path)]
    for sub in subskill_paths:
        parts.append(f"\n\n---\n\n{load_skill(sub)}")
    return "\n".join(parts)
