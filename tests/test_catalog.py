"""skills/catalog.json matches the code and the files it names."""
import ast
import json
import re

from mobile_panda_sim.paths import REPO

SKILLS = json.loads((REPO / "skills/catalog.json").read_text(encoding="utf-8"))["local_skills"]
REQUIRED = {"id", "name", "kind", "call", "code", "demo", "docs", "preconditions", "applies_to", "not_for", "validated"}


def methods(path):
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    return {node.name for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef) for node in cls.body if isinstance(node, ast.FunctionDef)}


def test_every_entry_is_complete():
    assert len({s["id"] for s in SKILLS}) == len(SKILLS)
    for skill in SKILLS:
        assert REQUIRED <= set(skill), skill["id"]
        assert skill["kind"] in ("classical", "rl")
        assert skill["validated"]["results"], skill["id"]


def test_named_files_exist():
    for skill in SKILLS:
        for key in ("code", "demo", "docs", "checkpoint", "training"):
            if key in skill:
                assert (REPO / skill[key]).is_file(), (skill["id"], key)


def test_calls_name_real_methods():
    for skill in SKILLS:
        method = re.findall(r"\.(\w+)\(", skill["call"])[-1]
        assert method in methods(skill["code"]), skill["id"]


def test_public_skills_are_listed():
    listed = " ".join(s["call"] for s in SKILLS)
    for name in ("pick", "place", "open", "close", "home", "suggest_base_pose", "pick_move"):
        assert f".{name}(" in listed, name
