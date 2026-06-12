#!/usr/bin/env python
"""Validate that every YAML config in configs/ points to a working script.

For each YAML config:
  1. Extract the target script path from 'script:' or 'commands:' fields
  2. Resolve it relative to repo root
  3. Run the script with --help and check exit code 0

Additionally validates ALL inference scripts (not just YAML-referenced ones)
since inference scripts may be used independently.

Run from repo root:   python configs/test_config_validation.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

import yaml


# ─────────────────────────────────────────────────────────────
# YAML parsing helpers
# ─────────────────────────────────────────────────────────────


def _extract_script_paths(yaml_path: Path) -> List[str]:
    """Extract Python .py paths from a YAML config.

    Handles two formats:
      script: 'inference/mitochondria/segment_mitochondria_ooc.py'
      commands:
        - python ${REPO}/inference/cristae/segment_cristae.py -c ${SELF}
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    scripts: List[str] = []

    # 1) 'script:' field — single script path
    if isinstance(cfg.get("script"), str):
        raw = cfg["script"]
        raw = raw.replace("${REPO}", str(_repo_root)).replace("${SELF}", str(yaml_path))
        scripts.append(raw)

    # 2) 'commands:' field — list of shell commands
    #    e.g. 'python ${REPO}/inference/cristae/segment_cristae.py -c ${SELF}'
    if isinstance(cfg.get("commands"), list):
        cmd_re = re.compile(r"(python|python3)\s+(\$\{REPO\})?(/?[\w/.]+\.py)")
        for cmd in cfg["commands"]:
            if not isinstance(cmd, str):
                continue
            cmd = cmd.replace("${REPO}", str(_repo_root)).replace("${SELF}", str(yaml_path))
            m = cmd_re.search(cmd)
            if m:
                scripts.append(m.group(3).lstrip("/"))

    # strip duplicates while preserving order
    seen = set()
    return [s for s in scripts if not (s in seen or seen.add(s))]


# ─────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────


def _validate_script(path: Path) -> Tuple[bool, str]:
    """Run <path> --help and return (ok, message)."""
    try:
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=_repo_root,
        )
        if result.returncode == 0 and "--help" in result.stdout:
            return True, "OK"
        else:
            return False, f"exit={result.returncode}"
    except subprocess.TimeoutExpired:
        return False, "timeout (30s)"
    except FileNotFoundError:
        return False, "file not found"
    except Exception as exc:
        return False, str(exc)


# ─────────────────────────────────────────────────────────────
# All inference scripts (including non-YAML-referenced ones)
# ─────────────────────────────────────────────────────────────


def _all_inference_scripts() -> List[Path]:
    """Return every Python inference script, regardless of YAML reference."""
    inf_dir = Path(_repo_root) / "inference"
    if not inf_dir.is_dir():
        return []
    return sorted(inf_dir.rglob("*.py"))


# ─────────────────────────────────────────────────────────────
# Test case
# ─────────────────────────────────────────────────────────────
def test_all_yaml_configs():
    """Discover all YAML configs, extract scripts, validate with --help."""

    configs_dir = Path(_repo_root) / "configs"

    # Phase 1: YAML-derived scripts
    yaml_configs = sorted(configs_dir.rglob("*.yaml")) if configs_dir.is_dir() else []

    results: List[Tuple[str, str, str, bool]] = []  # (yaml, script, status, ok)

    for yc in yaml_configs:
        scripts = _extract_script_paths(yc)
        rel_yaml = yc.relative_to(_repo_root)
        for s in scripts:
            script_path = Path(s) if s.startswith("/") else Path(_repo_root) / s
            # Always use relative path for clean display
            try:
                rel_script = str(script_path.resolve().relative_to(_repo_root))
            except ValueError:
                rel_script = str(script_path)
            ok, msg = _validate_script(script_path)
            results.append((str(rel_yaml), rel_script, msg, ok))

    # Phase 2: ALL inference scripts (not just YAML-referenced)
    all_inf = _all_inference_scripts()
    seen = {(r[0], r[1]) for r in results}   # (yaml, script) pairs
    for inf_py in all_inf:
        if inf_py.name.startswith("test_"):
            continue
        key = ("(inference)", str(inf_py.relative_to(_repo_root)))
        if key in seen:
            continue  # already covered by a YAML config
        seen.add(key)
        ok, msg = _validate_script(inf_py)
        results.append(("  (inference)", str(inf_py.relative_to(_repo_root)), msg, ok))

    # Phase 3: print results
    print("=" * 80)
    print("  Config to Script --help  validation  (27 configs + all inference)")
    print("=" * 80)
    print()

    max_yaml = 50
    max_script = 70
    for yaml_name, script_name, msg, ok in results:
        yn = yaml_name[:max_yaml].ljust(max_yaml)
        sn = script_name[:max_script].ljust(max_script)
        mark = "OK" if ok else "FAIL"
        status = f"{mark:4s}  {msg}"
        print(f"  {yn}  ->  {sn}  {status}")

    total = len(results)
    passed = sum(1 for _, _, _, ok in results if ok)

    print()
    print(f"  {passed}/{total} scripts OK")

    if any(not ok for _, _, _, ok in results):
        print("  FAILURES:")
        for yaml_name, script_name, msg, ok in results:
            if not ok:
                print(f"    {yaml_name} -> {script_name}")
        sys.exit(1)
    print("  All scripts passed --help")


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_all_yaml_configs()
