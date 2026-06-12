#!/usr/bin/env python3
"""Unified SLURM job submission via YAML configs.

Usage:
    python sbatch_runner.py submit configs/training/my_job.yaml --dry-run
    python sbatch_runner.py submit configs/training/my_job.yaml --delete

Each experiment YAML has:
    slurm_profile: <key>       # selects slurm_profiles/<key>.yaml
    script:                    # path to Python script (relative to repo)
    commands:                  # OR a list of bash cmds (use ${SELF}, ${REPO})
    env:                       # conda/micromamba settings
    args:                      # command-line arguments (only with 'script')

By default, all job files (scripts, logs, errors) are kept in logs/sbatch_jobs/
to make them easy to find and manage. Use --delete to remove files after job
completion.
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

import yaml


# ── Constants ──────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(REPO_ROOT, "slurm_profiles")
SBATCH_LOGS_DIR = os.path.join(REPO_ROOT, "logs", "sbatch_jobs")


# ── Loading ────────────────────────────────────────────────────────────

def load_experiment(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not cfg or not isinstance(cfg, dict):
        sys.exit(f"ERROR: {path} is empty or not a YAML mapping.")
    return cfg


def merge_slurm_profile(cfg):
    profile_name = cfg.get("slurm_profile")
    if not profile_name:
        sys.exit("ERROR: YAML must contain 'slurm_profile'.")

    profile_path = os.path.join(PROFILES_DIR, f"{profile_name}.yaml")
    if not os.path.isfile(profile_path):
        available = os.listdir(PROFILES_DIR) if os.path.isdir(PROFILES_DIR) else []
        sys.exit(f"ERROR: SLURM profile not found: {profile_path}\n"
                 f"       Available profiles: {', '.join(available)}")

    with open(profile_path) as f:
        profile = yaml.safe_load(f) or {}

    merged = dict(profile)
    merged.update(cfg.get("slurm", {}))
    cfg_merged = dict(cfg)
    cfg_merged["slurm"] = merged
    return cfg_merged


# ── Env helper ─────────────────────────────────────────────────────────

def _env_lines(env_cfg):
    env_type = env_cfg.get("type", "conda")
    env_path = env_cfg.get("path", "")
    env_name = env_cfg.get("name", "")

    lines = []
    if env_type == "conda" and env_name:
        lines.append("source ~/.bashrc")
        lines.append(f"conda activate {env_name}")
    elif env_type == "micromamba" and env_path:
        lines.append("source ~/.bashrc")
        lines.append(f"micromamba activate {env_path}")
    elif env_type == "micromamba" and env_name and not env_path:
        lines.append("source ~/.bashrc")
        lines.append(f"micromamba activate {env_name}")
    else:
        lines.append("source ~/.bashrc")
    return lines


# ── Variable substitution ──────────────────────────────────────────────

_VARS = {"${SELF}": None, "${REPO}": REPO_ROOT}

def _sub_vars(text):
    """Replace ${REPO} and ${SELF} variable references in a string."""
    for ph, val in _VARS.items():
        if ph in text:
            text = text.replace(ph, val)
    return text


def _render_args(script_path, args, resolved_cfg_path):
    """Convert yaml args dict to 'python script --flag value' string variables.

    Arg keys are used verbatim as ``--<key>`` (NO '_'->'-' conversion): the target
    scripts use argparse with underscore option strings (e.g. ``--experiment_name``)
    and reject the hyphenated form. If a target script genuinely uses a hyphenated
    option, write that arg key with a leading '-' (e.g. ``-foo-bar: value``).
    """
    _VARS["${SELF}"] = resolved_cfg_path
    parts = [f"python {script_path}"]
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
        elif v is None or v is False:
            continue
        elif isinstance(v, list):
            parts.append(f"--{k}")
            for item in v:
                s = str(item)
                parts.append("'" + s + "'" if any(c in s for c in " ,(") else str(item))
        else:
            flag = f"--{k}" if not k.startswith("-") else k
            val = _sub_vars(str(v))
            parts.append(f"{flag} {val}")
    return " ".join(parts)


# ── Script generation ──────────────────────────────────────────────────

def generate_script(cfg, dry_run, cfg_path=""):
    slurm = cfg.get("slurm", {})
    env_cfg = cfg.get("env", {})

    partition = slurm.get("partition", "")
    time_limit = slurm.get("time", "06:00:00")
    job_name = slurm.get("job_name", "job")
    cpus = slurm.get("cpus", 4)
    mem = slurm.get("mem", "32G")

    gpu_spec = slurm.get("gpu")
    constraint = slurm.get("constraint", "")
    qos = slurm.get("qos", "")
    nodes = slurm.get("nodes", 1)
    exclude = slurm.get("exclude", "")
    account = slurm.get("account", "")

    # Determine a sensible job name
    if cfg.get("script"):
        script_basename = os.path.splitext(os.path.split(cfg["script"])[1])[0]
        job_name = slurm.get("job_name", script_basename)
    elif cfg.get("commands"):
        job_name = slurm.get("job_name", "pipeline")

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH -c {cpus}",
        f"#SBATCH --mem {mem}",
        f"#SBATCH -t {time_limit}",
        f"#SBATCH --nodes {nodes}",
    ]
    if partition:
        lines.append(f"#SBATCH -p {partition}")
    if gpu_spec:
        lines.append(f"#SBATCH -G {gpu_spec}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    if constraint:
        lines.append(f"#SBATCH --constraint {constraint}")
    if qos:
        lines.append(f"#SBATCH --qos {qos}")
    if exclude:
        lines.append(f"#SBATCH --exclude {exclude}")

    # ── Environment ────────────────────────────────────────────────
    full_script = None
    if cfg.get("script"):
        fp = os.path.join(REPO_ROOT, cfg["script"].strip())
        if not os.path.isfile(fp):
            sys.exit(f"ERROR: Script not found: {fp}")
        full_script = fp
    lines.extend(_env_lines(env_cfg))

    # ── Commands (commands: list) ──────────────────────────────────
    cmds = cfg.get("commands")
    if cmds:
        if not isinstance(cmds, list):
            sys.exit("ERROR: 'commands' must be a list of strings.")
        resolved = os.path.abspath(cfg_path) if cfg_path else cfg_path
        vmap = {"${SELF}": resolved, "${CONFIG}": resolved, "${REPO}": REPO_ROOT}
        rendered = []
        for cmd in cmds:
            for ph, val in vmap.items():
                cmd = cmd.replace(ph, val)
            rendered.append(f"set -euo pipefail && {cmd}")
        lines.append(" && \\\n".join(rendered))

    # ── Single script + args ───────────────────────────────────────
    elif cfg.get("script"):
        cmd = _render_args(full_script, cfg.get("args", {}), os.path.abspath(cfg_path) if cfg_path else None)
        if gpu_spec:
            cmd = f"CUDA_VISIBLE_DEVICES=0 {cmd}"
        lines.append(cmd)

    else:
        sys.exit("ERROR: YAML must contain 'script'+ 'args' or 'commands'.")

    lines.extend(["", "echo 'Done.'", ""])
    script_text = "\n".join(lines)

    if dry_run:
        print("=" * 60)
        print("DRY RUN — would generate and submit:")
        print("=" * 60)
        print(script_text)
        print("=" * 60)
        return None

    # ── Submit ─────────────────────────────────────────────────────
    # Use a stable directory for all job files
    script_name = os.path.splitext(os.path.split(cfg_path)[1])[0] if cfg_path else "job"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sh = os.path.join(SBATCH_LOGS_DIR, f"{script_name}_{ts}.sh")
    log = os.path.join(SBATCH_LOGS_DIR, f"{script_name}_{ts}.log")
    err = os.path.join(SBATCH_LOGS_DIR, f"{script_name}_{ts}.err")

    os.makedirs(SBATCH_LOGS_DIR, exist_ok=True)
    with open(sh, "w") as f:
        f.write(script_text)
    os.chmod(sh, 0o755)

    result = subprocess.run(
        ["sbatch", "-o", log, "-e", err, sh],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"ERROR: sbatch failed (exit {result.returncode})")
        print(result.stderr.strip())
        # Delete files if requested (reverse of default behavior)
        if cfg.get("keep_tmp"):
            print(f"Keeping job files: {SBATCH_LOGS_DIR}")
        else:
            # Clean up files if delete flag was set
            try:
                os.remove(sh)
                os.remove(log) 
                os.remove(err)
            except OSError:
                pass
        sys.exit(1)

    # Delete files if requested (reverse of default behavior)
    if cfg.get("keep_tmp"):
        print(f"Keeping job files: {SBATCH_LOGS_DIR}")
    else:
        # Clean up files if delete flag was set
        try:
            os.remove(sh)
            os.remove(log) 
            os.remove(err)
        except OSError:
            pass

    return sh, log, err


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs from YAML configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles are in slurm_profiles/ — choose via slurm_profile key.
Commands can use ${SELF} (path to this config) and ${REPO} (repo root).
Examples:
  python sbatch_runner.py configs/training/mito_volem.yaml
  python sbatch_runner.py configs/evaluation/cristae/eval_0601.yaml --dry-run
  python sbatch_runner.py configs/training/mito_volem.yaml --delete  # delete files after completion
""",
    )
    
    # Check if we're dealing with legacy "submit" command
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        # Legacy mode: use subparser
        sub = parser.add_subparsers(dest="command")
        sub_req = sub.add_parser("submit", help="Submit a job from a YAML config")
        sub_req.add_argument("config", help="Path to experiment YAML config")
        sub_req.add_argument("--dry-run", action="store_true",
                             help="Print the generated script without submitting")
        sub_req.add_argument("--delete", action="store_true",
                             help="Delete temporary sbatch files after job completion (default is to keep them)")
        
        args = parser.parse_args()
        if args.command == "submit":
            cfg = load_experiment(args.config)
            cfg_merged = merge_slurm_profile(cfg)
            if getattr(args, "delete"):
                cfg_merged["keep_tmp"] = False
            else:
                cfg_merged["keep_tmp"] = True
            generate_script(cfg_merged, dry_run=getattr(args, "dry_run"), cfg_path=args.config)
        else:
            parser.print_help()
    else:
        # New mode: direct config argument
        parser.add_argument("config", help="Path to experiment YAML config")
        parser.add_argument("--dry-run", action="store_true",
                            help="Print the generated script without submitting")
        parser.add_argument("--delete", action="store_true",
                            help="Delete temporary sbatch files after job completion (default is to keep them)")
        
        args = parser.parse_args()
        cfg = load_experiment(args.config)
        cfg_merged = merge_slurm_profile(cfg)
        if getattr(args, "delete"):
            cfg_merged["keep_tmp"] = False
        else:
            cfg_merged["keep_tmp"] = True
        generate_script(cfg_merged, dry_run=getattr(args, "dry_run"), cfg_path=args.config)


if __name__ == "__main__":
    main()
