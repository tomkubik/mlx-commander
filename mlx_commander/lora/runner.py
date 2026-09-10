"""
Sequential Execution Engine for MLX Commander LoRA Queue.
Executes queued fine-tuning runs one-by-one, streams real-time logs,
tracks progress in queue.json, and fires desktop notifications upon completion.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .queue import QueueManager
from .tracking import WandbTracker


def notify_macos(title: str, message: str) -> None:
    """Fire a native macOS desktop notification via AppleScript."""
    try:
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], capture_output=True, check=False)
    except Exception:
        pass


def execute_single_run(
    mgr: QueueManager,
    run_id: str,
    wandb_project: Optional[str] = None,
    enable_wandb: bool = True,
) -> int:
    """Execute a single LoRA run by ID, streaming output to console, log file, and Weights & Biases."""
    r = mgr.get_run(run_id)
    if not r:
        return 1

    cfg_file = mgr.configs_dir / f"{r.id}.yaml"
    log_file = mgr.logs_dir / f"{r.id}.log"

    # Initialize Weights & Biases tracking if enabled and authenticated
    tracker = WandbTracker(project=wandb_project or r.wandb_project, enabled=enable_wandb)
    tracker.start_run(r)

    mgr.update_run_status(run_id, "running")
    print(f"\n[MLX Commander] Starting run: {r.name}")
    print(f"  Model:      {r.model}")
    print(f"  Data:       {r.data}")
    print(f"  Config:     {cfg_file}")
    print(f"  Output Log: {log_file}")
    if tracker.run_url:
        print(f"  W&B Run:    {tracker.run_url}")
    print("-" * 60)

    cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", str(cfg_file)]

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"=== MLX Commander LoRA Run: {r.name} ===\n")
        lf.write(f"Command: {' '.join(cmd)}\n\n")
        lf.flush()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in iter(proc.stdout.readline, ""):
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
                lf.flush()
                tracker.log_line(line)
            proc.stdout.close()
            code = proc.wait()
        except FileNotFoundError:
            try:
                proc = subprocess.Popen(
                    ["mlx_lm.lora", "--config", str(cfg_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in iter(proc.stdout.readline, ""):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    lf.write(line)
                    lf.flush()
                    tracker.log_line(line)
                proc.stdout.close()
                code = proc.wait()
            except Exception as e:
                err_str = f"Failed to spawn mlx_lm.lora: {e}"
                sys.stderr.write(err_str + "\n")
                lf.write(err_str + "\n")
                code = 127
        except Exception as e:
            err_str = f"Execution error: {e}"
            sys.stderr.write(err_str + "\n")
            lf.write(err_str + "\n")
            code = 1

    wandb_url = tracker.finish_run(exit_code=code)
    if wandb_url:
        r.wandb_url = wandb_url
        mgr.save()
        print(f"  [W&B] Run tracked at: {wandb_url}")

    if code == 0:
        mgr.update_run_status(run_id, "completed", exit_code=0)
        print(f"\n[OK] Run '{r.name}' completed successfully.")
    else:
        mgr.update_run_status(run_id, "failed", exit_code=code, error_message=f"Exited with code {code}")
        print(f"\n[FAIL] Run '{r.name}' failed with code {code}.")

    return code


def run_lora_queue(
    queue_dir: Optional[Path] = None,
    stop_on_failure: bool = False,
    wandb_project: Optional[str] = None,
    enable_wandb: bool = True,
) -> int:
    """
    Run all queued runs sequentially.
    Safe for Apple Silicon: one model trains at a time, preventing memory thrashing.
    """
    mgr = QueueManager(queue_dir)
    pending = mgr.get_pending_runs()

    if not pending:
        print("[MLX Commander] No queued runs found in", mgr.queue_dir)
        return 0

    total_runs = len(pending)
    print(f"\n===================================================")
    print(f"   MLX Commander :: Starting {total_runs} Queued LoRA Run(s)")
    print(f"   Directory: {mgr.queue_dir}")
    print(f"===================================================")

    completed_count = 0
    failed_count = 0

    for idx, r in enumerate(pending, 1):
        print(f"\n>>> Processing Job [{idx}/{total_runs}]: {r.name}")
        code = execute_single_run(mgr, r.id, wandb_project=wandb_project, enable_wandb=enable_wandb)
        if code == 0:
            completed_count += 1
        else:
            failed_count += 1
            if stop_on_failure:
                print(f"Stopping queue early due to failure in '{r.name}'.")
                break

    print(f"\n===================================================")
    print(f"  Queue Execution Finished!")
    print(f"  Total: {total_runs} | Completed: {completed_count} | Failed: {failed_count}")
    print(f"===================================================")

    notify_macos(
        "MLX Commander",
        f"Finished {total_runs} fine-tuning run(s): {completed_count} succeeded, {failed_count} failed.",
    )
    return 0 if failed_count == 0 else 1
