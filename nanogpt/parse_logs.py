"""
parse_logs.py

Parse NanoGPT training logs and extract the validation loss at each evaluated step.

The logs embed the full source of train_gpt_simple.py at the top, which contains
f-string templates like ``val_loss:{val_loss:.5f}``. The regex below only matches
*numeric* values, so those template lines are ignored.

Validation lines have the form:
    step:125/3350 val_loss:4.67375 train_time:1107.473s time_since_last_val:... allocated:...
"""

from __future__ import annotations

import re
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"

# step:<step>/<train_steps> val_loss:<value>
VAL_LINE_RE = re.compile(r"step:(\d+)/\d+\s+val_loss:([0-9]*\.?[0-9]+)")


def parse_val_losses(log_path: str | Path) -> list[tuple[int, float]]:
    """Return a list of ``(step, val_loss)`` tuples parsed from a training log.

    Lines that do not contain a numeric validation loss (including the embedded
    source code at the top of the log) are skipped.
    """
    log_path = Path(log_path)
    results: list[tuple[int, float]] = []
    with log_path.open("r", errors="ignore") as file:
        for line in file:
            match = VAL_LINE_RE.search(line)
            if match is None:
                continue
            step = int(match.group(1))
            val_loss = float(match.group(2))
            results.append((step, val_loss))
    return results


def parse_log_dir(log_dir: str | Path = LOG_DIR) -> dict[str, list[tuple[int, float]]]:
    """Parse every ``*.txt`` log in a directory, keyed by file name."""
    log_dir = Path(log_dir)
    return {
        log_file.name: parse_val_losses(log_file)
        for log_file in sorted(log_dir.glob("*.txt"))
    }


def visualize(results: list[tuple[int, float]]) -> None:
    """Plot step vs. val_loss."""
    import matplotlib.pyplot as plt

    steps = [step for step, _ in results]
    losses = [loss for _, loss in results]
    plt.figure()
    plt.plot(steps, losses, marker="o")
    plt.xlabel("step")
    plt.ylabel("val_loss")
    plt.title("Validation loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    log_file = LOG_DIR / "2026-09-17_01-47-30_239404.txt" # "2026-09-16_15-55-10_210626.txt" #

    results = parse_val_losses(log_file)
    for step, val_loss in results:
        print(f"step {step:>6}  val_loss {val_loss:.5f}")


if __name__ == "__main__":
    main()
