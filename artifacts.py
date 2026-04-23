import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from config import RunConfig


def create_run_dirs(output_root: str, run_name: str) -> dict[str, Path]:
    run_dir = Path(output_root) / run_name
    paths = {
        "run_dir": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "logs": run_dir / "logs",
        "samples": run_dir / "samples",
        "metrics": run_dir / "metrics",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_run_config(config: RunConfig, config_path: Path) -> None:
    with open(config_path, "w", encoding="utf-8") as file_handle:
        json.dump(asdict(config), file_handle, indent=2)


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    config: RunConfig,
    chars: list[str],
    stoi: dict[str, int],
    itos: dict[int, str],
) -> None:
    checkpoint = {
        "step": step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "tokenizer": {"chars": chars, "stoi": stoi, "itos": itos},
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    tokenizer = checkpoint["tokenizer"]
    tokenizer["itos"] = {int(key): value for key, value in tokenizer["itos"].items()}
    return checkpoint


def append_metrics_row(
    metrics_path: Path,
    step: int,
    train_loss: float,
    val_loss: float,
    learning_rate_value: float,
    tokens_per_second: float,
) -> None:
    should_write_header = not metrics_path.exists()
    with open(metrics_path, "a", encoding="utf-8", newline="") as file_handle:
        writer = csv.writer(file_handle)
        if should_write_header:
            writer.writerow(
                [
                    "step",
                    "train_loss",
                    "val_loss",
                    "learning_rate",
                    "tokens_per_second",
                ]
            )
        writer.writerow(
            [
                step,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{learning_rate_value:.6e}",
                f"{tokens_per_second:.2f}",
            ]
        )


@torch.no_grad()
def save_sample_snapshot(
    model: nn.Module,
    decode: Callable[[list[int]], str],
    sample_dir: Path,
    step: int,
    sample_tokens: int,
    runtime_device: str,
) -> None:
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=runtime_device)
    sample_ids = model.generate(context, max_new_tokens=sample_tokens)[0].tolist()
    sample_text = decode(sample_ids)
    sample_path = sample_dir / f"sample_step_{step:07d}.txt"
    with open(sample_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(sample_text)
    model.train()
