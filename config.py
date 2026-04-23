import argparse
from dataclasses import dataclass
from datetime import datetime

import torch

DEFAULT_DATA_PATH = "input.txt"
DEFAULT_OUTPUT_DIR = "runs"


@dataclass
class RunConfig:
    run_name: str
    output_root: str = DEFAULT_OUTPUT_DIR
    data_path: str = DEFAULT_DATA_PATH
    seed: int = 42
    batch_size: int = 64
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    max_iters: int = 10000
    eval_interval: int = 500
    eval_iters: int = 200
    checkpoint_interval: int = 500
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    train_split: float = 0.9
    sample_tokens: int = 500
    early_stop_patience_evals: int = 0
    early_stop_min_delta: float = 0.0
    max_wall_time_minutes: float = 0.0
    hourly_cost_usd: float = 0.0
    budget_cap_usd: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train char-level nanoGPT")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--sample-tokens", type=int, default=None)
    parser.add_argument("--early-stop-patience-evals", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--max-wall-time-minutes", type=float, default=None)
    parser.add_argument("--hourly-cost-usd", type=float, default=None)
    parser.add_argument("--budget-cap-usd", type=float, default=None)
    parser.add_argument("--resume", default="")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    config = RunConfig(
        run_name=run_name,
        output_root=args.output_root,
        data_path=args.data_path,
    )
    if args.max_iters is not None:
        config.max_iters = args.max_iters
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.eval_iters is not None:
        config.eval_iters = args.eval_iters
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = args.checkpoint_interval
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.grad_clip is not None:
        config.grad_clip = args.grad_clip
    if args.sample_tokens is not None:
        config.sample_tokens = args.sample_tokens
    if args.early_stop_patience_evals is not None:
        config.early_stop_patience_evals = args.early_stop_patience_evals
    if args.early_stop_min_delta is not None:
        config.early_stop_min_delta = args.early_stop_min_delta
    if args.max_wall_time_minutes is not None:
        config.max_wall_time_minutes = args.max_wall_time_minutes
    if args.hourly_cost_usd is not None:
        config.hourly_cost_usd = args.hourly_cost_usd
    if args.budget_cap_usd is not None:
        config.budget_cap_usd = args.budget_cap_usd
    return config
