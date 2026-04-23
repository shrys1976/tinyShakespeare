import argparse
import csv
import json
import logging
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import functional as F


DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
DEFAULT_DATA_PATH = "input.txt"
DEFAULT_OUTPUT_DIR = "runs"

batch_size = 64
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
max_iters = 10000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"


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


def apply_run_config(config: RunConfig) -> None:
    global batch_size
    global block_size
    global n_embd
    global n_head
    global n_layer
    global dropout
    global max_iters
    global eval_interval
    global eval_iters
    global learning_rate
    global device

    batch_size = config.batch_size
    block_size = config.block_size
    n_embd = config.n_embd
    n_head = config.n_head
    n_layer = config.n_layer
    dropout = config.dropout
    max_iters = config.max_iters
    eval_interval = config.eval_interval
    eval_iters = config.eval_iters
    learning_rate = config.learning_rate
    device = config.device


def download_dataset(data_path: str) -> None:
    if not os.path.exists(data_path):
        urllib.request.urlretrieve(DATA_URL, data_path)


def load_text(data_path: str) -> str:
    with open(data_path, "r", encoding="utf-8") as f:
        return f.read()


def build_tokenizer(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    return chars, stoi, itos, encode, decode


def build_codec_from_maps(stoi: dict[str, int], itos: dict[int, str]):
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    return encode, decode


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(

            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):

    # transformer block

    def __init__(self, n_embd, n_head):
        # multi head attentions
        # n_head -> number of embedding dims
        # n_head -> number of attention heads

        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        # residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


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
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


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
    stoi = tokenizer["stoi"]
    itos = {int(k): v for k, v in tokenizer["itos"].items()}
    tokenizer["itos"] = itos
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
    with open(metrics_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
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
) -> None:
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample_ids = model.generate(context, max_new_tokens=sample_tokens)[0].tolist()
    sample_text = decode(sample_ids)
    sample_path = sample_dir / f"sample_step_{step:07d}.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
    model.train()


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


def run_training(config: RunConfig, resume_path: str) -> None:
    apply_run_config(config)
    torch.manual_seed(config.seed)
    run_paths = create_run_dirs(config.output_root, config.run_name)
    logger = setup_logger(run_paths["logs"] / "train.log")
    save_run_config(config, run_paths["run_dir"] / "config.json")

    logger.info("Run directory: %s", run_paths["run_dir"])
    logger.info("Using device: %s", device)
    logger.info("Run config: %s", json.dumps(asdict(config), indent=2))

    download_dataset(config.data_path)
    text = load_text(config.data_path)

    start_step = 0
    best_val_loss = float("inf")
    checkpoint = load_checkpoint(resume_path) if resume_path else None

    if checkpoint:
        tokenizer = checkpoint["tokenizer"]
        chars = tokenizer["chars"]
        stoi = tokenizer["stoi"]
        itos = tokenizer["itos"]
        encode, decode = build_codec_from_maps(stoi, itos)
        logger.info("Loaded tokenizer from checkpoint: %s", resume_path)
    else:
        chars, stoi, itos, encode, decode = build_tokenizer(text)

    data = torch.tensor(encode(text), dtype=torch.long)
    split_idx = int(config.train_split * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    vocab_size = len(chars)

    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        logger.info("Resumed from step %d", start_step)

    metrics_file = run_paths["metrics"] / "metrics.csv"
    tokens_seen = max(0, start_step * batch_size * block_size)
    start_time = time.time()
    no_improve_evals = 0

    latest_ckpt_path = run_paths["checkpoints"] / "latest.pt"
    best_ckpt_path = run_paths["checkpoints"] / "best.pt"
    current_step = max(0, start_step)

    try:
        for step in range(start_step, max_iters):
            current_step = step
            xb, yb = get_batch("train", train_data, val_data)
            _, loss = model(xb, yb)

            if not torch.isfinite(loss):
                logger.error("Invalid loss at step %d: %s", step, loss.item())
                save_checkpoint(
                    run_paths["checkpoints"] / "nan_guard.pt",
                    model,
                    optimizer,
                    step,
                    best_val_loss,
                    config,
                    chars,
                    stoi,
                    itos,
                )
                raise RuntimeError("Loss became NaN/Inf, checkpoint saved.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            tokens_seen += batch_size * block_size

            should_eval = (step % eval_interval == 0) or (step == max_iters - 1)
            if should_eval:
                losses = estimate_loss(model, train_data, val_data)
                elapsed = max(1e-9, time.time() - start_time)
                tokens_per_second = tokens_seen / elapsed
                train_loss = losses["train"].item()
                val_loss = losses["val"].item()

                logger.info(
                    "step %d | train %.4f | val %.4f | tok/s %.2f",
                    step,
                    train_loss,
                    val_loss,
                    tokens_per_second,
                )
                append_metrics_row(
                    metrics_file,
                    step,
                    train_loss,
                    val_loss,
                    learning_rate,
                    tokens_per_second,
                )
                save_sample_snapshot(
                    model,
                    decode,
                    run_paths["samples"],
                    step,
                    config.sample_tokens,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_evals = 0
                    save_checkpoint(
                        best_ckpt_path,
                        model,
                        optimizer,
                        step,
                        best_val_loss,
                        config,
                        chars,
                        stoi,
                        itos,
                    )
                    logger.info(
                        "saved best checkpoint at step %d (val %.4f)",
                        step,
                        val_loss,
                    )
                else:
                    no_improve_evals += 1

                if (
                    config.early_stop_patience_evals > 0
                    and no_improve_evals >= config.early_stop_patience_evals
                ):
                    logger.info(
                        "Early stop triggered at step %d after %d evals "
                        "without improvement.",
                        step,
                        no_improve_evals,
                    )
                    save_checkpoint(
                        latest_ckpt_path,
                        model,
                        optimizer,
                        step,
                        best_val_loss,
                        config,
                        chars,
                        stoi,
                        itos,
                    )
                    break

                elapsed_hours = (time.time() - start_time) / 3600.0
                estimated_cost = elapsed_hours * config.hourly_cost_usd
                if (
                    config.budget_cap_usd > 0
                    and config.hourly_cost_usd > 0
                    and estimated_cost >= config.budget_cap_usd
                ):
                    logger.info(
                        "Budget cap reached at step %d (estimated $%.3f).",
                        step,
                        estimated_cost,
                    )
                    save_checkpoint(
                        latest_ckpt_path,
                        model,
                        optimizer,
                        step,
                        best_val_loss,
                        config,
                        chars,
                        stoi,
                        itos,
                    )
                    break

                if config.max_wall_time_minutes > 0:
                    elapsed_minutes = (time.time() - start_time) / 60.0
                    if elapsed_minutes >= config.max_wall_time_minutes:
                        logger.info(
                            "Wall-time cap reached at step %d "
                            "(elapsed %.2f minutes).",
                            step,
                            elapsed_minutes,
                        )
                        save_checkpoint(
                            latest_ckpt_path,
                            model,
                            optimizer,
                            step,
                            best_val_loss,
                            config,
                            chars,
                            stoi,
                            itos,
                        )
                        break

            should_checkpoint = (
                step % config.checkpoint_interval == 0
            ) or (step == max_iters - 1)
            if should_checkpoint:
                save_checkpoint(
                    latest_ckpt_path,
                    model,
                    optimizer,
                    step,
                    best_val_loss,
                    config,
                    chars,
                    stoi,
                    itos,
                )

    except KeyboardInterrupt:
        interrupt_path = run_paths["checkpoints"] / "interrupt.pt"
        save_checkpoint(
            interrupt_path,
            model,
            optimizer,
            current_step,
            best_val_loss,
            config,
            chars,
            stoi,
            itos,
        )
        logger.warning("Interrupted. Saved emergency checkpoint: %s", interrupt_path)
        return

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    print(output_text)
    logger.info("Training completed. Best val loss: %.4f", best_val_loss)


def main() -> None:
    args = create_arg_parser().parse_args()
    config = config_from_args(args)

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        checkpoint_config = RunConfig(**checkpoint["config"])
        if args.max_iters is not None:
            checkpoint_config.max_iters = args.max_iters
        if args.run_name:
            checkpoint_config.run_name = args.run_name
        if args.output_root:
            checkpoint_config.output_root = args.output_root
        config = checkpoint_config

    run_training(config=config, resume_path=args.resume)


if __name__ == "__main__":
    main()
