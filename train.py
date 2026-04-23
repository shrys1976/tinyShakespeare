import json
import time

import torch

from artifacts import (
    append_metrics_row,
    create_run_dirs,
    load_checkpoint,
    save_checkpoint,
    save_run_config,
    save_sample_snapshot,
    setup_logger,
)
from config import RunConfig, config_from_args, create_arg_parser
from main import (
    BigramLanguageModel,
    apply_run_config,
    build_codec_from_maps,
    build_tokenizer,
    download_dataset,
    estimate_loss,
    get_batch,
    load_text,
)


def run_training(config: RunConfig, resume_path: str) -> None:
    apply_run_config(config)
    torch.manual_seed(config.seed)
    run_paths = create_run_dirs(config.output_root, config.run_name)
    logger = setup_logger(run_paths["logs"] / "train.log")
    save_run_config(config, run_paths["run_dir"] / "config.json")

    logger.info("Run directory: %s", run_paths["run_dir"])
    logger.info("Using device: %s", config.device)
    logger.info("Run config: %s", json.dumps(config.__dict__, indent=2))

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

    model = BigramLanguageModel(vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        logger.info("Resumed from step %d", start_step)

    metrics_file = run_paths["metrics"] / "metrics.csv"
    tokens_seen = max(0, start_step * config.batch_size * config.block_size)
    start_time = time.time()
    no_improve_evals = 0

    latest_ckpt_path = run_paths["checkpoints"] / "latest.pt"
    best_ckpt_path = run_paths["checkpoints"] / "best.pt"
    current_step = max(0, start_step)

    try:
        for step in range(start_step, config.max_iters):
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
            tokens_seen += config.batch_size * config.block_size

            should_eval = (
                step % config.eval_interval == 0
            ) or (step == config.max_iters - 1)
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
                    config.learning_rate,
                    tokens_per_second,
                )
                save_sample_snapshot(
                    model,
                    decode,
                    run_paths["samples"],
                    step,
                    config.sample_tokens,
                    config.device,
                )

                improvement = best_val_loss - val_loss
                if improvement > config.early_stop_min_delta:
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
                        "without sufficient improvement.",
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
            ) or (step == config.max_iters - 1)
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

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
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
        if args.data_path:
            checkpoint_config.data_path = args.data_path
        config = checkpoint_config

    run_training(config=config, resume_path=args.resume)


if __name__ == "__main__":
    main()
