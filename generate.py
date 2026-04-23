import argparse

import torch

from main import BigramLanguageModel, RunConfig, apply_run_config, load_checkpoint


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--device", default="")
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = RunConfig(**checkpoint["config"])

    if args.device:
        config.device = args.device

    apply_run_config(config)
    tokenizer = checkpoint["tokenizer"]
    chars = tokenizer["chars"]
    stoi = tokenizer["stoi"]
    itos = tokenizer["itos"]

    model = BigramLanguageModel(vocab_size=len(chars)).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.prompt:
        unknown_chars = [ch for ch in args.prompt if ch not in stoi]
        if unknown_chars:
            unique_chars = sorted(set(unknown_chars))
            raise ValueError(
                f"Prompt contains unknown characters: {unique_chars}"
            )
        context_tokens = [stoi[ch] for ch in args.prompt]
        context = torch.tensor([context_tokens], dtype=torch.long, device=config.device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

    generated = model.generate(context, max_new_tokens=args.max_new_tokens)
    output = "".join([itos[idx] for idx in generated[0].tolist()])
    print(output)


if __name__ == "__main__":
    main()
