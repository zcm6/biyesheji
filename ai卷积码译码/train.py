from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data import make_numpy_batch
from media_validation import (
    build_validation_cases,
    evaluate_cases,
    format_report,
    report_to_dict,
)
from model import ConvCodeRNNDecoder, RNNDecoderConfig, config_to_dict, require_torch, torch, nn


MODEL_DIR = Path(__file__).resolve().parent / "models"


def default_output_path(modulation: str, order: int, channel_name: str) -> Path:
    return MODEL_DIR / f"bigru_{modulation.lower()}_{int(order)}_{channel_name.lower()}.pt"


def default_media_output_path(modulation: str, order: int, channel_name: str) -> Path:
    return MODEL_DIR / f"bigru_{modulation.lower()}_{int(order)}_{channel_name.lower()}_media_ber.pt"


def best_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(".best.pt")


def checkpoint_payload(
    model,
    config: RNNDecoderConfig,
    args: argparse.Namespace,
    step: int,
    validation: dict | None = None,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "config": config_to_dict(config),
        "train_args": vars(args),
        "step": step,
        "validation": validation,
    }


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Train a small BiGRU decoder for the project's convolutional code.")
    parser.add_argument("--max-steps", "--steps", dest="max_steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--information-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=12.0)
    parser.add_argument("--input-mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--modulation", choices=["MASK", "MPSK", "MQAM"], default="MQAM")
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--channel-name", default="AWGN")
    parser.add_argument("--k-factor", type=float, default=3.0)
    parser.add_argument("--roll-off", type=float, default=0.35)
    parser.add_argument("--gray-ok", action="store_true", default=True)
    parser.add_argument("--no-gray", dest="gray_ok", action="store_false")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--media-validation", action="store_true")
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--validation-snrs", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12])
    parser.add_argument("--validation-repeats", type=int, default=3)
    parser.add_argument("--validation-seed", type=int, default=20260521)
    parser.add_argument("--validation-window-steps", type=int, default=8192)
    parser.add_argument("--source-method", default="哈夫曼编码")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    config = RNNDecoderConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=True,
    )
    model = ConvCodeRNNDecoder(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    validation_cases = None
    output_path = args.output or (
        default_media_output_path(args.modulation, args.order, args.channel_name)
        if args.media_validation
        else default_output_path(args.modulation, args.order, args.channel_name)
    )
    best_path = best_output_path(output_path)
    best_report = None
    best_metric = float("inf")
    best_step = 0

    if args.media_validation:
        print(
            f"building fixed media validation cases for {args.modulation}-{args.order} "
            f"snrs={args.validation_snrs} repeats={args.validation_repeats}"
        )
        validation_cases = build_validation_cases(
            modulation=args.modulation,
            order=args.order,
            channel_name=args.channel_name,
            snrs=tuple(args.validation_snrs),
            repeats=args.validation_repeats,
            input_mode=args.input_mode,
            source_method=args.source_method,
            k_factor=args.k_factor,
            roll_off=args.roll_off,
            gray_ok=args.gray_ok,
            seed=args.validation_seed,
        )
        print(f"fixed media validation cases ready: {len(validation_cases)}")

    model.train()
    for step in range(1, args.max_steps + 1):
        x_np, y_np = make_numpy_batch(
            batch_size=args.batch_size,
            information_len=args.information_len,
            snr_min=args.snr_min,
            snr_max=args.snr_max,
            rng=rng,
            input_mode=args.input_mode,
            modulation=args.modulation,
            order=args.order,
            channel_name=args.channel_name,
            k_factor=args.k_factor,
            roll_off=args.roll_off,
            gray_ok=args.gray_ok,
        )
        x = torch.from_numpy(x_np).to(args.device)
        y = torch.from_numpy(y_np).to(args.device)

        logits = model(x)
        # The last convolutional-code steps are zero termination bits, while
        # validation BER is measured only on the original information bits.
        loss = criterion(
            logits[:, : args.information_len],
            y[:, : args.information_len],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0:
            print(f"step={step:5d} loss={loss.item():.6f}")

        should_validate = (
            validation_cases is not None
            and (step % args.validate_every == 0 or step == args.max_steps)
        )
        if should_validate:
            report = evaluate_cases(
                model,
                validation_cases,
                args.input_mode,
                args.device,
                window_steps=args.validation_window_steps,
            )
            print(format_report(step, report))
            if report.ai_avg_ber < best_metric:
                best_metric = report.ai_avg_ber
                best_report = report
                best_step = step
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint_payload(
                        model,
                        config,
                        args,
                        step,
                        validation=report_to_dict(report),
                    ),
                    best_path,
                )
                print(f"saved best media checkpoint: {best_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if validation_cases is None:
        torch.save(checkpoint_payload(model, config, args, args.max_steps), output_path)
        print(f"saved: {output_path}")
        return

    if best_report is None:
        raise RuntimeError("Media validation was enabled but no validation report was produced.")

    print(f"best media checkpoint: step={best_step} {format_report(None, best_report)}")
    if best_report.accepted:
        accepted_checkpoint = torch.load(best_path, map_location="cpu")
        torch.save(accepted_checkpoint, output_path)
        print(f"saved accepted media BER model: {output_path}")
    else:
        print(
            "best media checkpoint did not satisfy "
            "AI average BER <= Viterbi average BER; final model was not written."
        )


if __name__ == "__main__":
    main()
