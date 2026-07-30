# Teacher-forced lambada over the HF rwkv7 implementation, three arms:
#   baseline        bf16 as published (tonight's bit-verified 1.5B artifact)
#   big-w8          big projections int8-g64 round-tripped (the w8g64 recipe)
#   big+lora-w8     the same plus every LoRA down/up matrix (Bo: low-rank can be w8)
# Vectors and norms are never touched in any arm. Round-trip = quantize->dequantize
# in place, so the stock model serves the rounded weights: measures the accuracy of
# the scheme with zero kernel surgery.
#
# Scoring is lm-eval's accuracy notion: the target word's tokens must each be the
# argmax at their position given the gold prefix. Calibration gate: baseline must
# land near the published lm-eval fp16 number (0.6728) or the harness is wrong.
# --sabotage scores one position off and must collapse to ~0.
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM

GROUP = 64
BIG_RE = re.compile(r"\.(att\.(receptance|key|value|output)|ffn\.(key|value))\.weight$")
LORA_NAMES = tuple(f"att.{n}" for n in ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2"))


def roundtrip_(p: torch.Tensor, axis: int) -> str:
    """Symmetric int8 round-trip along the contraction axis, group 64 when it
    divides, otherwise one scale per full axis. Returns a tag for the audit log."""
    W = p.detach().float()
    if axis == 0:
        W = W.t()  # -> [N, K] with K the contraction dim
    N, K = W.shape
    if K % GROUP == 0:
        g, tag = GROUP, f"g{GROUP}"
    else:
        g, tag = K, f"per-row(K={K})"
    Wg = W.reshape(N, K // g, g)
    scale = (Wg.abs().amax(dim=2, keepdim=True) / 127.0).clamp(min=1e-8)
    q = torch.round(Wg / scale).clamp_(-127, 127)
    out = (q * scale).reshape(N, K)
    if axis == 0:
        out = out.t()
    p.data.copy_(out.to(p.dtype))
    return tag


def apply_arm(model, arm: str, hidden: int) -> None:
    if arm == "baseline":
        return
    counts = {}
    for name, p in model.named_parameters():
        if BIG_RE.search(name):
            tag = roundtrip_(p, axis=1)
            counts[f"big {tag}"] = counts.get(f"big {tag}", 0) + 1
        elif arm == "big+lora-w8" and name.endswith(LORA_NAMES):
            # x @ W chains: down mats are [hidden, rank], up mats [rank, hidden];
            # the contraction is axis 0 for both. Assert so a layout change screams.
            assert p.dim() == 2 and (p.shape[0] == hidden or p.shape[1] == hidden), name
            tag = roundtrip_(p, axis=0)
            counts[f"lora {tag}"] = counts.get(f"lora {tag}", 0) + 1
    print(f"[{arm}] quantized:", dict(sorted(counts.items())), flush=True)


@torch.no_grad()
def score(model, data, device, sabotage=False, limit=None) -> float:
    correct = total = 0
    for i, ex in enumerate(data):
        if limit and i >= limit:
            break
        ctx, tgt = ex["ctx"], ex["tgt"]
        ids = torch.tensor([ctx + tgt], device=device)
        logits = model(input_ids=ids).logits[0]
        start = len(ctx) - 1 + (1 if sabotage else 0)
        pred = logits[start : start + len(tgt)].argmax(dim=-1).tolist()
        correct += int(pred == tgt)
        total += 1
        if total % 1000 == 0:
            print(f"  {total}: acc so far {correct/total:.4f}", flush=True)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/hubout/rwkv7-1.5b-hf")
    ap.add_argument("--data", default="/data/lambada_ids.json")
    ap.add_argument("--arms", nargs="*", default=["baseline", "big-w8", "big+lora-w8"])
    ap.add_argument("--sabotage", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data = json.load(open(args.data))
    print(f"{len(data)} examples", flush=True)
    for arm in args.arms:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
        apply_arm(model, arm, model.config.hidden_size)
        acc = score(model, data, "cuda", sabotage=args.sabotage, limit=args.limit)
        print(f"RESULT {arm}: acc {acc:.4f}" + (" (SABOTAGE)" if args.sabotage else ""), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
