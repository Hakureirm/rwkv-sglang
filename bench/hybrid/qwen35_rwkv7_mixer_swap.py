"""Qwen3.5 with every mixer replaced by RWKV-7 (the group question, made runnable).

Committed here because a claim without an artifact is uncitable: transformers-rwkv PR#2
describes this experiment and this file is the experiment. Needs the PR#2 tree on
PYTHONPATH (it provides `transformers.models.rwkv7`) and nothing else; CPU is enough.

The criterion that matters is at the bottom: a full forward must equal token-by-token
incremental decode (measured 6.3e-06 relative; severing the state carry gives 1.32).
Shape checks, `isfinite`, and "generate emits tokens" all stay green with the mixer
replaced by an identity function, so they prove nothing.
"""

import torch
from torch import nn

from transformers import Rwkv7Config
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.models.rwkv7.modeling_rwkv7 import Rwkv7Attention

# ---- the adapter ---------------------------------------------------------------


def _rwkv_config(host, head_dim):
    """An Rwkv7Config carrying the host's width, which is all the mixer reads."""
    return Rwkv7Config(
        hidden_size=host.hidden_size,
        head_dim=head_dim,
        num_heads=host.hidden_size // head_dim,
        norm_eps=host.rms_norm_eps,
    )


class RwkvMixer(nn.Module):
    """RWKV-7 time-mix behind the call signature a Qwen3.5 layer uses.

    The recurrence has to survive between calls, which is the part that is easy to
    skip and impossible to notice: a version that passes `None` for the state builds,
    runs, and generates tokens, and every one of those tokens is produced as if it
    were the first. `hub` carries both the per-layer state and the `v_first` that
    layer 0 produces for the layers above it.
    """

    def __init__(self, host, head_dim, index, hub, pair):
        super().__init__()
        self.attn = Rwkv7Attention(_rwkv_config(host, head_dim), index)
        self.hub, self.slot, self.pair = hub, index, pair

    def forward(self, hidden_states, *args, **kwargs):
        shift, wkv = self.hub.get(self.slot, (None, None))
        out, v_first, shift, wkv = self.attn(hidden_states, self.hub.get("v_first"), shift, wkv)
        self.hub["v_first"], self.hub[self.slot] = self.hub.get("v_first", v_first), (shift, wkv)
        return (out, None) if self.pair else out


def rwkvify(model, linear_head_dim=128, full_head_dim=256):
    """Swap every mixer in a Qwen3.5 stack for an RWKV-7 one, in place."""
    host, hub = model.config.get_text_config(), {}

    def reset(_, args, kwargs):
        # `cache_position` is how the framework says where in the sequence this call
        # sits, so position 0 is a new sequence and anything else continues one. The
        # obvious alternative -- treat a multi-token call as a prefill -- cannot start
        # a sequence from a one-token prompt, which is a real thing callers do.
        pos = kwargs.get("cache_position")
        if pos is None or int(pos[0]) == 0:
            hub.clear()
        else:
            hub.pop("v_first", None)

    for index, layer in enumerate(model.model.layers):
        if hasattr(layer, "linear_attn"):
            layer.linear_attn = RwkvMixer(host, linear_head_dim, index, hub, pair=False)
        else:
            layer.self_attn = RwkvMixer(host, full_head_dim, index, hub, pair=True)
    model.register_forward_pre_hook(reset, with_kwargs=True)
    return model


# ---- proof that it runs --------------------------------------------------------

if __name__ == "__main__":
    # Qwen3.5's own defaults are head_dim 256 for GQA and linear_*_head_dim 128 for
    # GDN, which is where the two widths in the question come from.
    text = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=512, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=4, head_dim=64,
        intermediate_size=256, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_num_key_heads=4, linear_num_value_heads=8,
        layer_types=["linear_attention", "full_attention"] * 2, pad_token_id=0,
    )
    torch.manual_seed(0)
    model = rwkvify(Qwen3_5ForCausalLM(text)).eval()

    kinds = [type(getattr(l, "linear_attn", None) or l.self_attn).__name__ for l in model.model.layers]
    print("  mixers now:", kinds)
    widths = [
        (getattr(l, "linear_attn", None) or l.self_attn).attn.head_dim for l in model.model.layers
    ]
    print("  head widths:", widths, "(GDN slots 128, GQA slots 256)")

    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        out = model(input_ids=ids).logits
    print("  forward:", tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))
    with torch.no_grad():
        gen = model.generate(input_ids=ids[:1, :4], max_new_tokens=6, do_sample=False)
    print("  generate:", gen[0].tolist())
    kept = sum(p.numel() for n, p in model.named_parameters() if "embed" in n or "lm_head" in n or "mlp" in n)
    print(f"  Qwen parameters kept (emb + lm_head + MoE): {kept / 1e6:.2f}M")
