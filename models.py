# =============================================================
# C2C Model Definitions
# =============================================================

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache


# -------------------------------------------------------------
# Gumbel-Sigmoid Gate
# -------------------------------------------------------------


def gumbel_sigmoid(logits, temperature=1.0, hard=False, eps=1e-10):
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + eps) + eps)
    y = torch.sigmoid((logits + g) / temperature)

    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard.detach() - y.detach() + y
    return y


# -------------------------------------------------------------
# KV Cache Utilities
# -------------------------------------------------------------


def rebuild_past_key_values_like_receiver(_, fused_keys, fused_values):
    new_cache = DynamicCache()
    for i in range(fused_keys.shape[0]):
        new_cache.update(fused_keys[i], fused_values[i], layer_idx=i)
    return new_cache


# -------------------------------------------------------------
# C2C Fuser Module
# -------------------------------------------------------------
class C2CFuser(nn.Module):
    def __init__(self, model_receiver, model_sharer):
        super().__init__()
        self.sanity_check = False
        c1, c2 = model_receiver.config, model_sharer.config

        self.num_layers = c1.num_hidden_layers
        self.n_kv_heads = c1.num_key_value_heads

        fused_head_dim = c1.head_dim + c2.head_dim
        receiver_head_dim = c1.head_dim

        self.projection_k = nn.Linear(fused_head_dim, receiver_head_dim)
        self.projection_v = nn.Linear(fused_head_dim, receiver_head_dim)

        self.ff = nn.Sequential(
            nn.Linear(receiver_head_dim, receiver_head_dim),
            nn.GELU(),
            nn.Linear(receiver_head_dim, receiver_head_dim),
        )

        self.dw = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.n_kv_heads, self.n_kv_heads, 1),
            nn.Sigmoid(),
        )

        self.gate_param = nn.Parameter(torch.zeros(self.num_layers))
        self.temperature = 0.5

        ref = next(model_receiver.parameters())
        self.to(device=ref.device, dtype=torch.float32)

    def _layer_kv(self, cache, idx):
        layer = cache.layers[idx]
        return layer.keys, layer.values

    def forward(self, cache_receiver, cache_sharer):
        fused_keys, fused_values = [], []

        Lr, Ls = len(cache_receiver.layers), len(cache_sharer.layers)
        L = min(Lr, Ls, self.num_layers)
        sh_start = Ls - L

        for i in range(L):
            K_R, V_R = self._layer_kv(cache_receiver, i)
            K_S, V_S = self._layer_kv(cache_sharer, sh_start + i)

            K_R, V_R = K_R.float(), V_R.float()
            K_S, V_S = K_S.float(), V_S.float()

            K_cat = torch.cat([K_R, K_S], dim=-1)
            V_cat = torch.cat([V_R, V_S], dim=-1)

            fused_K = self.ff(self.projection_k(K_cat))
            fused_V = self.ff(self.projection_v(V_cat))

            fused_K = fused_K * self.dw(fused_K)
            fused_V = fused_V * self.dw(fused_V)

            gate = gumbel_sigmoid(self.gate_param[i], self.temperature)

            if self.training:
                out_K = K_R + gate * fused_K
                out_V = V_R + gate * fused_V
            else:
                b = 1 if gate > 0.8 else 0
                out_K = K_R + b * fused_K
                out_V = V_R + b * fused_V

            if self.sanity_check:
                # This sanity check will replace the gate with an alpha value
                alpha = 1
                out_K = alpha * K_R + (1 - alpha) * fused_K
                out_V = alpha * V_R + (1 - alpha) * fused_V

            out_K = out_K.to(cache_receiver.layers[i].keys.dtype)
            out_V = out_V.to(cache_receiver.layers[i].values.dtype)

            fused_keys.append(out_K)
            fused_values.append(out_V)

        return rebuild_past_key_values_like_receiver(
            cache_receiver,
            torch.stack(fused_keys),
            torch.stack(fused_values),
        )

    def set_temperature(self, temperature: float):
        """Set global gate temperature (useful for annealing from the training loop)."""
        self.temperature = float(temperature)


# -------------------------------------------------------------
# Receiver Wrapper
# -------------------------------------------------------------
class C2CReceiverForCausalLM(nn.Module):
    def __init__(self, model_rec, model_share, tokenizer=None):
        super().__init__()
        self.receiver = model_rec
        self.sharer = model_share
        self.c2c = C2CFuser(model_rec, model_share)
        self.tokenizer = tokenizer

        self._freeze(self.receiver)
        self._freeze(self.sharer)

    def _freeze(self, model):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    def forward(self, prompt_ids, prompt_mask, resp_ids, resp_mask, labels):
        with torch.no_grad():
            rec = self.receiver(prompt_ids, prompt_mask, use_cache=True)
            sha = self.sharer(prompt_ids, prompt_mask, use_cache=True)

        fused_cache = self.c2c(rec.past_key_values, sha.past_key_values)
        full_mask = torch.cat([prompt_mask, resp_mask], dim=1)

        out = self.receiver(
            resp_ids,
            attention_mask=full_mask,
            past_key_values=fused_cache,
            labels=labels,
            use_cache=False,
        )
        return out.loss

    @torch.no_grad()
    def generate(self, prompt_input_ids, prompt_mask, max_new_tokens=10, tokenizer=None, **generate_kwargs):
        if tokenizer is None:
            tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("tokenizer must be provided either in __init__ or as parameter to generate()")

        device = prompt_input_ids.device

        # 1) Prefill Receiver + Sharer on the prompt
        rec_out = self.receiver(
            input_ids=prompt_input_ids,
            attention_mask=prompt_mask,
            use_cache=True,
        )
        sh_out = self.sharer(
            input_ids=prompt_input_ids,
            attention_mask=prompt_mask,
            use_cache=True,
        )

        # 2) Fuse prompt KV caches
        fused_past = self.c2c(
            rec_out.past_key_values,
            sh_out.past_key_values,
        )

        # 3) Force first token(s) to be '\n\n' regardless of logits
        anchor_text = "\n\n"
        anchor_ids = tokenizer(anchor_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        next_token = anchor_ids[0]  # Take first token of anchor

        # 4) Manual generation loop (simpler than using generate())
        generated_ids = []

        for _ in range(max_new_tokens):
            generated_ids.append(next_token.item())

            outputs = self.receiver(
                input_ids=next_token.unsqueeze(0),
                past_key_values=fused_past,
                use_cache=True,
            )

            fused_past = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        gen_ids = torch.tensor(generated_ids, device=device)
        return gen_ids.unsqueeze(0)
