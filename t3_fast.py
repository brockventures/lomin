"""CUDA-graph single-token decoder for Chatterbox Turbo's T3 (GPT-2 medium).

Why: the eager decode loop in desktop_tts._t3_token_iter costs ~26 ms/token
on the RTX 4090 and is kernel-launch bound (24 layers x ~15 tiny kernels plus
the HF cache bookkeeping and the logits processors, all launched from python
every step). Here the whole step -- embedding, 24 GPT-2 blocks over a static
KV cache, LM head, the four logits processors, and the multinomial sample --
is captured ONCE into a CUDA graph at model load and replayed per token, so a
step is one launch.

Same math as the eager path (same weights, same processors in the same
order: temperature, top-k, top-p, repetition penalty on already-generated ids),
just fused: GPT-2 blocks are re-implemented over the model's own parameters
(no copies in fp32 mode), prefill still runs through the stock HF module and
its KV is copied into the static cache.

Everything torch-specific is imported lazily so desktop_tts.py stays
importable (and unit-testable) without torch.
"""

from __future__ import annotations

import sys


class T3FastDecoder:
    def __init__(self, t3, *, device="cuda", max_len=2048, dtype=None,
                 temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2):
        import torch
        self.torch = torch
        self.t3 = t3
        self.device = device
        self.max_len = max_len
        self.dtype = dtype or torch.float32
        self.temperature, self.top_k, self.top_p, self.rep = temperature, top_k, top_p, repetition_penalty
        cfg = t3.cfg
        self.n_layer, self.n_head = cfg.n_layer, cfg.n_head
        self.dim = cfg.hidden_size
        self.head_dim = self.dim // self.n_head
        self.eps = cfg.layer_norm_epsilon
        self.vocab = t3.speech_head.out_features
        self.graph = None
        self._collect_weights()
        L, H, D = max_len, self.n_head, self.head_dim
        kw = dict(device=device, dtype=self.dtype)
        self.k_cache = [torch.zeros(1, H, L, D, **kw) for _ in range(self.n_layer)]
        self.v_cache = [torch.zeros(1, H, L, D, **kw) for _ in range(self.n_layer)]
        self.mask = torch.full((1, 1, 1, L), float("-inf"), **kw)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.cur_tok = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.seen = torch.zeros(1, self.vocab, dtype=torch.bool, device=device)
        self.out_tok = torch.zeros(1, 1, dtype=torch.long, device=device)

    # -- weights -----------------------------------------------------------
    def _collect_weights(self):
        torch, dt = self.torch, self.dtype
        t3 = self.t3

        def cv(p):
            return p.detach().to(dt) if dt != p.dtype else p.detach()

        self.layers = []
        for blk in t3.tfmr.h:
            self.layers.append(dict(
                ln1=(cv(blk.ln_1.weight), cv(blk.ln_1.bias)),
                attn_w=cv(blk.attn.c_attn.weight), attn_b=cv(blk.attn.c_attn.bias),
                proj_w=cv(blk.attn.c_proj.weight), proj_b=cv(blk.attn.c_proj.bias),
                ln2=(cv(blk.ln_2.weight), cv(blk.ln_2.bias)),
                fc_w=cv(blk.mlp.c_fc.weight), fc_b=cv(blk.mlp.c_fc.bias),
                mp_w=cv(blk.mlp.c_proj.weight), mp_b=cv(blk.mlp.c_proj.bias),
            ))
        self.ln_f = (cv(t3.tfmr.ln_f.weight), cv(t3.tfmr.ln_f.bias))
        self.wpe = cv(t3.tfmr.wpe.weight)
        self.speech_emb_w = cv(t3.speech_emb.weight)
        self.head_w = cv(t3.speech_head.weight)  # (vocab, dim)
        self.head_b = cv(t3.speech_head.bias) if t3.speech_head.bias is not None else None

    # -- one decode step (this body is what gets captured) -------------------
    def _step(self):
        torch = self.torch
        F = torch.nn.functional
        H, D, dim = self.n_head, self.head_dim, self.dim
        pos = self.pos
        x = (self.speech_emb_w[self.cur_tok[0]] + self.wpe[pos]).view(1, 1, dim)
        self.mask.index_fill_(3, pos, 0.0)
        for i, ly in enumerate(self.layers):
            h = F.layer_norm(x, (dim,), ly["ln1"][0], ly["ln1"][1], self.eps)
            qkv = torch.addmm(ly["attn_b"], h.view(1, dim), ly["attn_w"])
            q, k, v = qkv.split(dim, dim=1)
            q = q.view(1, H, 1, D)
            self.k_cache[i].index_copy_(2, pos, k.reshape(1, H, 1, D))
            self.v_cache[i].index_copy_(2, pos, v.reshape(1, H, 1, D))
            a = F.scaled_dot_product_attention(q, self.k_cache[i], self.v_cache[i], attn_mask=self.mask)
            a = torch.addmm(ly["proj_b"], a.reshape(1, dim), ly["proj_w"])
            x = x + a.view(1, 1, dim)
            h = F.layer_norm(x, (dim,), ly["ln2"][0], ly["ln2"][1], self.eps)
            m = torch.addmm(ly["fc_b"], h.view(1, dim), ly["fc_w"])
            m = F.gelu(m, approximate="tanh")
            m = torch.addmm(ly["mp_b"], m, ly["mp_w"])
            x = x + m.view(1, 1, dim)
        x = F.layer_norm(x, (dim,), self.ln_f[0], self.ln_f[1], self.eps)
        logits = F.linear(x.view(1, dim), self.head_w, self.head_b).float()
        nxt = self._sample(logits)
        self.seen.scatter_(1, nxt, True)
        self.cur_tok.copy_(nxt)
        self.out_tok.copy_(nxt)
        pos.add_(1)

    def _sample(self, scores):
        """HF-equivalent processor chain, all on device, no host sync."""
        torch = self.torch
        neg_inf = float("-inf")
        if self.temperature not in (0, 1.0):
            scores = scores / self.temperature
        if self.top_k > 0:
            k = min(self.top_k, scores.size(-1))
            kth = torch.topk(scores, k)[0][..., -1, None]
            scores = scores.masked_fill(scores < kth, neg_inf)
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scores, descending=False)
            cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cum <= (1 - self.top_p)
            remove[..., -1:] = False
            remove = remove.scatter(1, sorted_idx, remove)
            scores = scores.masked_fill(remove, neg_inf)
        if self.rep != 1.0:
            pen = torch.where(scores < 0, scores * self.rep, scores / self.rep)
            scores = torch.where(self.seen, pen, scores)
        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # -- capture ----------------------------------------------------------------
    def capture(self):
        """Warm up on a side stream, then capture one step into a CUDA graph.
        Cache/mask/pos state is garbage afterwards; begin() resets it."""
        torch = self.torch
        self.begin_dummy()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(s):
            for _ in range(3):
                self.begin_dummy()
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            self._step()
        torch.cuda.synchronize()
        self.graph = g

    def begin_dummy(self):
        self.mask.fill_(float("-inf"))
        self.mask[..., :4] = 0
        self.pos.fill_(4)
        self.seen.zero_()
        self.cur_tok.zero_()

    # -- per-request ----------------------------------------------------------------
    def begin(self, past, first_tok: int):
        """Load the prefill KV (HF DynamicCache) and the first sampled token."""
        torch = self.torch
        layers = getattr(past, "layers", None)
        n = None
        for i in range(self.n_layer):
            if layers is not None:
                k, v = layers[i].keys, layers[i].values
            else:  # older transformers
                k, v = past[i]
            n = k.shape[2]
            if n + 1 > self.max_len:
                raise ValueError(f"prefill {n} exceeds static cache {self.max_len}")
            self.k_cache[i][:, :, :n].copy_(k)
            self.v_cache[i][:, :, :n].copy_(v)
        self.mask.fill_(float("-inf"))
        self.mask[..., :n] = 0
        self.pos.fill_(n)
        self.seen.zero_()
        self.seen[0, first_tok] = True
        self.cur_tok.fill_(first_tok)
        self.remaining = self.max_len - n - 1

    def step(self) -> int:
        """Run one graph replay; returns the new token (one device sync)."""
        if self.remaining <= 0:
            raise ValueError("static KV cache full")
        self.remaining -= 1
        self.graph.replay()
        return int(self.out_tok)


def try_build(t3, *, device, max_len, dtype_name="fp32"):
    """Build + capture, or return None (one stderr line) on any failure."""
    try:
        import torch
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
        dec = T3FastDecoder(t3, device=device, max_len=max_len, dtype=dtype)
        dec.capture()
        return dec
    except Exception as exc:  # noqa: BLE001 - any failure means "stay on eager"
        print(f"[desktop-tts] T3 fast path unavailable, using eager: {exc!r}", file=sys.stderr, flush=True)
        return None
