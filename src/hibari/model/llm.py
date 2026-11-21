"""Hibari Koto — Decoder-only Autoregressive Language Model Architecture."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from hibari.model.config import ModelConfig
from hibari.model.embedding import TokenEmbedding, precompute_rope_freqs
from hibari.model.normalization import RMSNorm
from hibari.model.transformer import TransformerBlock


class HibariKotoLM(nn.Module):
    """Hibari Koto decoder-only transformer (LLaMA-style: RMSNorm, RoPE, SwiGLU, tied embeddings).

    Size is set entirely by ``ModelConfig``; Hibari Koto
    (configs/model/koto.yaml) has 6,163,712 parameters.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = TokenEmbedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    norm_eps=config.norm_eps,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Final RMSNorm
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Output LM Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.embedding.weight

        # Precompute RoPE buffers (registered as non-persistent buffers)
        cos, sin = precompute_rope_freqs(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass for language modeling.

        Args:
            input_ids: [batch_size, seq_len]
            targets: [batch_size, seq_len] optional for loss calculation
            attention_mask: optional attention mask
            kv_caches: list of (k, v) cache per layer
            use_cache: whether to compute and return new kv_caches
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        past_len = kv_caches[0][0].shape[2] if kv_caches else 0

        # Padding mask: accept a [batch, total_len] 0/1 (or bool) mask, as produced
        # by DataCollatorForCausalLM, and convert it to an additive mask.
        if attention_mask is not None and attention_mask.dim() == 2:
            keep = attention_mask.to(torch.bool)
            attention_mask = torch.zeros(
                keep.shape, dtype=torch.float32, device=device
            ).masked_fill(~keep, float("-inf"))[:, None, None, :]

        # Ensure RoPE buffers cover every absolute position used in this call
        needed = past_len + seq_len
        if needed > self.rope_cos.shape[0]:
            cos, sin = precompute_rope_freqs(
                head_dim=self.config.head_dim,
                max_seq_len=needed + 128,
                theta=self.config.rope_theta,
                device=device,
            )
            self.rope_cos = cos
            self.rope_sin = sin

        cos = self.rope_cos.to(device)
        sin = self.rope_sin.to(device)

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        new_caches = [] if use_cache else None

        # Pass through transformer blocks
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_cache = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                mask=attention_mask,
                kv_cache=layer_cache,
                use_cache=use_cache,
            )
            if use_cache:
                new_caches.append(new_cache)

        # Final RMSNorm
        hidden_states = self.norm(hidden_states)

        # Project to vocabulary logits
        logits = self.lm_head(hidden_states)

        # Loss calculation if targets provided
        loss = None
        if targets is not None:
            # Shift targets for autoregressive next-token prediction if seq_len > 1 and targets match input_ids
            # If caller already shifted, calculate directly
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = (
                input_ids
                if input_ids.size(1) <= self.config.max_seq_len
                else input_ids[:, -self.config.max_seq_len :]
            )
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]  # last token logits

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat((input_ids, idx_next), dim=1)
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

        return input_ids
