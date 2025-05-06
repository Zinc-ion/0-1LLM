# -*- coding: utf-8 -*-
# @File:model.py
# @Author:Zinc-ion
# @Date:2025-04-06
# @IDE:PyCharm
from Config import LLMConfig
from typing import List,Tuple,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, pos_cis: torch.Tensor):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 <= ndim
        assert pos_cis.shape  == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)

    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_pos_cis(dim: int, end: int = int(32*1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t = torch.arange(end, device=freqs.device).float()

    freqs = torch.outer(t, freqs).float()

    pos_cis = torch.polar(torch.ones_like(freqs), freqs)
    return pos_cis


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Args:
        x: (bsz, seq_len, n_heads, head_dim)
        n_rep: int
    Returns:
        (bsz, seq_len, n_heads * n_rep, head_dim)
    """
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bsz, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)
    )



class Attention(nn.Module):
    def __init__(self,args: LLMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // self.n_heads

        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False) # bias=False表示不使用偏置项
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim,bias=False)  # bias=False表示不使用偏置项
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # bias=False表示不使用偏置项

        self.wo = nn.Linear(args.dim, args.dim, bias=False)  # bias=False表示不使用偏置项
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1)

        self.register_buffer('mask', mask)  # Register the mask as a buffer, so it won't be considered a model parameter

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        bsz, seq_len, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1,2),
            repeat_kv(xk, self.n_rep).transpose(1,2),
            repeat_kv(xv, self.n_rep).transpose(1,2),
        )
        scores = (xq @ xk.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores += self.mask[:, :, :seq_len, :seq_len]

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # dim = -1 表示最后一个维度
        scores = self.attn_dropout(scores)

        output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x: torch.Tensor):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)) )

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.variance_epsilon)
        return x * self.weight / norm


class SpongeBobBlock(nn.Module):
    def __init__(self,layer_id:int, config: LLMConfig):
        super().__init__()
        self.n_head = config.n_heads
        self.dim = config.dim
        self.head_dim = self.dim // self.n_head
        self.attention = Attention(config)
        self.attention_norm = RMSNorm(self.dim, eps=config.norm_eps)
        self.feed_forwrd = FeedForward(config)
        self.ffn_norm = RMSNorm(self.dim, eps=config.norm_eps)

    def forward(self,
                x,
                pos_cis,
                past_key_value=None,
                use_cache=False,):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        h = x + h_attn

        out = h + self.feed_forwrd(
            self.ffn_norm(h)
        )
        return out, past_kv

class SpongeBobModel(PreTrainedModel):
    config_class = LLMConfig
    base_model_prefix = "spongebob"

    def __init__(self, params: LLMConfig):
        self.params = params or LLMConfig()

        super().__init__(self.params)
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)

        self.layers = nn.ModuleList([SpongeBobBlock(l, params) for l in range(self.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.tok_embeddings.weight = self.output.weight

        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)
        self.OUT = CausalLMOutputWithPast() # 用于存储模型的输出

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args,
                ):
        past_key_value = past_key_value or [None] * len(self.layers)

        start_pos = args.get('start_pos', 0)

        h = self.dropout(self.tok_embeddings(input_ids))

        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []

        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_value[l],
                use_cache=use_cache,
            )
            past_kvs.append(past_kv)

        logits = self.output(self.norm(h))
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('past_key_values', past_kvs)

        @torch.inference_mode()
        def generate(self,
                     input_ids,
                     eos_token_id=2,
                     max_new_tokens=1024,
                     temperature=0.75,
                     top_p=0.90,
                     stream=False,
                     rp=1,
                     use_cache=True,
                     pad_token_id=0,
                     **args):
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
            start, first_seq, past_kvs = input_ids.shape[1], True, None

            while input_ids.shape[1] < max_new_tokens -1 :
                if first_seq or not use_cache:
                    out, first_seq = self(input_ids, past_key_value=past_kvs, use_cache=use_cache, **args), False

                else:
                    out = self(input_ids[:, -1:], past_key_value=past_kvs, use_cache=use_cache, start_pos=input_ids.shape[1] - 1, **args)

                logits, past_kvs = out.logits[:, -1, :], out.past_key_values

                logits[:, list(set(input_ids.tolist()[0]))] /= rp
                logits /= (temperature + 1e-9)

                if top_p is not None and top_p < 1.0 :
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    # 高级索引操作：scatter将sorted_indices_to_remove映射回原始索引
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    # 无穷小赋值：-float('Inf') 使被移除token的概率趋近于0
                    logits[indices_to_remove] = -float('Inf')

                input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                # 将新token拼接到已有序列上
                input_ids = torch.cat((input_ids, input_ids_next), dim=1)
                # 生成器返回新生成部分
                yield input_ids[:, start:]
                # 若生成的token为结束符，则停止生成
                if input_ids_next.item() == eos_token_id:
                    break