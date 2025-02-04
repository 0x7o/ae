import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.partitioning import param_with_axes  # для аннотаций параметров
from einops import rearrange, einsum, repeat
from typing import Optional

# ----------------- Вспомогательные функции позиционного кодирования -----------------
def create_mask(seq_len, dtype: jnp.dtype = jnp.float32):
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=dtype))

def fixed_pos_embedding(inv_freq, seq, dtype: jnp.dtype = jnp.float32):
    sinusoid_inp = jnp.einsum("i,j->ij", jnp.arange(seq, dtype=dtype), inv_freq)
    sinusoid_inp = repeat(sinusoid_inp, "... d -> ... (d r)", r=2)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)

def rotate_every_two(x, dtype: jnp.dtype = jnp.float32):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack((-x2, x1), axis=-1, dtype=dtype)
    return rearrange(x, "... d r -> ... (d r)")

def apply_rotary_pos_emb(x, sincos, dtype: jnp.dtype = jnp.float32):
    sin, cos = sincos
    return x * cos + rotate_every_two(x, dtype) * sin

# ----------------- RMSNorm -----------------
class RMSNorm(nn.Module):
    eps: float = 1e-8
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        # Вызываем param_with_axes напрямую, а не через self.
        scale = param_with_axes(
            "scale", nn.initializers.ones, (x.shape[-1],), self.dtype, axes=("embed",)
        )
        return x / rms * scale

# ----------------- SwiGLU FeedForward -----------------
class FeedForward(nn.Module):
    d_model: int
    ff_mult: int = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        hidden_dim = self.d_model * self.ff_mult

        def create_dense(features, name):
            return nn.Dense(
                features,
                dtype=self.dtype,
                name=name,
                kernel_init=nn.initializers.xavier_uniform(),
                param_axes={"kernel": ("embed", "ff"), "bias": ("ff",)},
                kernel_axes=("embed", "ff"),
                bias_axes=("ff",)
            )

        x_proj = create_dense(hidden_dim * 2, "wi")(x)
        x1, x2 = jnp.split(x_proj, 2, axis=-1)
        x_swiglu = create_dense(self.d_model, "wo")(jax.nn.silu(x2) * x1)
        return x_swiglu

# ----------------- Multihead GQA -----------------
class MultiheadGQA(nn.Module):
    d_model: int
    query_heads: int
    kv_heads: int
    dropout: float = 0.0
    layer_norm: bool = True
    layer_norm_eps: float = 1e-5
    gamma_init: float = 1.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, is_causal: bool = False, mask: Optional[jnp.ndarray] = None,
                 need_weights: bool = False, average_attn_weights: bool = False):
        b, n, _ = x.shape
        head_dim = self.d_model // self.query_heads
        kv_embed_dim = head_dim * self.kv_heads

        def create_dense(features, name):
            return nn.Dense(
                features,
                dtype=self.dtype,
                name=name,
                kernel_init=nn.initializers.xavier_uniform(),
                # Annotate kernel with both model and tensor axes
                param_axes={"kernel": ("embed", "kv"), "bias": ("kv",)},
                kernel_axes=("embed", "kv"),
                # For bias, use only one axis
                bias_axes=("kv",)
            )

        # Use the modified Dense creation for projections
        q = create_dense(self.d_model, "q_proj")(x)
        k = create_dense(kv_embed_dim, "k_proj")(x)
        v = create_dense(kv_embed_dim, "v_proj")(x)

        num_head_groups = self.query_heads // self.kv_heads
        q_grouped = rearrange(q, "b n (g h) d -> b g h n d", g=num_head_groups)
        scale = head_dim ** 0.5
        q_grouped = q_grouped / scale

        # Для различения осей длины последовательности: nq для запросов, nk для ключей.
        attn_logits = einsum(q_grouped, k, "b g h nq d, b nk h d -> b g h nq nk")
        if is_causal:
            causal_mask = jnp.tril(jnp.ones((n, n), dtype=bool))
            attn_logits = jnp.where(causal_mask[None, None, None, :, :], attn_logits, -1e10)
        if mask is not None:
            if mask.ndim == 2:
                mask_expanded = mask[None, None, None, :, :]
            elif mask.ndim == 3:
                mask_expanded = mask[:, None, None, :, :]
            else:
                raise ValueError(f"Unsupported mask ndim: {mask.ndim}")
            attn_logits = jnp.where(mask_expanded, attn_logits, -1e10)

        attn = nn.softmax(attn_logits, axis=-1)
        out_grouped = einsum(attn, v, "b g h nq nk, b nk h d -> b g h nq d")
        out = rearrange(out_grouped, "b g h nq d -> b nq (g h d)")
        if self.layer_norm:
            out = nn.LayerNorm(epsilon=self.layer_norm_eps, dtype=self.dtype, name="layer_norm")(out)
        out = nn.Dense(
            self.d_model,
            dtype=self.dtype,
            name="out_proj",
            kernel_init=nn.initializers.xavier_uniform(),
        )(out)
        attn_weights = None
        if need_weights:
            attn_weights = rearrange(attn, "b g h nq nk -> b nq nk (g h)")
            if average_attn_weights:
                attn_weights = jnp.mean(attn_weights, axis=1)
        return out, attn_weights

# ----------------- Позиционное кодирование -----------------
class PositionEncoding(nn.Module):
    d_model: int
    dtype: jnp.dtype = jnp.float32
    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[1]
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.d_model, 2, dtype=self.dtype) / self.d_model))
        pos_enc = fixed_pos_embedding(inv_freq, seq_len, dtype=self.dtype)
        return apply_rotary_pos_emb(x, pos_enc, dtype=self.dtype)

# ----------------- Transformer Block -----------------
class Block(nn.Module):
    d_model: int
    seq_len: int
    query_heads: int
    kv_heads: int
    dtype: jnp.dtype = jnp.float32
    @nn.compact
    def __call__(self, x, mask=None, is_causal: bool = False):
        attn_module = MultiheadGQA(
            d_model=self.d_model,
            query_heads=self.query_heads,
            kv_heads=self.kv_heads,
            dropout=0.0,
            layer_norm=True,
            dtype=self.dtype,
        )
        x_attn, _ = attn_module(x, is_causal=is_causal, mask=mask)
        x_norm = RMSNorm(dtype=self.dtype, name="rmsnorm")(x_attn)
        ff = FeedForward(d_model=self.d_model, dtype=self.dtype, name="ff")
        x_ff = ff(x_norm)
        return x_ff

# ----------------- Языковая модель (LM) -----------------
class LM(nn.Module):
    d_model: int
    n_layers: int
    query_heads: int
    kv_heads: int
    vocab_size: int
    seq_len: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, is_causal: bool = False):
        mask = create_mask(x.shape[-1], dtype=self.dtype)

        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            param_axes={"embedding": ("vocab", "embed")},
            embedding_axes=("vocab", "embed"),
            name="embed",
        )

        x = embed(x)
        x = PositionEncoding(d_model=self.d_model, dtype=self.dtype, name="p_e")(x)

        for i in range(self.n_layers):
            x = Block(
                d_model=self.d_model,
                seq_len=self.seq_len,
                query_heads=self.query_heads,
                kv_heads=self.kv_heads,
                dtype=self.dtype,
                name=f"block_{i}",
            )(x, mask=mask, is_causal=is_causal)

        output_dense = nn.Dense(
            self.vocab_size,
            dtype=self.dtype,
            name="out",
            kernel_init=nn.initializers.xavier_uniform(),
            param_axes={"kernel": ("embed", "vocab"), "bias": ("vocab",)},
            kernel_axes=("embed", "vocab"),
            bias_axes=("vocab",)
        )

        return output_dense(x)

# ----------------- Пример инициализации модели -----------------
if __name__ == "__main__":
    import jax.random
    model = LM(
        d_model=512,
        n_layers=6,
        query_heads=8,
        kv_heads=2,
        vocab_size=1000,
        seq_len=512,
    )
    dummy_input = jnp.ones((1, 512), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), dummy_input)
    print(params)
