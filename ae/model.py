import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange, repeat


def create_mask(seq_len, dtype: jnp.dtype = jnp.float32):
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=dtype))
    return mask


def fixed_pos_embedding(inv_freq, seq, dtype: jnp.dtype = jnp.float32):
    sinusoid_inp = jnp.einsum("i , j -> i j", jnp.arange(seq, dtype=dtype), inv_freq)
    sinusoid_inp = repeat(sinusoid_inp, "... d -> ... (d r)", r=2)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def rotate_every_two(x, dtype: jnp.dtype = jnp.float32):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack((-x2, x1), axis=-1, dtype=dtype)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(x, sincos, dtype: jnp.dtype = jnp.float32):
    sin, cos = sincos
    return (x * cos) + (rotate_every_two(x, dtype) * sin)


class LayerNorm(nn.Module):
    eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True, dtype=self.dtype)
        mean_of_squares = jnp.mean(
            jnp.square(x), axis=-1, keepdims=True, dtype=self.dtype
        )
        variance = mean_of_squares - jnp.square(mean)
        inv = 1.0 / jnp.sqrt(variance + self.eps)
        return inv * (x - mean)


class SelfAttentionHead(nn.Module):
    def __call__(self, v, k, q, wi, mask=None):
        q = q @ wi
        k = k @ wi
        v = v @ wi
        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(q.shape[-1])
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e10)
        return q + (nn.softmax(attn_weights) @ v)


class SelfAttention(nn.Module):
    n_heads: int
    d_model: int
    seq_len: int
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.heads = [SelfAttentionHead() for _ in range(self.n_heads)]
        self.out = nn.Dense(self.d_model, dtype=self.dtype)

    def __call__(self, v, k, q, wi, mask=None):
        outputs = [head(v, k, q, wi, mask) for head in self.heads]
        return self.out(jnp.concatenate(outputs, axis=-1, dtype=self.dtype))


class PositionEncoding(nn.Module):
    d_model: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[1]
        inv_freq = 1.0 / (
            10000 ** (jnp.arange(0, self.d_model, 2, dtype=self.dtype) / self.d_model)
        )
        pos_enc = fixed_pos_embedding(inv_freq, seq_len, dtype=self.dtype)
        return apply_rotary_pos_emb(x, pos_enc, dtype=self.dtype)


class FeedForward(nn.Module):
    def __call__(self, x, wi):
        x = nn.swish(x @ wi)
        x = x @ wi
        return x


class Block(nn.Module):
    d_model: int
    n_heads: int
    seq_len: int
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.attention = SelfAttention(
            n_heads=self.n_heads,
            d_model=self.d_model,
            seq_len=self.seq_len,
            dtype=self.dtype,
        )
        self.norm = LayerNorm(dtype=self.dtype)
        self.wi = self.param(
            "wi", nn.initializers.xavier_uniform(), (self.d_model, self.d_model)
        )
        self.ff = FeedForward()

    def __call__(self, x, mask=None):
        x = self.attention(x, x, x, self.wi, mask)
        x = self.norm(x)
        x = self.ff(x, self.wi)
        return x


class LM(nn.Module):
    d_model: int
    n_heads: int
    n_layers: int
    vocab_size: int
    seq_len: int
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.w_e = nn.Embed(self.vocab_size, self.d_model, dtype=self.dtype)
        self.p_e = PositionEncoding(d_model=self.d_model, dtype=self.dtype)
        self.blocks = [
            Block(
                d_model=self.d_model,
                n_heads=self.n_heads,
                seq_len=self.seq_len,
                dtype=self.dtype,
            )
            for _ in range(self.n_layers)
        ]
        self.out = nn.Dense(self.vocab_size, dtype=self.dtype)

    def __call__(self, x):
        mask = create_mask(x.shape[-1], dtype=self.dtype)
        x = self.w_e(x)
        x = self.p_e(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.out(x)
        return x
