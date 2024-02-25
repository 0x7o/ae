import jax.numpy as jnp
import flax.linen as nn


def create_mask(seq_len):
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return mask


class SelfAttentionHead(nn.Module):
    d_model: int
    seq_len: int

    def setup(self) -> None:
        self.qn = nn.Dense(self.d_model)
        self.kn = nn.Dense(self.d_model)
        self.vn = nn.Dense(self.d_model)
        self.mask = create_mask(self.seq_len)

    def __call__(self, v, k, q):
        q = self.qn(q)
        k = self.kn(k)
        v = self.vn(v)
        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(self.d_model)
        attn_weights *= self.mask
        return q + (nn.softmax(attn_weights) @ v)


class SelfAttention(nn.Module):
    n_heads: int
    d_model: int
    seq_len: int

    def setup(self) -> None:
        self.heads = [
            SelfAttentionHead(d_model=self.d_model, seq_len=self.seq_len)
            for _ in range(self.n_heads)
        ]
        self.out = nn.Dense(self.d_model)

    def __call__(self, v, k, q):
        outputs = [head(v, k, q) for head in self.heads]
        return self.out(jnp.concatenate(outputs, axis=-1))


class PositionEncoding(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[0]
        pos = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * -(jnp.log(10000.0) / self.d_model)
        )
        pos_enc = jnp.zeros((seq_len, self.d_model))
        pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(pos * div_term))
        pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(pos * div_term))
        return x + pos_enc


class FeedForward(nn.Module):
    d_model: int
    d_ff: int
    n_layers: int

    def setup(self) -> None:
        self.layers = [nn.Dense(self.d_ff) for _ in range(self.n_layers)]
        self.final_layer = nn.Dense(self.d_model)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.gelu(layer(x))
        x = self.final_layer(x)
        return x


class LM(nn.Module):
    d_model: int
    d_ff: int
    n_heads: int
    n_layers: int
    vocab_size: int
    seq_len: int

    def setup(self) -> None:
        self.w_e = nn.Embed(self.vocab_size, self.d_model)
        self.p_e = PositionEncoding(d_model=self.d_model)
        self.attention = SelfAttention(
            n_heads=self.n_heads, d_model=self.d_model, seq_len=self.seq_len
        )
        self.norm1 = nn.LayerNorm(self.d_model)
        self.ff = FeedForward(
            d_model=self.d_model, d_ff=self.d_ff, n_layers=self.n_layers
        )
        self.norm2 = nn.LayerNorm(self.d_model)
        self.linear = nn.Dense(self.vocab_size)

    def __call__(self, x):
        x = self.w_e(x)
        x = self.p_e(x)
        x = x + self.attention(x, x, x)
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        x = self.linear(x)
        x = nn.softmax(x)
        return x
