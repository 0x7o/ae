import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange, repeat


def create_mask(seq_len):
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return mask


def fixed_pos_embedding(inv_freq, seq):
    sinusoid_inp = jnp.einsum("i , j -> i j", jnp.arange(seq), inv_freq)
    sinusoid_inp = repeat(sinusoid_inp, "... d -> ... (d r)", r=2)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def rotate_every_two(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    return (x * cos) + (rotate_every_two(x) * sin)


class SelfAttentionHead(nn.Module):
    d_model: int
    seq_len: int

    def setup(self) -> None:
        self.qn = nn.Dense(self.d_model)
        self.kn = nn.Dense(self.d_model)
        self.vn = nn.Dense(self.d_model)

    def __call__(self, v, k, q, mask=None):
        q = self.qn(q)
        k = self.kn(k)
        v = self.vn(v)
        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(self.d_model)
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e10)
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

    def __call__(self, v, k, q, mask=None):
        outputs = [head(v, k, q, mask) for head in self.heads]
        return self.out(jnp.concatenate(outputs, axis=-1))


class PositionEncoding(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[1]
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.d_model, 2) / self.d_model))
        pos_enc = fixed_pos_embedding(inv_freq, seq_len)
        return apply_rotary_pos_emb(x, pos_enc)


class FeedForward(nn.Module):
    d_model: int
    d_ff: int

    def setup(self) -> None:
        self.layer1 = nn.Dense(self.d_ff)
        self.layer2 = nn.Dense(self.d_model)

    def __call__(self, x):
        x = nn.swish(self.layer1(x))
        x = self.layer2(x)
        return x


class Block(nn.Module):
    d_model: int
    d_ff: int
    n_heads: int
    seq_len: int

    def setup(self) -> None:
        self.attention = SelfAttention(
            n_heads=self.n_heads, d_model=self.d_model, seq_len=self.seq_len
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.ff = FeedForward(d_model=self.d_model, d_ff=self.d_ff)

    def __call__(self, x, mask):
        x = x + self.attention(x, x, x, mask)
        x = x + self.ff(x)
        x = self.norm(x)
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
        self.blocks = [
            Block(
                d_model=self.d_model,
                d_ff=self.d_ff,
                n_heads=self.n_heads,
                seq_len=self.seq_len,
            )
            for _ in range(self.n_layers)
        ]
        self.norm = nn.LayerNorm(self.d_model)

    def __call__(self, x):
        mask = create_mask(x.shape[-1])
        x = self.w_e(x)
        x = self.p_e(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return x


if __name__ == "__main__":
    import jax

    # Create a random key
    key = jax.random.PRNGKey(0)

    # Create an example input
    x = jnp.ones((10, 200), dtype=jnp.int32)
    print(x)

    # Initialize the model
    lm = LM(d_model=512, d_ff=2048, n_heads=8, n_layers=6, vocab_size=2, seq_len=200)
    params = lm.init(key, x)

    # Apply the model
    output = lm.apply(params, x)

    print(output)
    print(output.shape)
