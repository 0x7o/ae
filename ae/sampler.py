import jax
import time
import jax.numpy as jnp
from jax.sharding import PositionalSharding

from model import LM
from transformers import AutoTokenizer


class Sampler:
    def __init__(
        self,
        model: LM,
        tokenizer: AutoTokenizer,
        shard: PositionalSharding
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.shard = shard

    def sample(
        self, params: dict, prompt: str, max_length: int = 100, temperature: float = 1.0
    ):
        input_ids = self.tokenizer.encode(prompt, return_tensors="jax")
        input_ids = jax.device_put(input_ids, self.shard)

        generated = input_ids
        key = jax.random.PRNGKey(int(time.time() * 1000) % (2**32))

        for _ in range(max_length):
            outputs = self.model.apply(params, generated)
            next_token_logits = outputs[:, -1, :] / temperature
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_token_logits, axis=-1)
            generated = jnp.concatenate([generated, next_token[:, None]], axis=-1)

        generated = generated[0].tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return text
