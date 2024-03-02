from functools import partial

import jax
import time
import jax.numpy as jnp

from model import LM
from transformers import AutoTokenizer


class Sampler:
    def __init__(
        self,
        model: LM,
        tokenizer: AutoTokenizer,
        shard: jax.sharding.PositionalSharding,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.shard = shard

    @partial(jax.jit, static_argnums=(0,))
    def get_logits(self, params: dict, input_ids: jnp.ndarray):
        return self.model.apply(params, input_ids)

    def sample(
        self, params: dict, prompt: str, max_length: int = 100, temperature: float = 1.0
    ):
        input_ids = self.tokenizer.encode(prompt, return_tensors="jax")
        num_cores = 8

        if input_ids.shape[0] % num_cores != 0:
            padding_size = num_cores - (input_ids.shape[0] % num_cores)
            padding = jnp.zeros(
                (padding_size,) + input_ids.shape[1:], dtype=input_ids.dtype
            )
            input_ids = jnp.concatenate([input_ids, padding], axis=0)

        input_ids = jax.device_put(input_ids, self.shard)
        generated = input_ids
        key = jax.random.PRNGKey(int(time.time() * 1000) % (2**32))

        for _ in range(max_length):
            outputs = self.get_logits(params, generated)
            next_token_logits = outputs[:, -1, :] / temperature
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_token_logits, axis=-1)
            generated = jnp.concatenate([generated, next_token[:, None]], axis=-1)

        generated = generated[0].tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return text
