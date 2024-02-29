import jax
import jax.numpy as jnp

from model import LM
from transformers import AutoTokenizer


class Sampler:
    def __init__(
            self,
            model: LM,
            params: dict,
            tokenizer: AutoTokenizer,
            devices: list[jax.Device],
    ):
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.devices = devices

    def sample(self, prompt: str, max_length: int = 100, temperature: float = 1.0):
        input_ids = self.tokenizer.encode(prompt, return_tensors="jax")
        input_ids = jax.device_put(input_ids, self.devices[0])

        generated = input_ids
        key = jax.random.PRNGKey(0)

        for _ in range(max_length):
            outputs = self.model.apply(self.params, generated)
            next_token_logits = outputs[:, -1, :] / temperature
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_token_logits, axis=-1)
            generated = jnp.concatenate([generated, next_token[:, None]], axis=-1)

        generated = generated[0].tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return text
