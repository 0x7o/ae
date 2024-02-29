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
        jax.device_put(input_ids, self.devices[0])

        for _ in range(max_length):
            logits = self.model.apply(self.params, input_ids)
            logits = jnp.squeeze(logits, axis=0)
            logits = logits[-1, :] / temperature
            probs = jax.nn.log_softmax(logits)
            next_token = jax.random.categorical(jax.random.PRNGKey(0), probs, axis=-1)
            next_token = jnp.expand_dims(next_token, axis=0)
            input_ids = jnp.concatenate([input_ids, next_token[None, :]], axis=1)

        return self.tokenizer.decode(input_ids[0])
