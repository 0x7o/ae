import os
import json
from functools import partial

import wandb

import jax
import optax
import pickle
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from datasets import load_dataset
from transformers import AutoTokenizer

from model import LM
from tqdm import tqdm
from sampler import Sampler
from argparse import ArgumentParser


# TODO: pjit model parallelism

class Trainer:
    def __init__(self, config):
        self.config = config
        self.devices = jax.devices()
        self.devices = mesh_utils.create_device_mesh((len(jax.devices()), 1))
        print("Devices: ", self.devices)
        self.shard = PositionalSharding(self.devices)
        self.model, self.params = self.init_model(**config["model"])
        print(f"Model {config['model']} initialized.")
        print(
            f"{round(sum(p.size for p in jax.tree_util.tree_flatten(self.params)[0]) / 1_000_000, 2)}M parameters"
        )
        self.dataset = load_dataset(config["data"]["name"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer"])
        self.sampler = Sampler(self.model, self.tokenizer, self.shard)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.optim = None

    def init_model(self, d_model, n_heads, n_layers, vocab_size, seq_len):
        key = jax.random.PRNGKey(0)
        model = LM(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            vocab_size=vocab_size,
            seq_len=seq_len,
        )
        x = jnp.ones((1, seq_len), dtype=jnp.bfloat16)
        params = model.init(key, x)
        return model, params

    def cross_entropy(self, logits, targets, axis=-1):
        logprobs = nn.log_softmax(logits, axis=axis)
        nll = jnp.take_along_axis(
            logprobs, jnp.expand_dims(targets, axis=axis), axis=axis
        )
        cross_entropy = -jnp.mean(nll)
        return cross_entropy

    def loss_fn(self, params, inputs, targets):
        logits = self.model.apply(params, inputs)
        return self.cross_entropy(logits, targets, axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, params, optim_state, inputs, targets):
        inputs = jnp.asarray(inputs, dtype=jnp.bfloat16)
        targets = jnp.asarray(targets, dtype=jnp.bfloat16)
        loss, grads = jax.value_and_grad(self.loss_fn)(params, inputs, targets)
        updates, optim_state = self.optim.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return params, loss, optim_state

    def train(self):
        def tokenize_function(examples):
            return self.tokenizer(examples[self.config["data"]["text_column"]])

        tokenized_datasets = self.dataset.map(tokenize_function, batched=True)

        if self.config["train"]["optimizer"]["type"] == "adam":
            optim = optax.chain(
                optax.clip_by_global_norm(
                    self.config["train"]["optimizer"]["params"]["clip_norm"]
                ),
                optax.adam(
                    self.config["train"]["optimizer"]["params"]["learning_rate"]
                ),
                optax.apply_every(
                    self.config["train"]["optimizer"]["params"]["accum_steps"]
                ),
            )
        elif self.config["train"]["optimizer"]["type"] == "adamw":
            optim = optax.chain(
                optax.clip_by_global_norm(
                    self.config["train"]["optimizer"]["params"]["clip_norm"]
                ),
                optax.adamw(
                    self.config["train"]["optimizer"]["params"]["learning_rate"]
                ),
                optax.apply_every(
                    self.config["train"]["optimizer"]["params"]["accum_steps"]
                ),
            )
        else:
            raise ValueError("Invalid optimizer type")

        scheduler = None

        if self.config["train"].get("scheduler"):
            if self.config["train"]["scheduler"]["type"] == "cosine":
                total_steps = (
                        self.config["train"]["n_epochs"]
                        * len(tokenized_datasets[self.config["data"]["split"]])
                        // self.config["train"]["batch_size"]
                )
                warmup_steps = self.config["train"]["scheduler"]["warmup_steps"]
                scheduler = optax.linear_schedule(
                    init_value=0.0,
                    end_value=self.config["train"]["optimizer"]["params"][
                        "learning_rate"
                    ],
                    transition_steps=warmup_steps,
                )
                scheduler = optax.join_schedules(
                    schedules=[
                        scheduler,
                        optax.cosine_decay_schedule(
                            init_value=self.config["train"]["optimizer"]["params"][
                                "learning_rate"
                            ],
                            decay_steps=total_steps - warmup_steps,
                        ),
                    ],
                    boundaries=[warmup_steps],
                )
                optim = optax.chain(optim, optax.scale_by_schedule(scheduler))
            else:
                raise ValueError("Invalid scheduler type")

        optim_state = optim.init(self.params)
        batch_size = self.config["train"]["batch_size"]
        n_epochs = self.config["train"]["n_epochs"]
        output_dir = self.config["train"]["output_dir"]
        save_checkpoint_steps = self.config["train"]["save_checkpoint_steps"]
        indices = np.arange(len(tokenized_datasets[self.config["data"]["split"]]))
        np.random.shuffle(indices)

        def data_loader(dataset, batch_size, seq_len):
            buffer = []

            for example in dataset:
                example["input_ids"].append(self.tokenizer.eos_token_id)
                buffer.extend(example["input_ids"])

                while len(buffer) >= seq_len * batch_size:
                    batch = []
                    for _ in range(batch_size):
                        batch.append(buffer[:seq_len])
                        buffer = buffer[seq_len:]
                    yield np.array(batch)

            if buffer:
                remaining = len(buffer) // seq_len
                for _ in range(remaining):
                    yield np.array(buffer[:seq_len])
                    buffer = buffer[seq_len:]

        run = wandb.init(project="ae-dev", config=self.config)
        self.optim = optim
        step = 0

        for epoch in range(n_epochs):
            for batch in tqdm(
                    data_loader(
                        tokenized_datasets["train"],
                        batch_size,
                        self.config["model"]["seq_len"],
                    ),
                    total=len(tokenized_datasets["train"]) // batch_size,
                    desc=f"Epoch {epoch + 1}",
            ):
                batch = batch.reshape(-1, self.config["model"]["seq_len"])
                inputs, targets = batch[:, :-1], batch[:, 1:]
                inputs, targets = jax.device_put((inputs, targets), self.shard)
                params, loss, optim_state = self.train_step(
                    self.params, optim_state, inputs, targets
                )
                self.params = params
                step += 1

                if step % save_checkpoint_steps == 0:
                    checkpoint = {"params": jax.tree_map(lambda x: jnp.asarray(x, dtype=jnp.int32), self.params)}
                    os.makedirs(output_dir, exist_ok=True)

                    with open(
                            os.path.join(output_dir, f"checkpoint_{step}.pt"), "wb"
                    ) as f:
                        pickle.dump(checkpoint, f)

                    with open(os.path.join(output_dir, "config.json"), "w") as f:
                        json.dump(self.config["model"], f, indent=4)

                if self.config["train"].get("generate"):
                    if step % self.config["train"]["generate"]["steps"] == 0:
                        texts = [
                            [
                                self.sampler.sample(
                                    self.params,
                                    prompt=prompt,
                                    max_length=self.config["train"]["generate"][
                                        "max_length"
                                    ],
                                )
                            ]
                            for prompt in self.config["train"]["generate"]["prompts"]
                        ]
                        table = wandb.Table(data=texts, columns=["text"])
                        run.log({"generated_text": table}, step=step)

                if scheduler:
                    run.log(
                        {"loss": loss, "epoch": epoch, "lr": scheduler(step)}, step=step
                    )
                else:
                    run.log({"loss": loss, "epoch": epoch}, step=step)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to the config file", required=True
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    trainer = Trainer(config)
    trainer.train()
