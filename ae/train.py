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
from jax.experimental.pjit import pjit
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec, Mesh

from datasets import load_dataset
from transformers import AutoTokenizer

from model import LM
from tqdm import tqdm
from sampler import Sampler
from argparse import ArgumentParser


# TODO: restoring from checkpoint


class Trainer:
    def __init__(self, config):
        self.config = config
        self.devices = jax.devices()

        if self.config["train"]["dtype"] == "float32":
            self.dtype = jnp.float32
        elif self.config["train"]["dtype"] == "bfloat16":
            self.dtype = jnp.bfloat16
        else:
            raise ValueError("Invalid dtype")

        self.devices = mesh_utils.create_device_mesh((2, 4))
        print("Devices: ", self.devices)
        self.model = self.init_model(**config["model"])
        self.mesh = Mesh(devices=self.devices, axis_names=("vocab", "embed"))

        def init_fn(rng, x):
            return self.model.init(rng, x)

        init_fn_pjit = jax.experimental.maps.xmap(
            init_fn,
            in_axes=(["params"], ["data", ...]),
            out_axes=["params", "data", ...],
            axis_resources={
                "params": PartitionSpec("vocab", "embed"),
                "data": PartitionSpec(None),
            },
        )

        self.params = init_fn_pjit(
            jax.random.PRNGKey(0),
            jnp.ones((1, config["model"]["seq_len"]), dtype=jnp.int32),
        )

        self.data_sharding = NamedSharding(self.mesh, PartitionSpec("data", None))
        print(f"Model {config['model']} initialized.")
        print("Device assignment:")
        print(jax.tree_map(lambda x: print(x.device_buffer.device()), self.params))
        print(
            f"{round(sum(p.size for p in jax.tree_util.tree_flatten(self.params)[0]) / 1_000_000, 2)}M parameters"
        )
        self.dataset = load_dataset(config["data"]["name"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer"])
        self.sampler = Sampler(self.model, self.tokenizer, self.data_sharding)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.optim = None

    def init_model(self, d_model, n_heads, n_layers, vocab_size, seq_len):
        model = LM(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            vocab_size=vocab_size,
            seq_len=seq_len,
            dtype=self.dtype,
        )
        return model

    def cross_entropy(self, logits, targets, axis=-1):
        logprobs = nn.log_softmax(logits, axis=axis)
        nll = jnp.take_along_axis(
            logprobs, jnp.expand_dims(targets, axis=axis), axis=axis
        )
        cross_entropy = -jnp.mean(nll, dtype=self.dtype)
        return cross_entropy

    def train_step(self, params, optim_state, inputs, targets):
        def loss_fn(params):
            logits = self.model.apply(params, inputs)
            return self.cross_entropy(logits, targets, axis=-1)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jax.lax.pmean(grads, axis_name="model")
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
        total_steps = (
            self.config["train"]["n_epochs"]
            * len(tokenized_datasets[self.config["data"]["split"]])
            // self.config["train"]["batch_size"]
        )

        if self.config["train"].get("scheduler"):
            if self.config["train"]["scheduler"]["type"] == "cosine":
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

        train_step_pjit = pjit(
            self.train_step,
            in_shardings=(
                None,
                PartitionSpec("vocab", "embed"),
                None,
                PartitionSpec("data", None),
                PartitionSpec("data", None),
            ),
            out_shardings=(
                PartitionSpec("vocab", "embed"),
                None,
                None,
            ),
            static_argnums=(0, ),
        )

        def data_loader(dataset, batch_size, seq_len):
            while True:
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
        bar = tqdm(
            data_loader(
                tokenized_datasets["train"],
                batch_size,
                self.config["model"]["seq_len"],
            ),
            total=total_steps,
            desc="Training...",
        )

        for i, batch in enumerate(bar):
            batch = batch.reshape(-1, self.config["model"]["seq_len"])
            inputs, targets = batch[:, :-1], batch[:, 1:]
            inputs = jax.device_put(inputs, self.data_sharding)
            targets = jax.device_put(targets, self.data_sharding)
            self.params, loss, optim_state = train_step_pjit(
                self.params, optim_state, inputs, targets
            )
            bar.set_postfix({"loss": loss}, refresh=False)
            step += 1

            if step % save_checkpoint_steps == 0:
                print(f"\n*** Saving checkpoint at step {step} ***")
                checkpoint = {
                    "params": jax.tree_map(
                        lambda x: jnp.asarray(x, dtype=jnp.int32), self.params
                    )
                }
                os.makedirs(output_dir, exist_ok=True)

                with open(os.path.join(output_dir, f"checkpoint_{step}.pt"), "wb") as f:
                    pickle.dump(checkpoint, f)

                with open(os.path.join(output_dir, "config.json"), "w") as f:
                    json.dump(self.config["model"], f, indent=4)

                print(f"Checkpoint saved to {output_dir}")

            if self.config["train"].get("generate"):
                if step % self.config["train"]["generate"]["steps"] == 0:
                    print(f"\n*** Generating text at step {step} ***")
                    texts = [
                        [
                            self.sampler.sample(
                                self.params,
                                prompt=prompt,
                                max_length=self.config["train"]["generate"][
                                    "max_length"
                                ],
                                temperature=self.config["train"]["generate"][
                                    "temperature"
                                ],
                            ).replace("\n", "\\n")
                        ]
                        for prompt in self.config["train"]["generate"]["prompts"]
                    ]
                    table = wandb.Table(data=texts, columns=["text"])
                    run.log({"generated_text": table}, step=step)
                    print("Text generated.")

            if scheduler:
                run.log(
                    {
                        "loss": loss,
                        "lr": scheduler(step),
                        "epoch": i / (total_steps / n_epochs),
                    },
                    step=step,
                )
            else:
                run.log(
                    {"loss": loss, "epoch": i / (total_steps / n_epochs)}, step=step
                )


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
