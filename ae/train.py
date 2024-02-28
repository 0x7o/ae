import os
import json
import wandb

import jax
import optax
import pickle
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from datasets import load_dataset
from transformers import AutoTokenizer

from model import LM
from tqdm import tqdm
from argparse import ArgumentParser


class Trainer:
    def __init__(self, config):
        self.config = config
        self.devices = jax.devices()
        print("Devices: ", self.devices)
        self.model, self.params = self.init_model(**config["model"])
        print(f"Model {config['model']} initialized.")
        print(f"{sum(p.size for p in jax.tree_flatten(self.params)[0])} parameters")
        self.dataset = load_dataset(config["data"]["name"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer"])

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_model(self, d_model, d_ff, n_heads, n_layers, vocab_size, seq_len):
        key = jax.random.PRNGKey(0)
        model = LM(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            vocab_size=vocab_size,
            seq_len=seq_len,
        )
        x = jnp.ones((1, seq_len), dtype=jnp.int32)
        params = model.init(key, x)
        params = jax.device_put(params, self.devices[0])
        return model, params

    def cross_entropy(self, logits, targets, axis=-1):
        logprobs = nn.log_softmax(logits, axis=axis)
        nll = jnp.take_along_axis(
            logprobs, jnp.expand_dims(targets, axis=axis), axis=axis
        )
        cross_entropy = -jnp.mean(nll)
        return cross_entropy

    def loss_fn(self, params, batch):
        batch = batch.reshape(-1, self.config["model"]["seq_len"])
        inp, labels = batch[:, :-1], batch[:, 1:]
        logits = self.model.apply(params, inp)
        return self.cross_entropy(logits, labels, axis=-1)

    def train_step(self, optim, optim_state, batch):
        loss, grads = jax.value_and_grad(self.loss_fn)(self.params, batch["input_ids"])
        updates, optim_state = optim.update(grads, optim_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss, optim_state
    def train(self):
        def tokenize_function(examples):
            return self.tokenizer(examples[self.config["data"]["text_column"]])

        tokenized_datasets = self.dataset.map(tokenize_function, batched=True)

        if self.config["train"]["optimizer"]["type"] == "adam":
            optim = optax.adam(**self.config["train"]["optimizer"]["params"])
        else:
            raise ValueError("Invalid optimizer type")

        if self.config["train"]["scheduler"]["type"] == "cosine":
            total_steps = (
                self.config["train"]["n_epochs"]
                * len(tokenized_datasets[self.config["data"]["split"]])
                // self.config["train"]["batch_size"]
            )
            warmup_steps = self.config["train"]["scheduler"]["warmup_steps"]
            scheduler = optax.linear_schedule(
                init_value=0.0,
                end_value=self.config["train"]["optimizer"]["params"]["learning_rate"],
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
            eos_token_id = self.tokenizer.eos_token_id
            buffer = []

            for example in dataset:
                buffer.extend(example["input_ids"])

                while len(buffer) >= seq_len:
                    yield {
                        "input_ids": jnp.array(buffer[:seq_len]),
                        "attention_mask": jnp.ones(seq_len),
                    }
                    buffer = buffer[seq_len:]

            if buffer:
                buffer.extend([eos_token_id] * (seq_len - len(buffer)))
                yield {
                    "input_ids": jnp.array(buffer),
                    "attention_mask": jnp.ones(seq_len),
                }

        run = wandb.init(project="ae-dev", config=self.config)
        self.train_step = jax.pmap(self.train_step, axis_name='batch')

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
                batch = jax.device_put(batch, self.devices[0])
                loss, optim_state = self.train_step(optim, optim_state, batch)

                step += 1

                if step % save_checkpoint_steps == 0:
                    checkpoint = {"params": self.params}
                    os.makedirs(output_dir, exist_ok=True)

                    with open(
                        os.path.join(output_dir, f"checkpoint_{step}.pt"), "wb"
                    ) as f:
                        pickle.dump(checkpoint, f)

                run.log(
                    {"loss": loss, "epoch": epoch, "lr": scheduler(step)}, step=step
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
