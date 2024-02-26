import os
import json

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


def init_model(d_model, d_ff, n_heads, n_layers, vocab_size, seq_len):
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
    return model, params


def cross_entropy(logits, targets, axis=-1):
    logprobs = nn.log_softmax(logits, axis=axis)
    nll = jnp.take_along_axis(logprobs, jnp.expand_dims(targets, axis=axis), axis=axis)
    cross_entropy = -jnp.mean(nll)
    return cross_entropy


def loss_fn(model, params, batch):
    inp, labels = batch[:, :-1], batch[:, 1:]
    logits = model.apply(params, inp)
    return cross_entropy(logits, labels, axis=-1)


def train_step(model, params, optim, optim_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(model, params, batch)
    updates, optim_state = optim.update(grads, optim_state)
    params = optax.apply_updates(params, updates)
    return loss, params, optim_state


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to the config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    model, params = init_model(**config["model"])

    dataset = load_dataset(config["data"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer"])

    def tokenize_function(examples):
        return tokenizer(
            examples[config["data"]["text_column"]],
            padding="max_length",
            truncation=True,
            max_length=config["model"]["seq_len"],
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    if config["train"]["optimizer"]["type"] == "adam":
        optim = optax.adam(**config["train"]["optimizer"]["params"])
    else:
        raise ValueError("Invalid optimizer type")

    optim_state = optim.init(params)
    batch_size = config["train"]["batch_size"]
    n_epochs = config["train"]["n_epochs"]
    output_dir = config["train"]["output_dir"]
    save_checkpoint_steps = config["train"]["save_checkpoint_steps"]

    indices = np.arange(len(tokenized_datasets[config["data"]["split"]]))
    np.random.shuffle(indices)

    def data_loader(dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            yield {k: jnp.array(v) for k, v in dataset[i: i + batch_size].items()}

    step = 0
    for epoch in range(n_epochs):
        for batch in tqdm(
                data_loader(tokenized_datasets["train"], batch_size),
                total=len(tokenized_datasets["train"]) // batch_size,
                desc=f"Epoch {epoch + 1}",
        ):
            loss, params, optim_state = train_step(
                model, params, optim, optim_state, batch
            )
            step += 1

            if step % save_checkpoint_steps == 0:
                checkpoint = {"params": params, "optim_state": optim_state}
                os.makedirs(output_dir, exist_ok=True)

                with open(os.path.join(output_dir, f"checkpoint_{step}.pt"), "wb") as f:
                    pickle.dump(checkpoint, f)

        print(f"Epoch {epoch + 1} completed. Loss: {loss}")

    print("Training completed.")


if __name__ == "__main__":
    main()
