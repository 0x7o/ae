import os
import json
import pickle
import numpy as np

import wandb
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from jax.experimental.pjit import pjit
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec, Mesh
from jax.sharding import PartitionSpec as P

from datasets import load_dataset
from transformers import AutoTokenizer

from sharding import DEFAULT_RULES, with_sharding_constraint, get_sharding_from_rules
from model import LM  # импорт модели с аннотациями
from tqdm import tqdm
from sampler import Sampler  # ваша функция сэмплирования
from argparse import ArgumentParser

class Trainer:
    def __init__(self, config):
        self.config = config
        self.dtype = getattr(jnp, config["train"]["dtype"])

        # Создаем mesh для распределенных вычислений
        devices = mesh_utils.create_device_mesh((2, 2, 2))
        self.mesh = Mesh(devices, ("data", "model", "tensor"))

        # Инициализируем модель
        self.model = self.init_model(**config["model"])

        # Создаем dummy input и получаем структуру параметров
        dummy_input = jnp.ones((2, config["model"]["seq_len"]), dtype=jnp.int32)
        variables = self.model.init(jax.random.PRNGKey(0), dummy_input)

        # Получаем шардинг для всего дерева параметров
        params_sharding = get_sharding_from_rules(variables['params'], DEFAULT_RULES)

        def init_fn(rng, x):
            return self.model.init(rng, x)

        with self.mesh:
            # Применяем pjit с корректным шардингом
            init_fn = pjit(
                init_fn,
                in_shardings=(None, P("data", None)),
                out_shardings={"params": params_sharding}
            )
            self.params = init_fn(jax.random.PRNGKey(0), dummy_input)

        print("Model initialized with sharding rules")
        self.data_sharding = NamedSharding(self.mesh, PartitionSpec("data", None))
        self.dataset = load_dataset(config["data"]["name"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer"])
        self.sampler = Sampler(self.model, self.tokenizer, self.data_sharding)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.optim = None

    def init_model(self, d_model, query_heads, kv_heads, n_layers, vocab_size, seq_len):
        model = LM(
            d_model=d_model,
            n_layers=n_layers,
            vocab_size=vocab_size,
            seq_len=seq_len,
            dtype=self.dtype,
            query_heads=query_heads,
            kv_heads=kv_heads,
        )
        return model

    def cross_entropy(self, logits, targets, axis=-1):
        logprobs = nn.log_softmax(logits, axis=axis)
        nll = jnp.take_along_axis(logprobs, jnp.expand_dims(targets, axis=axis), axis=axis)
        return -jnp.mean(nll, dtype=self.dtype)

    def train_step(self, params, optim_state, inputs, targets):
        def loss_fn(params):
            inputs_sharded = with_sharding_constraint(
                inputs, self.mesh, ("data", None)
            )
            logits = self.model.apply({"params": params}, inputs_sharded)
            logits = with_sharding_constraint(
                logits, self.mesh, ("data", None, "model")
            )
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
                optax.clip_by_global_norm(self.config["train"]["optimizer"]["params"]["clip_norm"]),
                optax.adam(self.config["train"]["optimizer"]["params"]["learning_rate"]),
                optax.apply_every(self.config["train"]["optimizer"]["params"]["accum_steps"]),
            )
        elif self.config["train"]["optimizer"]["type"] == "adamw":
            optim = optax.chain(
                optax.clip_by_global_norm(self.config["train"]["optimizer"]["params"]["clip_norm"]),
                optax.adamw(self.config["train"]["optimizer"]["params"]["learning_rate"]),
                optax.apply_every(self.config["train"]["optimizer"]["params"]["accum_steps"]),
            )
        else:
            raise ValueError("Invalid optimizer type")
        scheduler = None
        total_steps = (self.config["train"]["n_epochs"] *
                       len(tokenized_datasets[self.config["data"]["split"]]) //
                       self.config["train"]["batch_size"])
        if self.config["train"].get("scheduler"):
            if self.config["train"]["scheduler"]["type"] == "cosine":
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
                            init_value=self.config["train"]["optimizer"]["params"]["learning_rate"],
                            decay_steps=total_steps - warmup_steps,
                        ),
                    ],
                    boundaries=[warmup_steps],
                )
                optim = optax.chain(optim, optax.scale_by_schedule(scheduler))
            else:
                raise ValueError("Invalid scheduler type")
        self.optim = optim
        optim_state = optim.init(self.params)
        batch_size = self.config["train"]["batch_size"]
        n_epochs = self.config["train"]["n_epochs"]
        output_dir = self.config["train"]["output_dir"]
        save_checkpoint_steps = self.config["train"]["save_checkpoint_steps"]
        indices = np.arange(len(tokenized_datasets[self.config["data"]["split"]]))
        np.random.shuffle(indices)
        train_step_pjit = pjit(
            self.train_step,
            in_shardings=(None, PartitionSpec("model", "tensor"),
                          None, PartitionSpec("data", None), PartitionSpec("data", None)),
            out_shardings=(PartitionSpec("model", "tensor"), None, None),
            static_argnums=(0,),
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
        step = 0
        bar = tqdm(data_loader(tokenized_datasets["train"], batch_size, self.config["model"]["seq_len"]),
                   total=total_steps, desc="Training...")
        for i, batch in enumerate(bar):
            batch = batch.reshape(-1, self.config["model"]["seq_len"])
            inputs, targets = batch[:, :-1], batch[:, 1:]
            inputs = jax.device_put(inputs, self.data_sharding)
            targets = jax.device_put(targets, self.data_sharding)
            with self.mesh:
                self.params, loss, optim_state = train_step_pjit(self.params, optim_state, inputs, targets)
            bar.set_postfix({"loss": loss}, refresh=False)
            step += 1
            if step % save_checkpoint_steps == 0:
                print(f"\n*** Saving checkpoint at step {step} ***")
                checkpoint = {"params": jax.tree_map(lambda x: jnp.asarray(x, dtype=jnp.int32), self.params)}
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f"checkpoint_{step}.pt"), "wb") as f:
                    pickle.dump(checkpoint, f)
                with open(os.path.join(output_dir, "config.json"), "w") as f:
                    json.dump(self.config["model"], f, indent=4)
                print(f"Checkpoint saved to {output_dir}")
            if self.config["train"].get("generate"):
                if step % self.config["train"]["generate"]["steps"] == 0:
                    print(f"\n*** Generating text at step {step} ***")
                    texts = [[self.sampler.sample(
                        self.params,
                        prompt=prompt,
                        max_length=self.config["train"]["generate"]["max_length"],
                        temperature=self.config["train"]["generate"]["temperature"],
                    ).replace("\n", "\\n")] for prompt in self.config["train"]["generate"]["prompts"]]
                    table = wandb.Table(data=texts, columns=["text"])
                    run.log({"generated_text": table}, step=step)
                    print("Text generated.")
            if scheduler:
                run.log({"loss": loss, "lr": scheduler(step), "epoch": i / (total_steps / n_epochs)}, step=step)
            else:
                run.log({"loss": loss, "epoch": i / (total_steps / n_epochs)}, step=step)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    trainer = Trainer(config)
    trainer.train()
