# (WIP) æ

Code base for training GPT-like models on TPUs with support for parallelization and scaling on JAX.

## To Do

- [x] Data parallelization on devices with `jax.sharding`
- [x] Support for bfloat16 during training
- [ ] Model parallelization with `jax.pjit` and `Mesh`
- [ ] Flash Attention support

## Special Thanks

- [Phil Wang](https://github.com/lucidrains) for the [PaLM-jax](https://github.com/lucidrains/PaLM-jax)
- [Hugging Face](https://huggingface.co/) for the [transformers](https://github.com/huggingface/transformers)