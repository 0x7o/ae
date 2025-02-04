# sharding.py
from functools import partial
from typing import Any, Sequence, Union
import re

import jax
from jax.sharding import PartitionSpec as P
from flax import struct


@struct.dataclass
class PartitionRules:
    """Правила разделения параметров модели."""
    rules: Sequence[tuple[Union[str, Sequence[str]], Any]]

    def get_partition_rules(self):
        """Преобразует правила в формат, понятный JAX."""
        patterns = []
        for rule_pattern, rule_spec in self.rules:
            if isinstance(rule_pattern, str):
                patterns.append((rule_pattern, rule_spec))
            else:
                # Если паттерн - это кортеж строк, объединяем их через /
                pattern = "/".join(rule_pattern)
                patterns.append((pattern, rule_spec))
        return patterns

    def get_param_spec(self, param_name: str) -> P:
        """Получает спецификацию разделения для конкретного параметра."""
        for pattern, spec in self.rules:
            if isinstance(pattern, str):
                if re.match(pattern, param_name):
                    return spec
            else:
                # Для паттерна в виде кортежа строк проверяем каждую часть
                param_parts = param_name.split('/')
                if len(param_parts) == len(pattern):
                    matches = all(
                        re.match(p, part)
                        for p, part in zip(pattern, param_parts)
                    )
                    if matches:
                        return spec
        return P()  # Возвращаем пустой PartitionSpec, если нет совпадений


DEFAULT_TRANSFORMER_RULES = [
    # Правила для attention слоев
    (("block_.*", "MultiheadGQA_.*", "q_proj", "kernel"), P("data", "model")),
    (("block_.*", "MultiheadGQA_.*", "k_proj", "kernel"), P("data", "model")),
    (("block_.*", "MultiheadGQA_.*", "v_proj", "kernel"), P("data", "model")),
    (("block_.*", "MultiheadGQA_.*", "out_proj", "kernel"), P("model", "data")),

    # Биасы в attention не шардируются
    (("block_.*", "MultiheadGQA_.*", ".*_proj", "bias"), P(None)),

    # Правила для feed-forward слоев
    (("block_.*", "ff", "wi", "kernel"), P("data", "model")),
    (("block_.*", "ff", "wo", "kernel"), P("model", "data")),
    (("block_.*", "ff", "(wi|wo)", "bias"), P(None)),

    # Эмбеддинги
    (("embed", "embedding"), P("model", "data")),

    # Выходной слой
    (("out", "kernel"), P("data", "model")),
    (("out", "bias"), P(None)),

    # Layer Norms не шардируются
    ((".*norm.*", "scale"), P(None)),
    ((".*norm.*", "bias"), P(None)),
]

# Создаем дефолтные правила
DEFAULT_RULES = PartitionRules(DEFAULT_TRANSFORMER_RULES)


def with_sharding_constraint(x, mesh, partition_specs):
    """Применяет ограничения шардинга к промежуточным активациям."""
    return jax.lax.with_sharding_constraint(x, P(*partition_specs))