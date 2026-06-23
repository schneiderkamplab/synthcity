# stdlib
import functools
import os
import sys
from types import ModuleType
from typing import Any, Callable, Optional


def disable_torch_dynamo() -> None:
    """Run synthcity Torch models eagerly without importing TorchDynamo."""
    if os.environ.get("TORCHDYNAMO_DISABLE", "").lower() in ("", "0", "false", "no"):
        return

    try:
        import torch
    except ImportError:
        return

    torch._disable_dynamo = _no_dynamo_wrapper  # type: ignore[attr-defined]
    _seed_existing_dynamo_wrappers()


def _no_dynamo_wrapper(
    fn: Optional[Callable] = None,
    recursive: bool = True,
) -> Any:
    if fn is not None:
        return fn

    return functools.partial(_no_dynamo_wrapper, recursive=recursive)


def _seed_existing_dynamo_wrappers() -> None:
    for name, module in list(sys.modules.items()):
        if name == "torch.optim" or name.startswith("torch.optim."):
            _seed_module_dynamo_wrappers(module)


def _seed_module_dynamo_wrappers(module: ModuleType) -> None:
    for value in vars(module).values():
        _seed_dynamo_wrapper(value)
        if isinstance(value, type):
            for member in vars(value).values():
                _seed_dynamo_wrapper(member)


def _seed_dynamo_wrapper(value: Any) -> None:
    wrapped = getattr(value, "__wrapped__", None)
    if wrapped is not None:
        try:
            wrapped.__dynamo_disable = wrapped
        except (AttributeError, TypeError):
            pass
