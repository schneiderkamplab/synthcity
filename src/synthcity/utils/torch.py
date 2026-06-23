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
    _patch_optimizer_steps(torch)


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


def _patch_optimizer_steps(torch: ModuleType) -> None:
    optim = getattr(torch, "optim", None)
    if optim is None:
        return

    for value in vars(optim).values():
        if not isinstance(value, type):
            continue
        step = getattr(value, "step", None)
        wrapped = getattr(step, "__wrapped__", None)
        if wrapped is not None:
            setattr(value, "step", _eager_optimizer_step(torch, wrapped))


def _eager_optimizer_step(torch: ModuleType, fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults.get("differentiable", False))
            return fn(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(previous_grad)

    return wrapper
