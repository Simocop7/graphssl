from __future__ import annotations
from typing import Callable, Dict, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic registry mapping string names to builder callables.

    Usage:
        ENCODERS: Registry = Registry()

        @ENCODERS.register("gcn")
        class GCNEncoder(nn.Module): ...

        enc = ENCODERS.build("gcn", in_channels=32, hidden_dim=64)
    """

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(fn: Callable[..., T]) -> Callable[..., T]:
            if name in self._builders:
                raise KeyError(f"Duplicate registration: {name!r}")
            self._builders[name] = fn
            return fn
        return deco

    def get_builder(self, name: str) -> Callable[..., T]:
        if name not in self._builders:
            raise KeyError(
                f"Unknown component {name!r}. Available: {list(self._builders)}"
            )
        return self._builders[name]

    def build(self, name: str, **kwargs) -> T:
        return self.get_builder(name)(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._builders

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def __repr__(self) -> str:
        return f"Registry({list(self._builders)})"
