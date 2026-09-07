# Contributing to GraphSSL

Thanks for considering a contribution. GraphSSL is research software developed
at [NECSTLab](https://necst.it), Politecnico di Milano — issues, PRs and
questions are all welcome.

## Setup

```bash
git clone https://github.com/Simocop7/graphssl.git
cd graphssl
pip install -e ".[dev]"
pre-commit install   # runs ruff on every commit
```

## Before opening a PR

```bash
pytest tests/ -v          # 107 passed, 1 skipped (AFGRL skips without faiss-cpu)
ruff check .               # lint
ruff format --check .      # formatting
mypy src/graphssl          # informational for now, see below — don't block on it
```

`pre-commit install` runs `ruff check --fix` and `ruff format` automatically on
`git commit`; CI re-runs the same checks plus the full test matrix (Python
3.10–3.12) on every push and PR.

## Code style

- Formatting and import order are enforced by `ruff format` / `ruff check`
  (config in `pyproject.toml`) — don't hand-format, let the tool do it.
- Docstrings follow the Google style already used throughout the codebase
  (`Args:` / `Returns:` sections) — match it for new public functions/classes.
- Type hints use the pre-PEP 604/585 style (`Optional[X]`, `List[X]`, not
  `X | None` / `list[X]`) to stay consistent with the existing codebase.
  Modernizing this repo-wide is tracked as a separate, deliberate follow-up —
  don't mix it into an unrelated PR.
- No new dependency on PyTorch Lightning, Hydra, or OmegaConf in `src/` —
  see the "Stack" section of `CLAUDE.md` for the project's zero-heavyweight-
  dependency stance. Optional/heavier deps (faiss, ogb, umap, ...) go behind
  an extra in `pyproject.toml`, never into the core install.

## Type checking status

mypy runs in CI as an **informational, non-blocking** job
(`continue-on-error: true` in `.github/workflows/tests.yml`). Most current
findings are false positives from `nn.Module.__getattr__` being typed
`Tensor | Module` — mypy can't statically tell a submodule access from a
tensor/parameter access unless the attribute is explicitly annotated. Rather
than blanket-suppressing or annotating everything in one pass, we're tightening
this module-by-module: if you touch a module and want to fix its mypy findings
along the way (explicit `self.foo: nn.Module = ...` annotations), that's a
welcome, easy-to-review addition to a PR. Don't let it block unrelated work.

## Adding a new model

Every model follows the same construction contract — see `CLAUDE.md`
("COME AGGIUNGERE UN NUOVO MODELLO") for the full checklist: config dataclass
in `config/schema.py`, `cfg.encoder.build(in_channels)` (never build the
encoder inline), `graph_level` derived from `cfg.encoder.pool`, and updates to
`models/__init__.py` / `src/graphssl/__init__.py` exports. Read both the
reference paper and the existing implementations of a similar model before
writing code.

## Adding a new encoder / loss / augmentation

Register it via the appropriate registry (`@ENCODERS.register("name")`,
`@LOSSES.register("name")`, `@AUGMENTS.register("name")`) rather than adding
special-casing elsewhere — see `src/graphssl/registry/`.

## Tests

New models/encoders/losses need tests under `tests/`, following the existing
per-component file layout (`test_<model>.py`). `tests/helpers.py` has shared
graph/batch builders — reuse them instead of duplicating fixture code.
AFGRL-related tests must stay auto-skippable when `faiss-cpu` isn't installed
(`pytest.importorskip("faiss", ...)`, see `tests/test_afgrl.py`).

## Commit messages

Keep commits scoped to one logical change (e.g. don't mix a formatter-adoption
diff with a behavior change) — it keeps `git blame` and review useful. No
required prefix convention, but a short imperative summary line is
appreciated.
