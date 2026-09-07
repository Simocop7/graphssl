## What

<!-- One or two sentences: what does this change, and why. -->

## Checklist

- [ ] `pytest tests/ -v` passes locally (107 passed, 1 skipped without faiss-cpu)
- [ ] `ruff check .` and `ruff format --check .` pass (or `pre-commit run --all-files`)
- [ ] New models/encoders/losses have tests under `tests/`
- [ ] `CLAUDE.md` updated if this changes a model's config, loss formula, or
      default hyperparameters (see "COME AGGIUNGERE UN NUOVO MODELLO")
- [ ] This PR is scoped to one logical change (no unrelated formatting/refactor
      mixed in — see `CONTRIBUTING.md`)

## Related issue

<!-- Closes #... , if applicable -->
