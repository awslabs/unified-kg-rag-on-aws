## Summary

<!-- What does this change do? Link related issues (e.g. "Closes #123"). -->

## Why

<!-- The problem or motivation; the diff already shows the what. -->

## Testing

<!-- How was this verified? Note any real-AWS runs separately. -->

- [ ] `uv run pytest -m "not aws"` passes
- [ ] `uv run pre-commit run --all-files` passes
- [ ] `CHANGELOG.md` updated for user-visible changes (breaking changes marked)
- [ ] Docs and `config-template.yaml` updated if behaviour or configuration changed
- [ ] Tests, fixtures and examples use synthetic data only (no customer data)

## Breaking changes

<!-- Configuration, stored index data, or public interfaces affected; or "None". -->
