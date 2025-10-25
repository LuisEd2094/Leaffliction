## Pre-commit — install & usage

1. Add a `.pre-commit-config.yaml` to the repo root. Example (adjust hooks to your stack):
```yaml
repos:
    - repo: https://github.com/pre-commit/pre-commit-hooks
        rev: v4.4.0
        hooks:
            - id: trailing-whitespace
            - id: end-of-file-fixer
            - id: check-yaml
            - id: check-added-large-files

    # Python examples
    - repo: https://github.com/psf/black
        rev: 24.1.0
        hooks:
            - id: black
    - repo: https://github.com/PyCQA/isort
        rev: 5.12.0
        hooks:
            - id: isort
```

2. Install pre-commit (pick one):
- pip: `pip install pre-commit`


3. Enable the git hook in this repo:
- `pre-commit install`
- to install the configured hook versions locally: `pre-commit install --install-hooks`

4. Run hooks manually:
- all files: `pre-commit run --all-files`
- single hook: `pre-commit run <hook-id> --all-files`
- on staged files (default): `pre-commit run`

5. Update hooks:
- `pre-commit autoupdate`
- then: `pre-commit run --all-files` to apply changes

6. CI integration (example step):
- Run `pre-commit run --all-files` as part of your CI job and fail the build if hooks fail.

Notes:
- Adjust `.pre-commit-config.yaml` hook list and revs to match the project stack.
- If you add new hooks, re-run `pre-commit install --install-hooks`.
- For repositories without source yet, keep the config generic (formatting + safety checks) and update when language-specific tools are known.