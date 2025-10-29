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

# How to use Augmentation module

This module will enrich and balance our dataset. By creating new images with slight alterations from our base repository, we make sure we have a more complex dataset for our training.

It has two behaviours:

1- Create 6 new images from a base image
2- Fill out a root directory with new images until a max is reached at each folder.

If we just want to create 6 new images with the available augmentations we run 
```Augmentation.py --path path/to/file```
This will create 6 new images.

If we want to fill out a folder with randomly selected augmentations we must run:

```Augmentation.py --root root/folder --num number_of_images_to_reach```

Example:

```Augmentation.py --root Apple/ --num 3000```

This will check the root folder Apple, will check each sub folder, like Apple_healty, and select random images and a random number of augmentations to fill out this folder images until it reaches 3000. 



