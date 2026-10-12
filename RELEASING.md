# Releasing

A release is a version tag, `vX.Y.Z`. Pushing the tag publishes the
documentation site at <https://kvoyager.github.io/torchsonn/> through
`.github/workflows/docs.yml`. It does not upload anything to PyPI.

## One-time setup on GitHub

1. **Settings → Pages → Build and deployment → Source:** GitHub Actions.
   This also creates the `github-pages` environment.
2. **Settings → Environments → `github-pages` → Deployment branches and
   tags:** add a rule of type *Tag* with the pattern `v*`. The environment
   accepts only the default branch until then, and a tag run fails at the
   deploy step with "Tag ... is not allowed to deploy to github-pages due to
   environment protection rules".

## Each release

1. In `CHANGELOG.md`, move the entries under `## Unreleased` into a new
   `## X.Y.Z` section.
2. Bump `version` in `pyproject.toml`. Releases bump the patch digit.
3. Update the version the README and the docs state:
   - the README citation's `version`, the only version the README names
     (PyPI shows the README as the project description, under a header
     that carries the version);
   - `docs/getting-started/install.md`: "These pages describe version
     X.Y.Z, the latest release on PyPI", and the tag (`@vX.Y.Z`) in its
     two GitHub install commands, so a reader of the published pages
     installs the version they describe.

   A page's "Checked against TorchSONN X.Y.Z" footer records the version
   the page was last checked against; change it when the page is checked
   again, not with every release.
4. Check that the tests pass and the site builds:

   ```bash
   pytest
   mkdocs build --strict
   ```

   `tests/test_readme.py` fails on what breaks the README on PyPI but not
   on GitHub: a relative link or image, a Mermaid block, a `$` outside
   code.
5. Commit everything the release changes as `release: X.Y.Z` and tag it,
   without pushing either. Step 6 builds the package from this commit, so
   the README, the changelog and the version must all be in it:

   ```bash
   git tag vX.Y.Z
   ```

   Until step 7 pushes them, a mistake is fixed by amending the commit and
   moving the tag (`git tag -f vX.Y.Z`).
6. Build the package from the tag and upload it to TestPyPI, a separate
   instance of PyPI for trying a release (the tools: `pip install build
   twine`; TestPyPI has its own account and API token):

   ```bash
   git status --short        # must print nothing
   git checkout vX.Y.Z
   rm -rf dist build
   python -m build
   twine check --strict dist/*
   twine upload --repository testpypi dist/*
   git checkout main
   ```

   `python -m build` packs the working tree, not a commit, so an edit that
   is not committed, or a file left in `build/` by an earlier build, ends
   up in the package. A clean tree at the tag and empty `dist/` and
   `build/` folders make the uploaded files match the tagged commit.
   `twine check` renders the README as PyPI will and fails if it cannot.

   Then open <https://test.pypi.org/project/torchsonn/X.Y.Z/> and check the
   description: the version in the header, both diagrams, a few links. To
   check the install as well (the dependencies come from PyPI):

   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ torchsonn==X.Y.Z
   ```

   TestPyPI, like PyPI, accepts each version's files once. A fix found
   here is amended into the release commit as in step 5; testing the fixed
   build on TestPyPI again takes the next patch version.
7. Push the branch and the tag:

   ```bash
   git push origin main
   git push origin vX.Y.Z
   ```

   On GitHub, **Actions → docs** shows the run the tag starts: the build
   job runs the drift test and the strict build, and the deploy job
   publishes the site.
8. Upload the same files to PyPI, without building again:

   ```bash
   twine upload dist/*
   ```

   PyPI accepts each version's files once and never replaces them, so a
   mistake found after the upload ships in the next patch release.

A manual run (**Actions → docs → Run workflow**) builds and deploys any
branch or tag, for example to republish a version after a docs fix on its
tag's commit, or to check the setup. A run on `main` publishes main's pages,
which may describe changes no release has yet.
