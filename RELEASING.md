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
   - the README citation's `version`;
   - `docs/getting-started/install.md`: "These pages describe version
     X.Y.Z", "install X.Y.Z from GitHub", and the tag (`@vX.Y.Z`) in its
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

5. Commit as `release: X.Y.Z`, then push the branch and the tag:

   ```bash
   git push origin main
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

6. On GitHub, **Actions → docs** shows the run: the build job runs the
   drift test and the strict build, and the deploy job publishes the site.
7. To publish the version on PyPI as well, build and upload it, then update
   "The latest release on PyPI is ..." in the README and in
   `docs/getting-started/install.md`:

   ```bash
   python -m build
   twine upload dist/*
   ```

A manual run (**Actions → docs → Run workflow**) builds and deploys any
branch or tag, for example to republish a version after a docs fix on its
tag's commit, or to check the setup. A run on `main` publishes main's pages,
which may describe changes no release has yet.
