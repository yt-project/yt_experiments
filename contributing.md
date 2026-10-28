# Contributing

Contributions are welcome!

## Developer notes

### Test suite

Tests are run with `pytest`, to install test dependencies:

```shell
pip install -e .[test,full]
```

To run tests:

``` shell
pytest -v .
```

### Releasing

Releases use a mostly automated pipeline of github actions triggered by pushing
a new version tag.

Assuming your local fork has a remote `upstream` pointing to this repo, first
make sure your local `main` branch matches `upstream`:

```shell
git checkout main
git fetch --all
git rebase upstream/main
```

Also make sure the version in `yt_experiments/_version.py` matches the upcoming
release. If not, stop here and create a PR to update the version to the upcoming
release.

Now create the new version tag:

```shell
git tag v2.1.3
```

And push it upstream

```shell
git push upstream v2.1.3
```

This will trigger the `build_and_publish.yaml` and `run_tests.yaml` actions.
`build_and_publish.yaml` builds the sdist and wheels, then uploads them to pypi.
If the pypi upload succeeds, a draft github release is created with the sdist
and wheels attached (including the pyodide wheel, which is not uploaded to pypi).

Next, go to the [release page](https://github.com/yt-project/yt_experiments/releases),
open up the draft release, add a short description of the release under the
version heading and double check the auto-generated release notes. When ready, hit
publish. Note that publishing the github release does not trigger any further
actions: the package is already on pypi at this point.

Note that the pypi publication configuration is setup via a [Trusted Publisher](https://docs.pypi.org/trusted-publishers/).
