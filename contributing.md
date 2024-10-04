# Contributing

Contributions are welcome!

## Developer notes

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

This will trigger the `build_and_publish.yaml` and `run_tests.yaml` actions. If
`build_and_publish.yaml` succeeds in building the sdist and wheels, then a new
github release draft will be created (but the release will not be pushed to pypi
yet!).

Next, go to the [release page](https://github.com/yt-project/yt_experiments/releases),
open up the draft release, edit the title and write up the release notes. When ready,
hit publish -- this will again trigger the `build_and_publish.yaml` action, but this
time it will push up to pypi on success.

Note that the pypi publication configuration is setup via a [Trusted Publisher](https://docs.pypi.org/trusted-publishers/)
under @chrishavlin's pypi account (chavlin).
