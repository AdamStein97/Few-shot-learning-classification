# SB Base Pip Packages

Library for the themes explorer

...Definition here....

### Development

#### With poetry
##### <a name="install-poetry"></a>Install Poetry
Install poetry following instructions in the [confluence document](https://street-bees.atlassian.net/l/c/60N0MR1P) 
and/or [Poetry's official documentation](https://python-poetry.org/docs/#introduction) 

##### Install project dependencies and create .so files from cythonised code
Issue:

```bash
make deps
```
##### Activate/deactivate environment
To activate your environment issue:
```bash
poetry shell
```
To deactivate use `ctrl-d`. Alternatively, issue `exit` or `deactivate`

Learn more about using Poetry in the links provided above.

#### With poetry, within conda

Note: Poetry includes a virtualenv, therefore adding conda may well be unnecessary. 
If you still want to go ahead with the combination: 
Install Poetry as described in the [Install Poetry](#install-poetry) section above. 
Then, follow instructions in the relevant section in the Poetry confulence page: 
[Installing poetry dependencies within a conda environment](https://street-bees.atlassian.net/wiki/spaces/ML/pages/1457848369/Using+Poetry+for+Python+Dependency+Management#Installing-poetry-dependencies-within-a-conda-environment)


### Tests

Run `python -m pytest` on the root folder and it will find the tests.

Tests will generate an HTML coverage report which will be available in the `htmlcov` folder (open `index.html` for easier inspection).

Issue:

```bash
make test
```

`test_basic.py` but is just a bootstrap for the library own tests.

Remember:

* Add the tests to the `tests` folder
* Try to have all classes and functions tested
* Test files should start with `test_`
* Try to have direct correlation between function/classes and tests
* Same folder structure
* Same file names preceded by `test_`

> The `tests` folder is not added to the library when it is installed in a client.
> Check the `pyproject.toml` for the `packages` if you need to change this behaviour.

### Formatting

We encourage you to add pre-commit githooks using all the available formatters. To do 
so, run

```shell
$ pre-commit install
 pre-commit installed at .git/hooks/pre-commit
```

Then, once you commit ensure that git hooks are activated (Pycharm for example has the
option to omit them). This will run the formatters  automatically on all files you 
modified, failing if there are any files requiring fixes.

In any case, the `CI` configuration will most likely run checking for all formatters 
before even start the tests so it makes sure all committed code were in conformity.

To check if your code is formatted correctly issue:

```bash
make check
```

#### isort

Any time new code is added, run `isort` to sort the imports on the files.

```shell
$ isort -rc -y
 Fixing ./setup.py
 Fixing ./config/dotenv.py
 Skipped 1 files
$
```

Check `.isort.cfg` to see the style we use.

#### black

Any time new code is created, run `black` to apply the default black formatting on the 
files.

```shell
$ black .
 reformatted ./sb_base_pip_package/config/dotenv.py
 All done! ✨ 🍰 ✨
 1 file reformatted, 6 files left unchanged.
$
```

## Integration & Deployment

* Add MLFlow, DVC and Circle CI here

> The `config` folder is not added to the library when it is installed in a client.
> Check the `pyproject.toml` for the `packages` if you need to change this behaviour.


## Environment variables & Secrets

> At some point in the future we might move the secrets to the **AWS Secrets Manaver**
> until there the current practice is to use `blackbox` and encrypt the keys on the 
> repo.

For the most libraries it wont be necessary, however, many running scripts may need to
store/download files on S3 or follow specific paths to get to some files.

In this case we shall use the files `./config/.env.environment`
And import the `import config.dotenv` in any place you may need to use one ENV variable 
as in:

```python
import os
import config.dotenv

print(os.getenv("VAR_TEST", 'Default value'))
```

> The `config` folder and the secret files are not added to the library when it is 
> installed in a client.
> Check the `pyproject.toml` for the `packages` if you need to change this behaviour.

## Command line scripts

As some of the job will be to work and run the library locally, all the scripts should 
be on the `scripts`:

```shell
$ cd scripts
scripts/$ python run_lib.py args*
 Script running
```

> The `scripts` folder is not added to the library when it is installed in a client.
> Check the `pyproject.toml` for the `packages` if you need to change this behaviour.
