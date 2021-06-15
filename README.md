# PyMBE

A project to experiment with and validate semantics for `KerML` and `SysML v2`. Running core semantics and interpreting models is a foundational capability for any modeling software.

# Get set up...

## 1. Clone this repo

```bash
git clone https://github.com/bjorncole/pymbe.git
cd pymbe
```

## 2. Get `mamba`

> or stick to `conda` if you like...  just change `mamba` to `conda` in the instructions below.

If you have anaconda or miniconda, install `mamba` (it's faster and better than conda):

```bash
conda install mamba
```

If you don't have `anaconda` or `miniconda`, just get [Mambaforge](https://github.com/conda-forge/miniforge/releases/tag/4.9.2-5).

## 2. Build `base-pymbe`

If you don't have `anaconda-project`, install [anaconda-project](https://anaconda-project.readthedocs.io):

```bash
mamba env update --name base-pymbe --file deploy/env_files/env-base.yml
activate base-pymbe
```
OR (on Unix)
```
source activate base-pymbe
```


## 5. Setup the Development Environment

> This will setup necessary environments, install the non-packaged dependencies, and `pymbe` in editable mode.

```bash
doit
```

# ... and get going!

You can then get a running instance of JupyterLab by running:

```bash
doit lab
```

Copy the URL where JupyterLab is running into your preferred browser, and you should be good to go!

## Widgets

You can interact with the SysML v2 data using widgets, as illustrated below:
![Composed Widget](https://user-images.githubusercontent.com/1438114/113528145-bb494280-958d-11eb-8d9f-5b8f7d2b1dbe.gif)

> If you can't see the animation clearly, click on it to see it in higher resolution.
