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

You may need to specify the channel:
```conda install -c conda-forge mamba```

If you don't have `anaconda` or `miniconda`, just get [Mambaforge](https://github.com/conda-forge/miniforge/releases/tag/4.9.2-5).

## 3. Setup the `base` environment

This will be an environment that has [`anaconda-project`](https://anaconda-project.readthedocs.io) and some other dependencies to run the development commands.

For added consistency, use the provided `.condarc` file:

```bash
CONDARC=.github/.condarc        # linux
set CONDARC=.github\.condarc    # windows
```

```bash
conda env create -f .github/environment.yml -p envs/pymbe
```

> You don't have to repeat these steps unless you delete your base environment.

## 4. Activate the environment

You will have to activate the base environment to make sure you are using the appropriate version.  You will have to do this every time you open a new shell (e.g., windows command prompt, a linux bash) unless you automate the activation of the environment.

```bash
conda activate envs/pymbe
```

### Additional configuration

> If you are going to be changing the `anaconda-project` environments, it is a good idea to make `anaconda-project` use `mamba` instead of `conda` (it is much faster)

> Remember you will have to do this when you start your shell unless you set these environment variables permanently.

```bash
CONDA_EXE=mamba        # linux
set CONDA_EXE=mamba    # windows
```

> Remember to set your `.condarc` path as explained in `Step 3`.

## 5. Setup the Development Environment

> This will install the non-packaged dependencies and `pymbe` in editable mode.

```bash
anaconda-project run setup
```

# ... and get going!

You can then get a running instance of JupyterLab by running:

```bash
anaconda-project run lab
```

> Remember if you open a new shell, you will have to activate your base environment as explained in `Step 4`.

Copy the URL where JupyterLab is running into your preferred browser, and you should be good to go!

## Widgets

You can interact with the SysML v2 data using widgets, as illustrated below:
![Composed Widget](https://user-images.githubusercontent.com/1438114/113528145-bb494280-958d-11eb-8d9f-5b8f7d2b1dbe.gif)

> If you can't see the animation clearly, click on it to see it in higher resolution.

## Setup SysML Kernel

If you want to experiment with the SysML v2 kernel, you can install it on the environment you want by running the following command:

```bash
anaconda-project run sysml:kernel:install
```

## Setup SysML API Kernel

You can setup and run a local SysML Pilot Implementation API server by running:

```bash
anaconda-project run sysml:api:setup  # to install the service
anaconda-project run sysml:api:start  # to start the server
anaconda-project run sysml:api:stop   # to stop the server
```
