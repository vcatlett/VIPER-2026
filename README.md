# VIPER-2026

Start by creating a `conda` environment with Python 3.11. 

```bash
conda create -n viper-env-3.11 python=3.11
conda activate viper-env-3.11
```

Next, install `pta_replicator`

```bash
conda install pip
pip install git+https://github.com/bencebecsy/pta_replicator.git
```

Then, install `enterprise` via the `conda-forge` channel

```bash
conda install -c conda-forge enterprise-pulsar
```

Finally, install the remaining dependencies

```bash
pip install python-dotenv ipykernel
```

yeet