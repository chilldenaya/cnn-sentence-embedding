# CNN for sentence embedding


# Start in local (Apple M1)
Create conda env config `tf-metal-arm64.yaml`
```
name: tf-metal
channels:
  - apple
  - conda-forge
dependencies:
  - python=3.9  ## specify desired version
  - pip
  - tensorflow-deps

  ## uncomment for use with Jupyter
  ## - ipykernel

  ## PyPI packages
  - pip:
    - tensorflow-macos
    - tensorflow-metal  ## optional, but recommended
```

Create a conda environment
```
conda env create -n tf -f tf-metal-arm64.yaml
```

Activate env
```
conda activate tf
```

Verify install
```
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
