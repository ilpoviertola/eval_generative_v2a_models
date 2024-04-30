# Evaluation of Generative Visual to Audio (V2A) Models

This repo provides an easy way to evaluate generative V2A models. It is based on PyTorch.

## Usage

This section walks you through the process of evaluating a generative v2a model. The following steps are required:

### 1. Install environment

First, you need to install the required environment. You can do this by running the following command:

```bash
conda env create -f conda_env_cu12.1.yaml
```

Also, [AudioTools by Descript Inc.](https://github.com/descriptinc/audiotools) is needed for audio processing and PASST model for metrics. You can install them by running the following commands:

```bash
pip install git+https://github.com/descriptinc/audiotools
pip install git+https://github.com/kkoutini/passt_hear21@0.0.19#egg=hear21passt
```

### 2. Download Synchformer checkpoints

This evaluation pipeline uses [Synchformer](https://github.com/v-iashin/Synchformer) model to analyze the audio-visual synchronization. Run the following command to download the Synchformer checkpoints:

```bash
bash ./checkpoints/download_synchformer_checkpoints.sh
```

### 3. Run the evaluation pipeline

Use run_evaluations.ipynb to run the pipeline. All the required steps are described in the notebook.
