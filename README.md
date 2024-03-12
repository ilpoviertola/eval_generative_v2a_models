# Evaluation of Generative Audio Models

This repo is a collection of scripts to evaluate generative audio models. It is based on PyTorch.

## Usage

This section walks you through the process of evaluating a generative audio model. The following steps are required:

### 1. Install environment

First, you need to install the required environment. You can do this by running the following command:

```bash
conda env create -f conda_env.yaml
```

### 3. Download Synchformer checkpoints

This evaluation pipeline uses [Synchformer](https://github.com/v-iashin/Synchformer) model to analyze the audio-visual synchronization. Run the following command to download the Synchformer checkpoints:

```bash
bash ./checkpoints/download_synchformer_checkpoints.sh
```
