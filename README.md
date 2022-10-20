# [Moreh] Running on HAC Machine
![](https://badgen.net/badge/Moreh-HAC/fail/red) ![](https://badgen.net/badge/Nvidia-A100/passed/green)

Requiring `torch>=1.8.1`. Tested only on A100 VM with `torch==1.12.1`.

## Prepare

### Data
For testing purpose, we use `imagenet_100cls`, a subset of ImageNet with 100 classes.
Get the dataset from [here](http://ref.deploy.kt-epc.moreh.io:8080/reference/dataset/imagenet_100cls.tar.gz)
and extract it. The structure of the dataset is already compatible.

### Code
```bash
git clone https://github.com/loctxmoreh/ConvNeXt
cd ConvNeXt
```

### Environment
Create a conda environment using the `a100env.yml` file:
```bash
conda env create -f a100env.yml
conda activate convnext-torch
```

## Run

### Training
Assuming the ImageNet dataset is located at `/data/work/dataset/imagenet_100cls`,
and a directory `/data/work/convnext_output` is created to store checkpoints,
run the following script to start training:
```bash
./train-single-node.sh
```

You can edit the script to change some parameters.

**NOTE**: due to the use of `cosine_scheduler`, the number of training epochs
must be set to something large enough, e.g. 25

### Finetuning
Make sure there is a checkpoint at `/data/work/convnext_output/checkpoint-best-ema.pth`,
then run the finetune script:
```bash
./finetune-single-node.sh
```
