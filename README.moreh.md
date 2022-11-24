# [Moreh] Official ConvNeXt implementation running on Moreh AI Framework
![](https://badgen.net/badge/Moreh-HAC/passed/green) ![](https://badgen.net/badge/Nvidia-A100/passed/green)

Requiring `torch>=1.8.1`.
Tested on HAC VM with `torch==1.10.0+cpuonly.moreh0.2.0`
and on A100 VM with `torch==1.12.1`.


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
Create a conda environment using the `a100env.yml` file (on A100 machines) or `hacenv.yml` (on HAC machine):
```bash
# On A100 VM
conda env create -f a100env.yml

# On HAC VM
conda env create -f hacenv.yml

# then activate the env
conda activate convnext-torch
```

## Run
Edit the two `train-single-node.sh` and `finetune-single-node.sh` scripts and change:
- `dataset_dir` to the location of the `imagenet_100cls` dataset
- `output_dir` to whatever directory used to store checkpoints
- and other configuration params

### Training
```bash
./train-single-node.sh
```

**NOTE**: due to the use of `cosine_scheduler`, the number of training epochs
in `train-single-node.sh` must be set to something large enough, e.g. 25

### Finetuning
Make sure after training, there is a checkpoint at
`${output_dir}/checkpoint-best-ema.pth`, then run the finetune script:
```bash
./finetune-single-node.sh
```
