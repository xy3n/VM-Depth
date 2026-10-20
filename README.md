# VM-Depth: Motion-Aware Cost Volume Framework for Self-Supervised Monocular Depth Estimation in Dynamic Scenes



> **VM-Depth: Motion-Aware Cost Volume Framework for Self-Supervised Monocular Depth Estimation in Dynamic Scenes**<br>




## Setup

To get started, please create the conda environment by running

```bash
cd vmdepth
conda env create -f environment.yaml
conda activate vmdepth
```

## Train

To train a KITTI model, run:

```bash
python -m vmdepth.train \
    --data_path <your_KITTI_path> \
    --log_dir <your_save_path> \
    --model_name <your_model_name>
```


For instructions on downloading the KITTI dataset, see [Monodepth2](https://github.com/nianticlabs/monodepth2)

To train a Cityscapes model, run:

```bash
python -m vmdepth.train \
    --data_path <your_preprocessed_cityscapes_path> \
    --log_dir <your_save_path> \
    --model_name <your_model_name> \
    --dataset cityscapes_preprocessed \
    --split cityscapes_preprocessed \
    --freeze_teacher_epoch 5 \
    --height 192 --width 512
```

If you have not yet processed the Cityscapes dataset, please refer to [ManyDepth](https://github.com/nianticlabs/manydepth) for processing.


## Evaluation

### KITTI dataset

First, run `export_gt_depth.py` to extract ground truth files.

To evaluate a model on KITTI, run:

```bash
python -m vmdepth.evaluate_depth \
    --data_path <your_KITTI_path> \
    --load_weights_folder <your_model_path> \
    --eval_mono \
    --eval_split eigen
```


### Cityscapes dataset

The ground truth depth files [Here](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip).

To evaluate a model on Cityscapes, run:

```bash
python -m vmdepth.evaluate_depth \
    --data_path <your_cityscapes_path> \
    --load_weights_folder <your_model_path> \
    --eval_mono \
    --eval_split cityscapes
```
And to evaluate a model on Cityscapes (Dynamic region only), run:

```bash
python -m vmdepth.evaluate_depth_dynamic \
    --data_path <your_cityscapes_path> \
    --load_weights_folder <your_model_path> \
    --eval_mono \
    --eval_split cityscapes
```

#### Cityscapes Dynamic Semantic-Category Masks

For the Cityscapes dynamic semantic-category evaluation, we directly use the pre-generated semantic masks publicly provided by the [DynamicDepth](https://github.com/AutoAILab/DynamicDepth) project. We did not develop or release an independent tool for generating dynamic-category masks. 

##### Source
The dynamic semantic-category masks are provided by the [DynamicDepth](https://github.com/AutoAILab/DynamicDepth) project.


##### Download Link
Download dynamic semantic-category masks for Cityscapes dataset from [Google Drive](https://drive.google.com/drive/folders/1GqDVWIIpK02rBPILDn1UmwlfFjGJyRrE?usp=sharing).

##### Usage Instructions
1. Extract `val_mask.zip` (5 MB) and place the `val_mask` folder under `vmdepth/data/path/CS/`. Please replace `path` with the actual data path used in your setup.
2. Switch to the dynamic-region dataloader before running the evaluation.


### UAV dataset
The UAV evaluation protocol and the corresponding usage instructions are provided in this repository.

To evaluate a model on the UAV dataset, run:

```bash
python -m vmdepth.evaluate_depth_UAV \
    --data_path <your_UAV_path> \
    --load_weights_folder <your_model_path> \
    --eval_mono \
    --eval_split eigen
```

## Citation Note

This repository provides the official implementation of the paper:

> "VM-Depth: Motion-Aware Cost Volume Framework for Self-Supervised Monocular Depth Estimation in Dynamic Scenes"

which is currently submitted to *The Visual Computer*.

If you use this codebase in your research, please cite the paper above. 


## Acknowledgments

This project builds upon several prior works in self-supervised monocular depth estimation. We sincerely acknowledge the contributions of the following open-source projects:

[ManyDepth](https://github.com/nianticlabs/manydepth)  
[DS-Depth](https://github.com/xingy038/DS-Depth)  
[ManyDepth2](https://github.com/kaichen-z/Manydepth2)  
[DynamicDepth](https://github.com/AutoAILab/DynamicDepth)


