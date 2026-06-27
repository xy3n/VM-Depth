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

To train a CityScapes model, run:

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

If you have not yet processed the CityScapes data set, please refer to [ManyDepth](https://github.com/nianticlabs/manydepth) for processing.


## Evaluation

### KITTI dataset

First you have run `export_gt_depth.py` to extract ground truth files.

To evaluate a model on KITTI, run:

```bash
python -m vmdepth.evaluate_depth \
    --data_path <your_KITTI_path> \
    --load_weights_folder <your_model_path>
    --eval_mono
    --eval_split eigen
```

### Cityscapes dataset

The ground truth depth files [Here](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip).

To evaluate a model on Cityscapes, run:

```bash
python -m vmdepth.evaluate_depth \
    --data_path <your_cityscapes_path> \
    --load_weights_folder <your_model_path>
    --eval_mono \
    --eval_split cityscapes
```
And to evaluate a model on Cityscapes (Dynamic region only), run:

```bash
python -m vmdepth.evaluate_depth_dynamic \
    --data_path <your_cityscapes_path> \
    --load_weights_folder <your_model_path>
    --eval_mono \
    --eval_split cityscapes
```

Please make sure you switch the dynamic region dataloader. And the dynamic object masks for Cityscapes dataset can download from [Here](https://github.com/AutoAILab/DynamicDepth).

 

## Citation Note

This repository provides the official implementation of the paper:

> "VM-Depth: Motion-Aware Cost Volume Framework for Self-Supervised Monocular Depth Estimation in Dynamic Scenes
"

which is currently submitted to *The Visual Computer*.

If you use this codebase in your research, please cite the paper above. 


## Acknowledgments

This project builds upon several prior works in self-supervised monocular depth estimation. We sincerely acknowledge the contributions of the following open-source projects:

 [ManyDepth](https://github.com/nianticlabs/manydepth)  
 [DS-Depth](https://github.com/xingy038/DS-Depth)  
 [ManyDepth2](https://github.com/kaichen-z/Manydepth2)


