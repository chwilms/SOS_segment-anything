# SAM for SOS

This is the Segment Anything Model (SAM) version for our [Segment Object System (SOS)](https://github.com/chwilms/SOS).

## Installation

Please follow the general installation instructions of SAM as described in [the original README.md](https://github.com/chwilms/SOS_segment-anything/blob/main/README_SAM.md#installation). Additionally, install the following packages to your Python environment:

```
pip install scikit-image tqdm
```

## Usage

This version of SAM contains two new scripts for applying SAM to images based on the prompts provided by the object priors evaluated in our [SOS paper](). See also our main SOS git for [example prompt files from various object priors](). 

To apply SAM as part of SOS with object priors other than the *Grid*, use the [applySAM.py](https://github.com/chwilms/SOS_segment-anything/blob/main/applySAM.py) script with the following parameters: the path to input images, e.g. COCO's train2017 folder, the path to the SAM checkpoint, the path to the prompt file, and the path for the output directory:

```
python applySAM.py /data/SOS/coco/train2017 /data/SOS/SAM_checkpoints/sam_vit_h_4b8939.pth /data/SOS/prompts/prompts_DINO.json /data/SOS/segments/segments_DINO.json
```

For the *Grid* obejct prior, which is essentially SAM's everything mode, use the [applySAM_grid.py](https://github.com/chwilms/SOS_segment-anything/blob/main/applySAM_grid.py) script with the same parameters, except for the prompt file, which is not necessary here:

```
python applySAM_grid.py /data/SOS/coco/train2017 /data/SOS/SAM_checkpoints/sam_vit_h_4b8939.pth /data/SOS/segments/segments_Grid.json
```
