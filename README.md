# Multimodal Unlearnable Examples: Protecting Data against Multimodal Contrastive Learning


## Requirements

- Both Linux and Windows are supported, but we strongly recommend Linux for performance and compatibility reasons.
- 64-bit Python 3.7+ installation.

## Setup Environment and Install dependencies


### Conda (recommended)

Please follow the instructions at the following link to set up
anaconda: [Anaconda Setup](https://docs.anaconda.com/anaconda/install/index.html)

The following commands create a conda environment inside the repository with the dependencies.

```bash
conda env create --prefix ./env -f environment.yml
source activate ./env
```

### Pip

The requirements can be directly installed without creating a conda environment.

```bash
pip install -r requirements.txt
```
### Clean Training with Flick8K
```
python -m src.main --name clean_flick8k --train_data /data/clip/Flicker-8k/train.csv --eval_test_data_dir /data/clip/Flicker-8k/test.csv --image_key images  --caption_key caption  --device_id 0
```


### Protecting Multimodal Unlearnable Examples ()

set the text trigger length as 3
```
python -m src.poison --name poison_token_3_shuffle --train_data /data/clip/Flicker-8k/train.csv  --image_key images  --caption_key caption  --device_id 0 --token_num 3 --lr 1e-4 
```

set the text trigger length as 5
```
python -m src.poison --name poison_token_5_shuffle --train_data /data/clip/Flicker-8k/train.csv  --image_key images  --caption_key caption  --device_id 1 --token_num 5 --lr 1e-4
```

### Evaluating Performance 

```
python -m src.poison_main --name eval_token_3_shuffle --train_data /data/clip/Flicker-8k/train.csv --eval_test_data_dir /data/clip/Flicker-8k/test.csv --image_key images  --caption_key caption  --device_id 0 --save_pert poison_token_3_shuffle --token_num 3  --lr 5e-4
```

```
python -m src.poison_main --name eval_token_5_shuffle --train_data /data/clip/Flicker-8k/train.csv --eval_test_data_dir /data/clip/Flicker-8k/test.csv --image_key images  --caption_key caption  --device_id 1 --save_pert poison_token_5_shuffle --token_num 5  --lr 5e-4
```