


# Fundust image classification

[![GitHub license](https://img.shields.io/github/license/用户名/仓库名)](https://github.com/用户名/仓库名/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## 🛠️ Install dependencies
```python
conda create -n fundus python=3.11.7
conda activate fundus
pip install -r requirements.txt
```

## 📁 Data preparation
download fairvlmed10k datasets
```python

mv data/* /src/fairvlmed10k

```

The Harvard-FairSeg dataset can be accessed via this [link](https://github.com/Harvard-Ophthalmology-AI-Lab/FairSeg?tab=readme-ov-file), and the Harvard-FairVLMed dataset can be accessed via this [link](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP).

```
data
 ├── FairSeg_data_summary.csv
 └── FairVLMed_data_summary.csv
```

We provide the files `FairSeg_data_summary.csv` and `FairVLMed_data_summary.csv`, which contain the filenames used in our experiments, along with information on whether each file is used for the training, validation, or test set. In addition to the demographic information and medical records provided by the source data, we categorize ages under 65 as "young" and ages 65 and above as "elderly".

## 🚀 Train

```python
python Train.py --pretrain_model  --model efficientnet-b0 --data_root /root/ --dataset fairvlmed10k --use_fake_data  --epochs 100 --top_precentege 0
```
Common parameters:
- --model: Model selection
- --epochs: Number of epochs
- --device: Training device (cuda:0/cpu)
- --data_root: Root directory of the dataset
- --dataset: Name of the dataset used
- --pretrain_model: Whether to use a pre-trained model
- --use_fake_data: Whether to use generated data
- --top_percentage: Percentage of generated data to use
- --balance_data: Whether to balance the quantities of each class
- --balance_attribute: Attribute to balance
- --best_model_path: Path to the loaded model (useful only for running the test set)

## 🧪 Test

```python
python --model efficientnet-b0 --data_root /root/ --dataset fairvlmed10k --best_model_path checkpoints/label_classification/best_auc.pth
```
