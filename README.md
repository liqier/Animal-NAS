# Animal-NAS
PyTorch Source code for "[Spatial Temporal Graph Convolutional Networks for four-legged wildlife Skeleton-Based Action Recognition with Neural Architecture Search]"

## Requirements
- python packages
  -pytorch = 1.2.0
  -tqdm
  -pyyaml
  -scikit-learn
  -seaborn

## Animal-Skeleton
Animal-Skeleton is our proposed skeleton-based dynamic multispecies animal behavior recognition dataset. We not only released the complete skeleton coordinate data set, but also provided some RGB animal video images obtained from pose detection

## train/test splits
we randomly divide the training samples into 10 folds, where 9 folds are used for training, and the remaining 1-fold is used for validation.

## Model searching
'''
python train_search.py
'''

## License
All materials in this repository are released under the Apache License 2.0.
  
