# Animal-NAS
PyTorch Source code for "[Automatically recognizing four-legged animal behaviors to enhance welfare using spatial temporal graph convolutional networks]"

## Requirements
- python packages
  -pytorch = 1.2.0
  -tqdm
  -pyyaml
  -scikit-learn
  -seaborn

## Animal-Skeleton
Animal-Skeleton is our proposed skeleton-based dynamic multispecies animal behavior recognition dataset. We released the complete skeleton coordinate data set(Ambling.rar, Galloping.rar, Lying.rar, Sitting.rar, Standing.rar).

## train/test splits
we randomly divide the training samples into 10 folds, where 9 folds are used for training(data_joint_train.npy and label_train.pkl), and the remaining 1-fold is used for validation(data_joint_val.npy and label_val.pkl).

## Model searching
'''
python train_search.py
'''

## Citation
If you use this code or dataset, please cite this article as: Yaqin Zhao, Liqi Feng, Jiaxi Tang, Wenxuan Zhao, Zhipeng Ding, Ao Li and Zhaoxiang Zheng, Automatically recognizing four- legged animal behaviors to enhance welfare using spatial temporal graph convolutional networks, Applied Animal Behaviour Science, (2021) doi:https://doi.org/10.1016/j.applanim.2022.105594

## License
All materials in this repository are released under the Apache License 2.0.
  
