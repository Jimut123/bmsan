# BMSAN
Improving Multi Scale Attention Networks: Bayesian Optimization for Segmenting medical imagery

## Dependencies
For installing the required dependencies, please use the following packages as mentioned in [dependencies](nvdia_env.txt).



## Running the code

For running the code, we need to download the following datasets and keep them in proper folder.

#### Download the datasets

* [ISIC 2017](https://drive.google.com/u/3/uc?id=1-PAttyqJLImKMdz0gTLM3soiWWvi8PAD) (ISIC-2017_Training_Data.zip 5.4G)
* [BRAIN MRI](https://drive.google.com/u/3/uc?id=1FO-LWzFiU__2jrYXOpHw53CeoaHVoZh9) (brain.zip 352M) 
* [SKIN LESION](https://drive.google.com/u/2/uc?id=1oPVUwkc4R_JgPUYkvVwGtUj6pHQw1bue) (skin_lesion.zip 48M)
* [LUNGS](https://drive.google.com/u/3/uc?id=1wkephfRzoFhrcRbz_7a97_muu8ecSJ9d) (lung.zip 1.0G)
* [KVASIR-SEG](https://drive.google.com/u/3/uc?id=1VzOUEvIyhmOGQUBGzfI7M8GMW2NjIFeH) (Kvasir-SEG.zip 44M)
* [CHEST](https://drive.google.com/u/2/uc?id=1OF4L6tHP-3cyVrK7NhWYnNTUWJBT4gDu) (chest_qq_files.zip 626M) 
* [NERVE](https://drive.google.com/u/3/uc?id=1-EQ_Tiu2m38143y_j0QgYs5Fj5-f6hit) (nerve.zip 2.1G)


After downloading the datasets, unzip them via ``unzip`` command (in linux) into the respective directories.
For example if it is brain.zip (i.e., Brain MRI dataset) then we need to place it in the BrainMRI
folder. Each of the respective folders should have their respective datasets. Example of a folder directory may
be shown here:

```
-- BrainMRI
  |  _K_Fold_drrmsan
  |  brain
  |  AttnR2UNet_kFold
  |  AttnUNet_kFold
  |  ModifiedUNET_kFold
  |  MSAN_2
  |  MSAN_3
  | ...
  |  R2UNet_kFold
```

The main folders should have their respective state-of-the-art comparison model and the proposed models. Please find
the relevant models and run the `.py` file respectively.

For doing Bayesian Optimisation we have to first run the files present in the `GetDice` folder by selecting the best MSAN model.
This will collect the dataset for the Bayesian Optimisation part and then we may run the files present in the Final folder to get the
best possible weighted mask combination. We again use this weights for training segmentation on the best possible MSAN model.


## Contribution

Please check [CONTRIBUTING.md](https://github.com/Jimut123/drrmsan/blob/main/CONTRIBUTING.md)



## Acknowledgements

The authors are thankful to **Swathy Prabhu Mj** for arranging Asus RTX 2080 Ti (12 GB) and Quadro GV100 (32 GB) GPUs with 64 GB RAM,  to hasten the research. The first author is thankful to **Br. Tamal Maharaj** and **Dr. Jadab Kumar Pal** for their suggestions. The Authors would also like to thank [lixiaolei1982](https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-) and [nibtehaz](https://github.com/nibtehaz/MultiResUNet) for their implementations of state-of-the art models. 


# BibTeX and citations

```
@misc{pal_2021_drrmsan,
  author = {Jimut Bahan Pal and Dripta Mj},
  title = {DRRMSAN: Deep Residual Regularized Multi-Scale Attention Networks for segmentation of medical images.},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Jimut123/drrmsan/}},
}
```
***
