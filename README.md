# BMSAN
### This repository contains the code of our paper: [Improving Multi Scale Attention Networks: Bayesian Optimization for Segmenting medical imagery](https://www.tandfonline.com/doi/full/10.1080/13682199.2023.2174657) which  has been recently published at The Imaging Science Journal.


**Abstract:** Current deep learning-based image segmentation methods are notable for their use of large number of parameters and extensive computational resources in training. There is a persistent need for more efficient flexible systems without compromising on precision. This work proposes a novel model that combines the best of deep learning and probabilistic machine learning to segment a wide variety of medical image datasets with state-of-the-art accuracy and limited resources. The approach benefits from the introduction of new diverse attention modules that serve multiple purposes including capturing of relevant information at different scales. These proposed attention modules are generic and can potentially be used with other architectures to boost performance. In addition, Bayesian optimization is employed to tune multi-scale weight hyperparameters of the model. The architecture combined with one of the proposed novel attention modules and tuned hyperparameters achieves the best results in segmenting ISIC 2017, LUNGS, NERVE, Skin Lesion and CHEST datasets. Finally, the explainability of the network is analysed by visualizing the feature map learned from the attention modules.

## Proposed Model

<center>
  <img src="https://github.com/Jimut123/bmsan/raw/main/docs/DRRMSAN_Model.png">
</center>



## Modified U-Net

<center>
  <img src="https://github.com/Jimut123/bmsan/raw/main/docs/Modified_UNet.png">
</center>



## Proposed Attention Module 1

<center>
  <img src="https://github.com/Jimut123/bmsan/raw/main/docs/Attention_Gate_1.png">
</center>



## Proposed Attention Module 2

<center>
  <img src="https://github.com/Jimut123/bmsan/raw/main/docs/Attention_Gate_2.png">
</center>



## Proposed Attention Module 3

<center>
  <img src="https://github.com/Jimut123/bmsan/raw/main/docs/Attention_Gate_3.png">
</center>


 ***
## Results

<center>
  <img src="https://github.com/Jimut123/bmsan/raw/main/docs/table_1.png">
</center>




<center>
  <img src="https://github.com/Jimut123/bmsan/raw/main/docs/table_3.png">
</center>



<center>
  <img src="https://github.com/Jimut123/bmsan/raw/main/docs/table_4.png">
</center>



<center>
  <img src="https://github.com/Jimut123/bmsan/raw/main/docs/table_5.png">
</center>


<center>
  <img src="https://github.com/Jimut123/bmsan/raw/main/docs/table_2.png">
</center>


<center>
  <img src="https://github.com/Jimut123/bmsan/raw/main/docs/fig_10.png">
</center>


## Dependencies
For installing the required dependencies, please use the following packages as mentioned in [dependencies](nvdia_env.txt).
Note the code might not work on the current versions of the libraries, hence please follow the dependencies.

## Running the code

For running the code, we need to download the following datasets and keep them in proper folder.

#### Download the datasets

Use the figshare mirrors: [mirror1](https://figshare.com/articles/dataset/DATA_BMSAN_ISIC/21224762) and [mirror2](https://figshare.com/articles/dataset/DATA_BMSAN/21224756)

After downloading the datasets, unzip them via ``unzip`` command (in linux) into the respective directories.
For example if it is brain.zip (i.e., Brain MRI dataset) then we need to place it in the BrainMRI
folder. Each of the respective folders should have their respective datasets. Example of a folder directory may
be shown here:

```
-- BrainMRI
  |  K_Fold_drrmsan
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
This will collect the dataset for the Bayesian Optimisation part and then we may run the files present in the ``Final`` folder to get the
best possible weighted mask combination. We again use this weights for training segmentation on the best possible MSAN model.


## Contribution

Please check [CONTRIBUTING.md](https://github.com/Jimut123/drrmsan/blob/main/CONTRIBUTING.md)



## Acknowledgements

The [authors](https://github.com/Jimut123/drrmsan/blob/main/AUTHORS.md) are thankful to **Swathy Prabhu Mj** for arranging Asus RTX 2080 Ti (12 GB) and Quadro GV100 (32 GB) GPUs with 64 GB RAM,  to hasten the research. The first author is thankful to **Br. Tamal Maharaj** and **Dr. Jadab Kumar Pal** for their suggestions. The Authors would also like to thank [lixiaolei1982](https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-) and [nibtehaz](https://github.com/nibtehaz/MultiResUNet) for their implementations of state-of-the art models. 


# BibTeX and citations

```
@article{doi:10.1080/13682199.2023.2174657,
  author = {Pal,Jimut Bahan and Mj,Dripta},
  title = {Improving Multi Scale Attention Networks: Bayesian Optimization for Segmenting medical images},
  journal = {The Imaging Science Journal},
  year = {2023},
  doi = {10.1080/13682199.2023.2174657},
}
```
***
