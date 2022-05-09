# DP-RTF-Learning
A python implementation of “**<a href="https://ieeexplore.ieee.org/document/9582746" target="_blank">Learning Deep Direct-Path Relative Transfer Function for Binaural Sound Source Localization</a>**”, IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), 2021.



## Datasets
+ **Head-related impulse responses (HRIRs)**: from <a href="https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/" target="_blank">CIPIC</a> database
+ **Binaural room impulse responses (BRIRs)**: generated by Roomsim toolbox 
+ **TIMIT dataset** 
+ **Diffuse noise**: generated by <a href="https://github.com/ehabets/ANF-Generator" target="_blank">arbitrary noise field generator</a> with noise signals from NOISEX-92 database 

  
## Quick start
+ **Preparation**
  - Add soft link of "common" file to "DPRTF" file
    ```
    ln -s [original path] [target path]
    ```
  - Generate the lists of source signals and BRIRs, direct-path relative tranfer functions (DP-RTFs), room acoustic settings, and sensor signals for training, validation and test stages. 
    ```
    python -m common.getData --stage [*] --data [*] 
    ```

+ **Training**
  ```
  python run.py --gpu-id [*]
  ```
+ **Test**
  ```
  python run.py --gpu-id [*] --test
  ```
+ **Pretrained models**
  - exp/00000000/model_12.pth: trained with fixed data
  - exp/00000001/model_52.pth: trained with random data (generated on-the-fly)

## Citation
```
@article{yang2021dprtf,
    Author = "Bing Yang and Hong Liu and Xiaofei Li",
    Title = "Learning deep direct-path relative transfer function for binaural sound source localization",
    Journal = "{IEEE/ACM} Transactions on Audio, Speech, and Language Processing (TASLP)",
    Volume = {29},	
    Pages = {3491-3503},
    Year = {2021}}
```
```
@InProceedings{yang2021dprtf1,
    author = "Bing Yang and Xiaofei Li and Hong Liu",
    title = "Supervised direct-path relative transfer function learning for binaural sound source localization",
    booktitle = "Proceedings of {IEEE} International Conference on Acoustics, Speech and Signal Processing (ICASSP)",
    year = "2021",
    pages = "825-829"}
```
```
@article{yang2021dprtf2,
    Author = "Bing Yang and Runwei Ding and Yutong Ban and Xiaofei Li and Hong Liu",
    Title = "Enhancing Direct-Path Relative Transfer Function Using Deep Neural Network for Robust Sound Source Localization",
    Journal = "{CAAI} Transactions on Intelligence Technology",
    Pages = {1-9},
    Year = {2021}}
```

## Licence
MIT
