# DP-RTF-Learning
A python implementation of “**<a href="https://ieeexplore.ieee.org/document/9582746" target="_blank">Learning Deep Direct-Path Relative Transfer Function for Binaural Sound Source Localization</a>**”, IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), 2021.
+ **Contributions** 
  - A DP-RTF learning framework that embeds the sensor signals to a low-dimensional localization feature space is designed, which disentangles the localization cues from other factors including source signals, noise, reverberation, etc. 
    - **a Novel DP-RTF Learning Network**
    - **leveraging Monaural Speech Enhancement to Improve the Robustness of DP-RTF Estimation**
    - **generalization to Unseen Binaural Configurations**
    <div align=center>
    <img src=https://user-images.githubusercontent.com/74909427/218234395-fabb10d6-9463-4455-8214-06ad186bd8ed.png width=94% />
    </div>
  - The DP-RTF learning based localization method takes full use of the spatial and spectral cues, which is demonstrated to perform better than several other methods on both simulated and real-world data in the noisy and reverberant environment.
    <div align=center>
    <img src=https://user-images.githubusercontent.com/74909427/218235435-19332838-9948-4c97-ab0d-752aec4cd6fd.png width=41% />
    <img src=https://user-images.githubusercontent.com/74909427/218235464-b007df3e-2cf0-45d8-a8e4-e46c2ad8829e.png width=51% />
    </div>
    <div align=center>
    <img src=https://user-images.githubusercontent.com/74909427/218235769-df0bc161-f73e-4955-8f5c-77d8228ef498.png width=95% />
    </div>
    
## Datasets
+ **Head-related impulse responses (HRIRs)**: from <a href="https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/" target="_blank">CIPIC database</a> 
+ **Binaural room impulse responses (BRIRs)**: generated by <a href="https://github.com/bingo-todd/Roomsim_Campbell" target="_blank">Roomsim toolbox</a>
+ **TIMIT dataset** 
+ **Diffuse noise**: generated by <a href="https://github.com/ehabets/ANF-Generator" target="_blank">arbitrary noise field generator</a> with noise signals from <a href="http://spib.linse.ufsc.br/noise.html" target="_blank">NOISEX-92 database</a>
  
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
If you find our work useful in your research, please consider citing:
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

## Licence
MIT
