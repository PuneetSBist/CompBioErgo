# ERGO
This project uses the original ERGO model(https://github.com/louzounlab/ERGO) for predicting TCR-peptide binding.

## Requirements
```text
python 3.6
pytorch
numpy 
scikit-learn 
```

## Model Training
The main module for training is `proj_ergo.py`.
Check out the hardcoded values in the command arguments 
Use dataset in proj_data for tcr_split or epi_split

## References
1. [Springer I, Besser H, Tickotsky-Moskovitz N, Dvorkin S and Louzoun Y (2020)
Prediction of Specific TCR-Peptide Binding From Large Dictionaries of TCR-Peptide Pairs.
Front. Immunol. 11:1803. doi: 10.3389/fimmu.2020.01803](https://www.frontiersin.org/articles/10.3389/fimmu.2020.01803/full)
