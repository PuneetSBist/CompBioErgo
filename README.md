# ERGO
This project uses the original ERGO model(https://github.com/louzounlab/ERGO) for predicting TCR-peptide binding with fe modifications in parameters as well difefrent model modification

## Requirements
```text
python 3.6
pytorch
numpy 
scikit-learn 
```

## Model Training
The main module for training is `proj_ergo.py`.
python proj_ergo.py train lstm cuda --train_data_path <train csv> --test_data_path <test csv>
It will automatically create a new directory(with timestamp) and store the training and validation metrics as well as best performing model
Use dataset in proj_data for tcr_split or epi_split

## Model Prediction
python proj_ergo.py train lstm cuda --test_data_path <test csv> --model_file <saved model filed during training>


## References
1. [Springer I, Besser H, Tickotsky-Moskovitz N, Dvorkin S and Louzoun Y (2020)
Prediction of Specific TCR-Peptide Binding From Large Dictionaries of TCR-Peptide Pairs.
Front. Immunol. 11:1803. doi: 10.3389/fimmu.2020.01803](https://www.frontiersin.org/articles/10.3389/fimmu.2020.01803/full)
