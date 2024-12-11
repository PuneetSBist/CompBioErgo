module load mamba/latest
source activate myenv

python /home/pbist/AlgoCompBio/Ergo_proj/CompBioErgo/proj_ergo.py train cuda --train_data_path '/home/pbist/AlgoCompBio/data/BAP/tcr_split/train.csv' --test_data_path '/home/pbist/AlgoCompBio/data/BAP/tcr_split/test.csv' --model_type rf --temp_model_path /home/pbist/AlgoCompBio/Ergo_proj/CompBioErgo/RunData/TCR_split_32Batch_Drop20/lstm_psb_checkpoints_best

python C:\Users\bistp\Downloads\CLass\CompBio\Project\Python\CompBioErgo\proj_ergo.py train cuda --kfold 5 --train_data_path C:\Users\bistp\Downloads\CLass\CompBio\Project\Python\CompBioErgo\proj_data\BAP\tcr_split\train.csv --test_data_path C:\Users\bistp\Downloads\CLass\CompBio\Project\Python\CompBioErgo\proj_data\BAP\tcr_split\test.csv --roc_file 'roc_file'
