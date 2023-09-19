# NARE-T2
## Improved NARE with transformer-based model for session-based recommendation
This is the code for improved version of the paper "Improved Session-based Recommendation using Graph-based Item Embedding" eCom, SIGIR 2020. 
In this version, recommendation model based on LSTM with attention is replaced by transformer.

## Datasets
-> The following datasets are used in our experiments. 
YOOCHOOSE: http://2015.recsyschallenge.com/challenge.html or https://www.kaggle.com/chadgostopp/recsys-challenge-2015
Download yoochoose-clicks.dat from the above link and save the file in Datasets folder 

DIGINETICA: http://cikm2016.cs.iupui.edu/cikm-cup or https://competitions.codalab.org/competitions/11161
Download train-item-views.csv from the above link and save the file in Datasets folder
-> Run preprocess.py with dataset name as argument to generate train and test files. (dataset name: diginetica/yoochoose/sample)
Example: python preprocess.py --dataset=yoochoose
-> The required files are already available in the appropriate folders. 
## Compute TransE Embedding 
-> Run create_transe_emb_diginetica.py to generate KG triples and compute transe embedding on diginetica dataset. Parameters are optimized based on validation data. The best values are given in the code.
-> Run create_transe_emb_yoochoose.py to generate KG triples and compute transe embedding on diginetica dataset. Parameters are optimized based on validation data. The best values are given in the code.
-> The required files are already available in the appropriate folders. 
## Run the model
-> python main.py main.py [--dataset DATASET] [--batchSize BATCHSIZE] [--dim_feedforward FEED_FORWARD_UNITS]
               [--heads HEADS] [--epochs EPOCH] [--lr LR] [--pos_encoding POS_ENCODING]
               [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--num_layers encoder-layers]
               [--dropout encoder_dropout] [--patience PATIENCE]
               [--validation] [--valid_portion VALID_PORTION] 

-> Mandatory Arguments 
  1) --dataset: name of the dataset
  2) --pos_encoding: style of positional encoding (0 for fixed and 1 for trainable)
-> Best Results are obtained on Yoochoose1_64 with the following parameters (Remaining are default)
python main.py --dataset yoochoose1_64 --dim_feedforward 64 --num_layers 2 --pos_encoding 0 --dropout 0.4
-> Best Results are obtained on Diginetica with the following parameters.
 python main.py --dataset diginetica --dim_feedforward 512 --num_layers 4 --pos_encoding 1 --dropout 0.5 --lr_dc_step 1
-> Install MADGRAD optimizer (https://github.com/facebookresearch/madgrad)
## Requirements
-> TransE embedding is implemented using Ampligraph (https://docs.ampligraph.org/en/1.4.0/generated/ampligraph.latent_features.TransE.html)
-> ampligraph 1.4.0
-> tensorflow or tensorflow-gpu '>=1.15.2,<2.0.0'
-> Python 3
-> Pytorch 2.0.1
## Citation
Please cite our original paper if you use our code 
@article{srilakshmi2020improved,
  title={Improved Session based Recommendation using Graph-based Item Embedding},
  author={Srilakshmi, Madiraju and Chowdhury, Gourab and Sarkar, Sudeshna},
  year={2020}
}
