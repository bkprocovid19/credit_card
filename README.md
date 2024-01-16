# AntiFraud
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/fraud-detection-on-amazon-fraud)](https://paperswithcode.com/sota/fraud-detection-on-amazon-fraud?p=semi-supervised-credit-card-fraud-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/node-classification-on-amazon-fraud)](https://paperswithcode.com/sota/node-classification-on-amazon-fraud?p=semi-supervised-credit-card-fraud-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/fraud-detection-on-yelp-fraud)](https://paperswithcode.com/sota/fraud-detection-on-yelp-fraud?p=semi-supervised-credit-card-fraud-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/node-classification-on-yelpchi)](https://paperswithcode.com/sota/node-classification-on-yelpchi?p=semi-supervised-credit-card-fraud-detection)

A Financial Fraud Detection Framework.

Source codes implementation of papers:
- `MCNN`: Credit card fraud detection using convolutional neural networks, in ICONIP 2016. 
- `STAN`: Spatio-temporal attention-based neural network for credit card fraud detection, in AAAI2020
- `STAGN`: Graph Neural Network for Fraud Detection via Spatial-temporal Attention, in TKDE2020
- `GTAN`: Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation, in AAAI2023
- `RGTAN`: Enhancing Attribute-driven Fraud Detection with Risk-aware Graph Representation, 



## Usage

### Data processing
1. Run `unzip /data/Amazon.zip` and `unzip /data/YelpChi.zip` to unzip the datasets; 
2. Run `python feature_engineering/data_process.py
`
to pre-process all datasets needed in this repo.

### Training & Evalutaion
<!-- 
To use fraud detection baselines including GBDT, LSTM, etc., simply run

```
python main.py --method LSTM
python main.py  --method GBDT
```
You may change relevant configurations in `config/base_cfg.yaml`. -->

To test implementations of `MCNN`, `STAN` and `STAGN`, run
```
python main.py --method mcnn
python main.py --method stan
python main.py --method stagn
```
Configuration files can be found in `config/mcnn_cfg.yaml`, `config/stan_cfg.yaml` and `config/stagn_cfg.yaml`, respectively.

Models in `GTAN` and `RGTAN` can be run via:
```
python main.py --method gtan
python main.py --method rgtan
```
For specification of hyperparameters, please refer to `config/gtan_cfg.yaml` and `config/rgtan_cfg.yaml`.



### Data Description

There are three datasets, YelpChi, Amazon and S-FFSD, utilized for model experiments in this repository.

<!-- YelpChi and Amazon can be downloaded from [here](https://github.com/YingtongDou/CARE-GNN/tree/master/data) or [dgl.data.FraudDataset](https://docs.dgl.ai/api/python/dgl.data.html#fraud-dataset).

Put them in `/data` directory and run `unzip /data/Amazon.zip` and `unzip /data/YelpChi.zip` to unzip the datasets. -->

YelpChi and Amazon datasets are from [CARE-GNN](https://dl.acm.org/doi/abs/10.1145/3340531.3411903), whose original source data can be found in [this repository](https://github.com/YingtongDou/CARE-GNN/tree/master/data).

S-FFSD is a simulated & small version of finacial fraud semi-supervised dataset. Description of S-FFSD are listed as follows:
|Name|Type|Range|Note|
|--|--|--|--|
|Time|np.int32|from $\mathbf{0}$ to $\mathbf{N}$|$\mathbf{N}$ denotes the number of trasactions.  |
|Source|string|from $\mathbf{S_0}$ to $\mathbf{S}_{ns}$|$ns$ denotes the number of transaction senders.|
|Target|string|from $\mathbf{T_0}$  to $\mathbf{T}_{nt}$ | $nt$ denotes the number of transaction reveicers.|
|Amount|np.float32|from **0.00** to **np.inf**|The amount of each transaction. |
|Location|string|from $\mathbf{L_0}$  to $\mathbf{L}_{nl}$ |$nl$ denotes the number of transacation locations.|
|Type|string|from $\mathbf{TP_0}$ to $\mathbf{TP}_{np}$|$np$ denotes the number of different transaction types. |
|Labels|np.int32|from **0** to **2**|**2** denotes **unlabeled**||


> We are looking for interesting public datasets! If you have any suggestions, please let us know!

## Test Result
The performance of five models tested on three datasets are listed as follows:
| |YelpChi| | |Amazon| | |S-FFSD| | |
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
| |AUC|F1|AP|AUC|F1|AP|AUC|F1|AP|
|MCNN||- | -| -| -| -|0.7129|0.6861|0.3309|
|STAN|- |- | -| -| -| -|0.7422|0.6698|0.3324|
|STAGN|- |- | -| -| -| -|0.7659|0.6852|0.3599|
|GTAN|0.9241|0.7988|0.7513|0.9630|0.9213|0.8838|0.8286|0.7336|0.6585|
|RGTAN|0.9498|0.8492|0.8241|0.9750|0.9200|0.8926|0.8461|0.7513|0.6939|

> `MCNN`, `STAN` and `STAGN` are presently not applicable to YelpChi and Amazon datasets.

## Repo Structure
The repository is organized as follows:
- `models/`: the pre-trained models for each method. The readers could either train the models by themselves or directly use our pre-trained models;
- `data/`: dataset files;
- `config/`: configuration files for different models;
- `feature_engineering/`: data processing;
- `methods/`: implementations of models;
- `main.py`: organize all models;
- `requirements.txt`: package dependencies;

    
## Requirements
```
python           3.7
scikit-learn     1.0.2
pandas           1.3.5
numpy            1.21.6
networkx         2.6.3
scipy            1.7.3
torch            1.12.1+cu113
dgl-cu113        0.8.1
absl-py==2.0.0
alabaster==0.7.13
annotated-types==0.6.0
astunparse==1.6.3
autopep8==2.0.4
Babel==2.13.1
beautifulsoup4==4.12.2
blinker==1.7.0
boto3==1.33.8
botocore==1.33.8
bs4==0.0.1
cachetools==5.3.2
catboost==1.2.2
certifi==2023.7.22
charset-normalizer==3.3.2
click==8.1.7
colorama==0.4.6
contourpy==1.1.1
cycler==0.11.0
dgl==1.1.2
dglgo==0.0.2
dockerfile-parse==2.0.1
docutils==0.20.1
et-xmlfile==1.1.0
filelock==3.13.1
Flask==3.0.0
Flask-SQLAlchemy==3.1.1
flatbuffers==23.5.26
fonttools==4.42.1
fsspec==2023.10.0
gast==0.5.4
glob2==0.7
google-auth==2.23.4
google-auth-oauthlib==1.1.0
google-pasta==0.2.0
graphviz==0.20.1
greenlet==3.0.1
grpcio==1.59.3
h5py==3.10.0
idna==3.4
imagesize==1.4.1
imbalanced-learn==0.11.0
isort==5.12.0
itsdangerous==2.1.2
Jinja2==3.1.2
jmespath==1.0.1
joblib==1.3.2
keras==2.15.0
kiwisolver==1.4.5
libclang==16.0.6
littleutils==0.2.2
Markdown==3.5.1
MarkupSafe==2.1.3
matplotlib==3.8.0
ml-dtypes==0.2.0
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.0
numpydoc==1.6.0
oauthlib==3.2.2
ogb==1.3.6
openpyxl==3.1.2
opt-einsum==3.3.0
outdated==0.2.2
packaging==23.1
pandas==2.1.1
Pillow==10.0.1
pipdeptree==2.13.2
pipz==0.9.875
plotly==5.18.0
protobuf==4.23.4
psutil==5.9.6
pyasn1==0.5.0
pyasn1-modules==0.3.0
pycodestyle==2.11.1
pydantic==2.5.1
pydantic_core==2.14.3
Pygments==2.16.1
pyparsing==3.1.1
python-dateutil==2.8.2
pytz==2023.3.post1
PyYAML==6.0.1
rdkit-pypi==2022.9.5
requests==2.31.0
requests-oauthlib==1.3.1
rsa==4.9
ruamel.yaml==0.18.5
ruamel.yaml.clib==0.2.8
s3transfer==0.8.2
scikit-learn==1.3.2
scipy==1.11.3
seaborn==0.13.0
six==1.16.0
snowballstemmer==2.2.0
soupsieve==2.5
Sphinx==7.2.6
sphinxcontrib-applehelp==1.0.7
sphinxcontrib-devhelp==1.0.5
sphinxcontrib-htmlhelp==2.0.4
sphinxcontrib-jsmath==1.0.1
sphinxcontrib-qthelp==1.0.6
sphinxcontrib-serializinghtml==1.1.9
SQLAlchemy==2.0.23
sympy==1.12
tabulate==0.9.0
tenacity==8.2.3
tensorboard==2.15.1
tensorboard-data-server==0.7.2
tensorflow==2.15.0
tensorflow-estimator==2.15.0
tensorflow-intel==2.15.0
tensorflow-io-gcs-filesystem==0.31.0
termcolor==2.3.0
threadpoolctl==3.2.0
torch==2.1.0
tqdm==4.66.1
typer==0.9.0
typing_extensions==4.8.0
tzdata==2023.3
urllib3==2.0.7
Werkzeug==3.0.1
wrapt==1.14.1
```

## Run 
To run the application, compile file app.py


## Citing

If you find *Antifraud* is useful for your research, please consider citing the following papers:

    @inproceedings{Xiang2023SemiSupervisedCC,
        title={Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation},
        author={Sheng Xiang and Mingzhi Zhu and Dawei Cheng and Enxia Li and Ruihui Zhao and Yi Ouyang and Ling Chen and Yefeng Zheng},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        year={2023}
    }
    @article{cheng2020graph,
        title={Graph Neural Network for Fraud Detection via Spatial-temporal Attention},
        author={Cheng, Dawei and Wang, Xiaoyang and Zhang, Ying and Zhang, Liqing},
        journal={IEEE Transactions on Knowledge and Data Engineering},
        year={2020},
        publisher={IEEE}
    }
    @inproceedings{cheng2020spatio,
        title={Spatio-temporal attention-based neural network for credit card fraud detection},
        author={Cheng, Dawei and Xiang, Sheng and Shang, Chencheng and Zhang, Yiyi and Yang, Fangzhou and Zhang, Liqing},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={34},
        number={01},
        pages={362--369},
        year={2020}
    }
    @inproceedings{fu2016credit,
        title={Credit card fraud detection using convolutional neural networks},
        author={Fu, Kang and Cheng, Dawei and Tu, Yi and Zhang, Liqing},
        booktitle={International Conference on Neural Information Processing},
        pages={483--490},
        year={2016},
        organization={Springer}
    }
