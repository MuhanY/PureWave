# PureWave
ECG artifact detection project


## plotECG.py
ECG visulization<br>
Usage: python plotECG.py --train_dir [training data directory]


## main.py
Artifact detection model training and testing
1. Environment<br>
Ubuntu 22.04.3 LST (64-bit) with kernel version 5.4.0-137-generic, GCC 11.4.0, Python 3.9.20
joblib 1.3.2, matplotlib 3.7.5, neurokit2 0.2.10, numpy 1.26.4, pandas 2.1.4, pycaret 3.3.2, scikit_learn 1.4.2, scipy 1.11.4
2. Shell commands<br>
conda create -n purewave python==3.9.20
conda activate purewave
python -m pip install -r requirements.txt
Model train: python main.py --MODE train
Model test: python main.py --MODE test --MODEL_PATH models/rf_tv
Model (private) test: python main.py --MODE private
