import warnings
import random
import os
import datetime
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    accuracy_score,
    classification_report,
)
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch, butter, filtfilt
from joblib import Parallel, delayed
import neurokit2 as nk

from pycaret.classification import setup, compare_models, tune_model, finalize_model, predict_model, save_model, load_model

from utils import lead_label_generation, normalize_signal_raw, normalize_signal_features, Chunking, extract_global_features_vectorized, feature_names


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def train(LABEL_CSV_PATH = 'data/label_data_v2.csv',
          SIGNAL_PICKLE_PATH = 'data/Signal_Train.pkl',     
          MODEL_SAVE_DIR=ROOT/'model',
          CHUNK_LENGTH_SEC = 6,
          STEP_LENGTH_SEC = 2,
          RANDOM_STATE = 2024,
          chunking = False,
          GPU = False):
    # -----------------------------------------
    # 1. Load Dataset
    # -----------------------------------------
    print('Loading dataset...')
    label_data = pd.read_csv(LABEL_CSV_PATH)
    signal_df = pd.read_pickle(SIGNAL_PICKLE_PATH)

    # Filter mismatched labels
    label_data_mismatched = label_data[label_data['Label_changed'] == 1]
    label_data = label_data[label_data['Label_changed'] == 0]
    signal_df_mismatched = [signal_df[index] for index in label_data_mismatched['Index']]
    signal_df = [signal_df[index] for index in label_data['Index']]

    # Label (global) and signal data
    label_data.reset_index(inplace=True)
    signal_data = np.array(signal_df)  # Shape: (num_samples, seq_length, num_leads)

    # Label and signal data including mismatched ones
    signal_data_mismatched = np.array(signal_df_mismatched)
    label_data_mismatched.reset_index(inplace=True)

    # -----------------------------------------
    # 2-1. Preprocessing - Basics
    # -----------------------------------------
    # Create a mapping from artifact types to integer labels
    artifact_types_list = ["NCR", "SF", "HF", "BW", "MA"]  # Sort from low-priority(NCR) to high-priority(MA)
    artifact_type_map = {artifact: idx + 1 for idx, artifact in enumerate(artifact_types_list)}  # Start from 1
    no_artifact_label = 0  # Label for "no artifact"
    
    # Lead-wise artifact type label generation for every 0.2-second interval
    lead_labels = lead_label_generation(signal_data, label_data, artifact_type_map, no_artifact_label)  # Shape: (samples, time_steps, leads)
                            
    # Global binary labels for artifact detection
    global_labels = label_data['New_label'].values.astype(np.float32)  # Binary artifact labels
    
    # Apply minmax normalization to the entire dataset
    signal_data_normalized = np.array([normalize_signal_raw(sig) for sig in signal_data])
    
    # Split data into train+validation and test sets
    signals_train_val, signals_test, global_labels_train_val, global_labels_test, lead_labels_train_val, lead_labels_test = train_test_split(
        signal_data_normalized,
        global_labels,
        lead_labels,
        test_size=0.2,  # 20% of the data reserved for the test set
        stratify=global_labels,  # Ensuring class distribution is maintained
        random_state=RANDOM_STATE
    )
    print(f"Train/Validation samples: {len(signals_train_val)}")
    print(f"Test samples: {len(signals_test)}")
    
    # Prepare mismatched data test set
    signal_data_mismatched_processed = np.array([normalize_signal_raw(sig) for sig in signal_data_mismatched])
    global_labels_mismatched = label_data_mismatched['Original_label'].values.astype(np.float32)
    lead_labels_mismatched = np.zeros((len(signal_data_mismatched_processed), 12))

    # Sample 20%, same as the matched data
    _s, signals_mismatched_test, _g, global_labels_mismatched_test, _l, lead_labels_mismatched_test = train_test_split(
        signal_data_mismatched_processed,
        global_labels_mismatched,
        lead_labels_mismatched,
        test_size=0.2,  # same as matched data
        random_state=RANDOM_STATE
    )
    print(f"Mismatched test samples: {signals_mismatched_test.shape[0]}")
    
    if chunking:
        # -----------------------------------------
        # 2-2. Preprocessing - Chunking
        # -----------------------------------------
        # Set chunk size and step: 6-sec legnth and 2-sec step, total 3 chunks for each signal (default)
        chunk_size = CHUNK_LENGTH_SEC*500  # 6-sec length (default)
        step = STEP_LENGTH_SEC*500      # 2-sec step (default)

        signals_chunks_tv, global_label_chunks_tv, lead_labels_chunks_tv = Chunking(signals_train_val, global_labels_train_val, lead_labels_train_val, chunk_size, step)
        signals_chunks_test, global_labels_chunks_test, lead_labels_chunks_test = Chunking(signals_test, global_labels_test, lead_labels_test, chunk_size, step)
        signals_mismatched_chunks, global_labels_mismatched_chunks, lead_labels_mismatched_chunks = Chunking(signals_mismatched_test, global_labels_mismatched_test, lead_labels_mismatched_test, chunk_size, step)

        ### Global label is not a global label for the entire ECG anymore. It is a global label for the specific chunk.
        print(f"Train/Validation chunk samples: {signals_chunks_tv.shape[0]}")
        print(f"Test chunk samples: {signals_chunks_test.shape[0]}")
        print(f"Mismatched chunk test samples: {features_mismatched_chunks.shape[0]}")

        # Shuffle the train/validation chunks
        indices = np.arange(signals_chunks_tv.shape[0])
        shuffled_indices = np.random.permutation(indices)

        signals_chunks_tv = signals_chunks_tv[shuffled_indices]
        global_labels_chunks_tv = global_labels_chunks_tv[shuffled_indices]
        lead_labels_chunks_tv = lead_labels_chunks_tv[shuffled_indices]
        
        # Feature extraction
        features_chunks_tv = extract_global_features_vectorized(signals_chunks_tv, fs=500, num_leads=12)
        features_chunks_test = extract_global_features_vectorized(signals_chunks_test, fs=500, num_leads=12)
        features_mismatched_chunks_test = extract_global_features_vectorized(signals_mismatched_chunks, fs=500, num_leads=12)
        # Total number of features = 30*12 + 2 + 10 = 372
        print(f"Total number of features: {features_chunks_tv.shape[1]}")

        # Normalize features using StandardScaler
        features_chunks_tv = normalize_signal_features(features_chunks_tv)
        features_chunks_test = normalize_signal_features(features_chunks_test)
        features_mismatched_chunks = normalize_signal_features(features_mismatched_chunks)
    
    # -----------------------------------------
    # 2-2. Preprocessing - Feature extraction
    # -----------------------------------------
    # Make feature datasets
    features_tv = extract_global_features_vectorized(signals_train_val, fs=500, num_leads=12)
    features_test = extract_global_features_vectorized(signals_test, fs=500, num_leads=12)
    features_mismatched_test = extract_global_features_vectorized(signals_mismatched_test, fs=500, num_leads=12)

    # Total number of features = 30*12 + 2 + 10 = 372
    print(f"Total number of features: {features_tv.shape[1]}")
    
    # Normalize features
    features_tv = normalize_signal_features(features_tv)
    features_test = normalize_signal_features(features_test)
    features_mismatched_test = normalize_signal_features(features_mismatched_test)
    
    # -----------------------------------------
    # 3. Training
    # -----------------------------------------
    # Data prep for training (using pycaret)
    # Train-val
    train_X = features_tv
    train_y = global_labels_train_val
    train_data = pd.DataFrame(train_X.copy())
    train_data['target'] = train_y
    # Test
    test_X = features_test
    test_y = global_labels_test
    test_data = pd.DataFrame(test_X.copy())
    test_data['target'] = test_y
    # Name the features of dataset
    train_data.columns = feature_names()
    test_data.columns = feature_names()
    
    # Setup pycaret settings (using SMOTE)
    exp = setup(train_data, target='target', normalize=False, fix_imbalance=True, session_id=RANDOM_STATE, use_gpu=GPU)
    
    # Train, tune, and finalize the ET classifier
    RF_model = compare_models(include = ['rf'])
    tuned_model = tune_model(RF_model)
    final_model = finalize_model(tuned_model)
    
    # Save the trained model
    # Model save directory creation
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(str(MODEL_SAVE_DIR), exist_ok=True)
    save_model(final_model, str(MODEL_SAVE_DIR / 'RF_model_{}'.format(timestamp)))
    
    return final_model, test_data, test_y

def test(test_model, test_data, test_y):
    # Make predictions on the test set
    predictions = predict_model(test_model, raw_score=True, data=test_data, verbose=False, round=6)
    y_pred = predictions['prediction_label']
    y_pred_prob = predictions['prediction_score_1']
    y_true = test_y

    # Calculate performance metrics for the whole ECG (F1 score, AUROC, and MCC)
    f1 = f1_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auroc = float('nan')  # Handle cases where AUROC cannot be computed
    mcc = matthews_corrcoef(y_true, y_pred)

    # Calculate the CPI score
    cpi = 0.25 * f1 + 0.25 * auroc + 0.5 * mcc

    print("### Predictions for the whole ECG ###")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"CPI Score: {cpi:.4f}")
    
def private(final_model, private_signal, cutoff=75):
    signal_data = np.array(private_signal)
    print(f"Private data samples: {signal_data.shape[0]}")
    
    # Normalization
    signal_data_normalized = np.array([normalize_signal_raw(sig) for sig in signal_data])
    
    # Feature extraction
    features_data = extract_global_features_vectorized(signal_data_normalized, fs=500, num_leads=12)

    # Total number of features = 30*12 + 2 + 10 = 372
    print(f"Total number of features: {features_data.shape[1]}")
    
    # Normalize features
    features_data = normalize_signal_features(features_data)
    
    # Make predictions
    test_data = pd.DataFrame(features_data.copy())
    test_data.columns = feature_names(target_include=False)
    predictions = predict_model(final_model, raw_score=True, data=test_data, verbose=False, round=6)
    y_pred_prob = predictions['prediction_score_1']
    
    # Cut the predictions for the top 75 entries.
    sorted_prob = np.sort(y_pred_prob)
    cutoff_index = len(sorted_prob) - cutoff  # The index of the cutoff value
    cutoff_value = (sorted_prob[cutoff_index] + sorted_prob[cutoff_index - 1]) / 2
    y_pred = (y_pred_prob >= cutoff_value).astype(int)
    
    # Save predictions as a csv file.
    os.makedirs(str(ROOT/'out'), exist_ok=True)
    out = pd.DataFrame({'Probabilities': y_pred_prob, 'Predictions': y_pred})
    out.to_csv('out/Purewave_1014_Private_Possibility.csv', index=True)
    print("Successfully saved predictions.")


def run(TEST_DATA_PATH=ROOT/'data/test/test_1013.csv',
        PRIVATE_SIGNAL_PATH=ROOT/'data/Signal_Test_Private.pkl',
        LABEL_CSV_PATH=ROOT/'data/label_data_v2.csv',
        SIGNAL_PICKLE_PATH=ROOT/'data/Signal_Train.pkl', 
        MODEL_SAVE_DIR=ROOT/'models',
        MODEL_PATH=ROOT/'models/rf_total',
        chunking=False,
        RANDOM_STATE = 2024,
        GPU = False,
        MODE = 'test'
        ):
    
    # Ignore warnings
    warnings.filterwarnings("ignore")

    # if GPU:
    #     # Check if CUDA is available, else use CPU
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     print(f"Using device: {device}")
    
    # # Set random seed for PyTorch
    # torch.manual_seed(RANDOM_STATE)
    # # If using CUDA (GPU), set the random seed for CUDA as well
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(RANDOM_STATE)
    #     torch.cuda.manual_seed_all(RANDOM_STATE)  # If using multi-GPU
    # Set seed for NumPy (if used)
    np.random.seed(RANDOM_STATE)
    # Set seed for Python's built-in random module (if used)
    random.seed(RANDOM_STATE)
    
    if MODE == "train":
        final_model, test_data, test_y = train(LABEL_CSV_PATH, SIGNAL_PICKLE_PATH, MODEL_SAVE_DIR, chunking, RANDOM_STATE = RANDOM_STATE, GPU = GPU)
        test(final_model, test_data, test_y)
        
    elif MODE == "test": # MUST use the model not trained on the test data (use <model>_tv).
        loaded_model = load_model(MODEL_PATH)
        test_data = pd.read_csv(TEST_DATA_PATH)
        test_y = test_data.to_numpy()[:, -1]
        test(loaded_model, test_data, test_y)
        
    elif MODE == "private": # Must use the whole model (use <model>_total).
        loaded_model = load_model(MODEL_PATH)
        private_signal = pd.read_pickle(PRIVATE_SIGNAL_PATH)         
        private(loaded_model, private_signal, cutoff=75)
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--TEST_DATA_PATH', default=ROOT/'data/test_1013.csv', help='Path for test data (featurized)')
    parser.add_argument('--PRIVATE_SIGNAL_PATH', default=ROOT/'data/Signal_Test_Private.pkl', help='Path for private signal data')
    parser.add_argument('--LABEL_CSV_PATH', default=ROOT/'data/label_data_v2.csv', help='Path for train label (leadwise label)')
    parser.add_argument('--SIGNAL_PICKLE_PATH', default=ROOT/'data/Signal_Train.pkl', help='Path for train signal')
    parser.add_argument('--MODEL_SAVE_DIR', default=ROOT/'models', help='Path for model saving')
    parser.add_argument('--MODEL_PATH', default=ROOT/'models/rf_total', help='Model to load')
    parser.add_argument('--chunking', default=False, help='use data chunking')
    parser.add_argument('--RANDOM_STATE', default=2024, help='Random state. Default is 2024')
    parser.add_argument('--GPU', type=lambda x: x.lower() == 'true', default=False, help='If True, Use GPU (defaults to False)')
    parser.add_argument('--MODE', default='test', help='train: training using train dataset, test: testing using train dataset (20%), private: make predictions for private test data.')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)