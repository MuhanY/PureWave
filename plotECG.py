import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def run(train_dir=ROOT / 'K-MEDICON 2024_SUB1_TRAINING SET', 
        save_dir=ROOT / 'images2',
        dpi=450,  # dpi of plotted images
        color='grey', # ECG line color
):
    train_dir = Path(train_dir)
    save_dir = Path(save_dir)
    
    # Load the signal and target data
    with open(str(train_dir / 'Signal_Train.pkl'), 'rb') as f1:
        signal_data = pickle.load(f1)
    with open(str(train_dir / 'Target_Train.pkl'), 'rb') as f2:
        target_data = pickle.load(f2)

    # Get indices of artifact and normal ECG
    artifact_idx = target_data.index[target_data['Target'] == 1].tolist()
    normal_idx = target_data.index[target_data['Target'] == 0].tolist()    
    
    print(f'{len(artifact_idx)} artifacts and {len(normal_idx)} normals.')
    
    # Divide Signal data into artifact and normal dataframes
    df_artifact = [signal_data[i] for i in artifact_idx]
    df_normal = [signal_data[i] for i in normal_idx]
    
    # Make the output directory
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        (save_dir / "artifact").mkdir()
        (save_dir / "normal").mkdir()
    
    # Save artifact and normal ECG plots 
    for idx, df in enumerate(df_artifact):
        plot_ecg_signal(df.to_numpy(), file_name=str(save_dir / f"artifact/ecg_plot_{artifact_idx[idx]}.png"), dpi = dpi, color=color)
        print(f'artifact #: {idx + 1}/{len(artifact_idx)}')
        
    for idx, df in enumerate(df_normal):
        plot_ecg_signal(df.to_numpy(), file_name=str(save_dir / f"normal/ecg_plot_{normal_idx[idx]}.png"), dpi = dpi, color=color)
        print(f'normal #: {idx + 1}/{len(normal_idx)}')
        
    
def plot_ecg_signal(signal, sampling_rate=500, file_name="ecg_plot.png", save=True, dpi='figure', color='grey'):
    # Time axis (in seconds)
    time = np.arange(signal.shape[0]) / sampling_rate
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Titles of subplots
    titles = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    # titles = list(range(1, 13))

    # Loop over each lead
    for i in range(signal.shape[1]):
        plt.subplot(6, 2, i+1)  # 6 rows, 2 columns of subplots for 12 leads
        plt.plot(time, signal[:, i]/100, color='blue', linewidth=0.5)

        # Set the x and y limits based on the standards
        plt.xlim(0, signal.shape[0] / sampling_rate)
        plt.ylim(-3, 3)  # Assuming the signal ranges between -2mV and 2mV

        # Grid with 5mm x 5mm (small squares) and 25mm x 25mm (large squares)
        plt.grid(True, which='both', color=color, linestyle='-', linewidth=0.4)
        plt.minorticks_on()
        
        # Customizing ticks to match the ECG scale
        plt.xticks(np.arange(0, (signal.shape[0] / sampling_rate) + 1, 1), np.arange(0, (signal.shape[0] / sampling_rate) + 1, 1),
                   fontsize = 8)
        plt.yticks(np.arange(-3, 3.1, 1), np.arange(-3, 3.1, 1), fontsize = 8)
        
        plt.title(titles[i])

    # Layout adjustment
    plt.tight_layout()

    if save:
        # Save the plot as an image
        plt.savefig(file_name, dpi = dpi)
    else:
        plt.show()
    plt.close()

def parse_opt():      
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default=ROOT / 'K-MEDICON 2024_SUB1_TRAINING SET', help='train data path')
    parser.add_argument('--save_dir', default=ROOT / 'images2', help='output image directory')
    parser.add_argument('--dpi', default=450, type=int, help='dpi of plotted images')
    parser.add_argument('--color', default='grey', help='ECG line color')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))
            
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)