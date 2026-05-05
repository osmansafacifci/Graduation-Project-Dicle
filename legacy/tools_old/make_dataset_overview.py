import pandas as pd
import numpy as np
import pickle
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
RAW_DIR = DATA_DIR / "raw"
FEATURES = INTERMEDIATE_DIR / "features_top8_cycles.csv"
MAT_PATH = PROJECT_ROOT / "data-driven-prediction-of-battery-cycle-life-before-capacity-degradation-master" / "dataset" / "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
BATCH1 = RAW_DIR / "batch1.pkl"
BATCH2 = RAW_DIR / "batch2.pkl"
PLOTS_DIR = PROJECT_ROOT / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

# Dataset stats
prefix_labels = {'b1': 'Batch 1 (2017-05-12)', 'b2': 'Batch 2 (2018-02-20)'}
df = pd.read_csv(FEATURES)
table_rows = []
for prefix, label in prefix_labels.items():
    sub = df[df['cell_id'].str.startswith(prefix)]
    table_rows.append(
        {
            'Dataset': label,
            'Cells': sub['cell_id'].nunique(),
            'Rows (25/50/100)': len(sub),
            'Signals per row': 8,
        }
    )
combined = {
    'Dataset': 'Combined',
    'Cells': df['cell_id'].nunique(),
    'Rows (25/50/100)': len(df),
    'Signals per row': 8,
}
table_rows.append(combined)
df_stats = pd.DataFrame(table_rows)
print('Dataset summary table:')
print(df_stats.to_string(index=False))

# Plot sample voltage curves
cycle_indices = [5, 95]
curves = {}
with h5py.File(MAT_PATH, 'r') as f:
    batch = f['batch']
    cycles_group = f[batch['cycles'][0, 0]]  # first cell
    for idx in cycle_indices:
        ref = cycles_group['V'][idx, 0]
        dataset = f[ref]
        if dataset.attrs.get('MATLAB_empty', 0):
            continue
        V = np.array(dataset).reshape(-1)
        t = np.array(f[cycles_group['t'][idx, 0]]).reshape(-1)
        curves[idx] = (t, V)

if curves:
    plt.figure(figsize=(6, 4))
    for idx, (t, V) in curves.items():
        plt.plot(t, V, label=f'Cycle {idx+1}')
    plt.xlabel('Time (hours)')
    plt.ylabel('Voltage (V)')
    plt.title('Sample voltage curves (cell b1c0)')
    plt.legend()
    plt.grid(alpha=0.3)
    voltage_path = PLOTS_DIR / 'sample_voltage_curves.png'
    plt.tight_layout()
    plt.savefig(voltage_path, dpi=200)
    plt.close()
    print(f'Saved {voltage_path}')
else:
    print('No non-empty voltage curves found for requested cycles.')

# Plot capacity fade from summaries
capacity_cells = ['b1c0', 'b1c5']
with open(BATCH1, 'rb') as fp:
    batch1 = pickle.load(fp)
cap_data = {}
for cid in capacity_cells:
    if cid in batch1:
        qd = np.asarray(batch1[cid]['summary']['QD'], dtype=float)
        cap_data[cid] = qd

del batch1

if BATCH2.exists():
    with open(BATCH2, 'rb') as fp:
        batch2 = pickle.load(fp)
    cid = 'b2c0'
    if cid in batch2:
        cap_data[cid] = np.asarray(batch2[cid]['summary']['QD'], dtype=float)
    del batch2

if cap_data:
    plt.figure(figsize=(6, 4))
    for cid, qd in cap_data.items():
        cycles = np.arange(len(qd))
        plt.plot(cycles, qd, label=cid)
    plt.xlabel('Cycle')
    plt.ylabel('Discharge capacity (Ah)')
    plt.title('Capacity fade across cycles')
    plt.legend()
    plt.grid(alpha=0.3)
    cap_path = PLOTS_DIR / 'sample_capacity_fade.png'
    plt.tight_layout()
    plt.savefig(cap_path, dpi=200)
    plt.close()
    print(f'Saved {cap_path}')
else:
    print('No capacity data plotted.')
