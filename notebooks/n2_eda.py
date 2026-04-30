import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data.dataset import NOMODataset
from src.config.paths import Paths
import seaborn as sns

def run_eda():
    Paths.init_project()
    print("Loading dataset...")
    ds = NOMODataset(split='train')
    
    measurements = []
    genders = []
    
    for sample in ds.samples:
        measurements.append(sample['measurements'].numpy())
        genders.append(sample['gender'])
        
    meas_df = pd.DataFrame(measurements, columns=ds.hparams.meas_cols)
    meas_df['Gender'] = genders
    
    print("\n--- Summary Statistics ---")
    print(meas_df.groupby('Gender').describe().T)
    
    # Plotting
    plot_path = Paths.EXPERIMENTS_DIR / "eda_distributions.png"
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(ds.hparams.meas_cols):
        sns.histplot(data=meas_df, x=col, hue='Gender', ax=axes[i], kde=True)
        axes[i].set_title(col)
        
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nPlots saved to {plot_path}")

if __name__ == "__main__":
    run_eda()
