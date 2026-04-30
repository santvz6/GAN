import os
import torch
import pandas as pd
from tqdm import tqdm

from src.config.paths import Paths
from src.config.hparams import HParams
from src.data.dataset import NOMODataset
from src.models import Generator
from src.smpl_module_project.measure import MeasureSMPL
from src.smpl_module_project.evaluate import evaluate_mae

def evaluate():
    Paths.init_project()
    hp = HParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    G = Generator(hp).to(device)
    # Find latest checkpoint
    ckpt_files = list(Paths.EXPERIMENTS_DIR.glob("wgangp_ckpt_*.pt"))
    if not ckpt_files:
        print("No checkpoints found. Please train first.")
        return
        
    latest_ckpt = sorted(ckpt_files, key=os.path.getmtime)[-1]
    print(f"Loading checkpoint {latest_ckpt}...")
    ckpt = torch.load(latest_ckpt, map_location=device)
    G.load_state_dict(ckpt['G_state_dict'])
    G.eval()
    
    # Load dataset
    test_ds = NOMODataset(split='test')
    
    # SMPL Measurer
    measurer = MeasureSMPL(model_root=Paths.SMPL_DIR.parent)
    
    all_maes = []
    
    print("Evaluating...")
    with torch.no_grad():
        for i in tqdm(range(len(test_ds))):
            meas_target, _, gender = test_ds[i]
            
            # Target measurements as dict (for evaluate_mae)
            target_dict = {col: meas_target[j].item() for j, col in enumerate(hp.meas_cols)}
            
            # Generate fake betas
            z = torch.randn(1, hp.noise_dim).to(device)
            meas_cond = meas_target.unsqueeze(0).to(device)
            fake_betas = G(z, meas_cond)
            
            # Evaluate generated shape
            measurer.from_body_model(gender=gender, shape=fake_betas.cpu())
            # Measure only the columns we care about if possible, or all
            # evaluate_mae will match keys. However, the names in MeasureSMPL
            # might not perfectly match the NOMO3D names.
            # We will map NOMO3D names to SMPL measurement names if they differ.
            
            # Let's measure all
            measurer.measure(measurer.all_possible_measurements)
            estim_dict = measurer.measurements
            
            # Compute MAE
            # Note: We need a mapping between NOMO3D column names and SMPL measurement names.
            # For simplicity in this script, we'll try to find matching keys (case-insensitive) or assume mapping.
            # TODO: Implement precise name mapping if needed.
            
            # Example mapping
            mapping = {
                "BUST_Circ": "chest_circumference",
                "NaturalWAIST_Circ": "waist_circumference",
                "HIP_Circ": "hip_circumference",
                # add more mappings as needed
            }
            
            mapped_estim = {}
            for target_k, smpl_k in mapping.items():
                if target_k in target_dict and smpl_k in estim_dict:
                    mapped_estim[target_k] = estim_dict[smpl_k]
                    
            if mapped_estim:
                mae = evaluate_mae({k: target_dict[k] for k in mapped_estim.keys()}, mapped_estim)
                all_maes.append(mae)

    if not all_maes:
        print("No valid measurements matched for evaluation.")
        return

    # Average MAEs
    avg_mae = {k: sum(d[k] for d in all_maes) / len(all_maes) for k in all_maes[0].keys()}
    
    df = pd.DataFrame({"Measurement": list(avg_mae.keys()), "MAE (cm)": list(avg_mae.values())})
    print("\n--- Evaluation Results ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    evaluate()
