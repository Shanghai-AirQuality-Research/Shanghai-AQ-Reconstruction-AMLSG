import pandas as pd
from src.amlsg_engine import AMLSGEngine

def execute_temporal_backcasting(reference_path, historical_path, targets):
    """
    Stage I: Temporal Transfer Learning.
    Reconstructs historical city-level baselines (X_S3) using modern references (X_S1).
    """
    print(">>> Starting Stage I: Temporal Backcasting...")
    
    # Load Datasets
    ref_df = pd.read_csv(reference_path) # 2020-2025 Data
    hist_df = pd.read_csv(historical_path) # 2014-2019 Data (with missing cols)
    
    reconstructed_data = hist_df.copy()
    
    for target in targets:
        print(f"Processing Target: {target}")
        # Initialize AMLSG Engine
        engine = AMLSGEngine(target_label=target, mode='high_fidelity')
        
        # Train on Reference Period (Data-Rich)
        engine.fit(ref_df, time_limit=300)
        
        # Infer on Historical Period (Data-Sparse)
        preds = engine.predict(hist_df)
        reconstructed_data[target] = preds
        
    return reconstructed_data
