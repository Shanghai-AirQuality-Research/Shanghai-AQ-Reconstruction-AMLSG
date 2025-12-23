import os
from src.stage1_temp import execute_temporal_backcasting
from src.stage2_spatial import execute_spatial_downscaling
from src.stage3_hybrid import execute_hybrid_reconstruction

# Configuration
DATA_DIR = 'data/'
REF_FILE = os.path.join(DATA_DIR, 'city_level_reference_2020_2025.csv')
HIST_FILE = os.path.join(DATA_DIR, 'city_level_historical_input.csv')

def main():
    print("Initializing Hierarchical Spatio-Temporal Reconstruction Framework...")
    
    # 1. Temporal Backcasting (City Level)
    # Reconstruct O3, CO, PM10 for 2014-2019
    city_targets = ['O3_mean', 'CO_mean', 'PM10_mean']
    if os.path.exists(REF_FILE):
        execute_temporal_backcasting(REF_FILE, HIST_FILE, city_targets)
    else:
        print("Demo data not found. Please place data in 'data/' directory.")

    # 2. Spatial Downscaling & 3. Hybrid Reconstruction
    # (Placeholder calls for demonstration)
    print("Stage I Complete. Proceeding to Stage II & III...")

if __name__ == "__main__":
    main()
