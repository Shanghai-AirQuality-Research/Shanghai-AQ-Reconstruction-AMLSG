import pandas as pd
from src.amlsg_engine import AMLSGEngine

def execute_spatial_downscaling(city_data, district_data, targets):
    """
    Stage II: Macro-to-Micro Spatial Transfer.
    Downscales city-level vectors to district-level specifics.
    """
    print(">>> Starting Stage II: Spatial Downscaling...")
    
    # Merge City features into District data
    # f(City_Vector, District_ID) -> District_Pollutant
    merged_train = pd.merge(district_data, city_data, on=['Year', 'Month'], suffixes=('', '_city'))
    
    for target in targets:
        engine = AMLSGEngine(target_label=target)
        engine.fit(merged_train, time_limit=600)
        # Prediction logic would follow here...
        print(f"Spatial model for {target} trained successfully.")
