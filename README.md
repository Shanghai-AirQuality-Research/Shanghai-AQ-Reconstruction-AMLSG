# High-Resolution Spatio-Temporal Air Quality Inventory for Shanghai (2014–2025)

## Overview
This repository contains the official implementation of the **Hierarchical Multi-Layer Stacked Generalization (AMLSG)** framework. 

This research addresses the **"Reverse Data Quality Pyramid"** challenge in urban environmental monitoring. By utilizing a dual-transfer learning strategy (temporal backcasting and spatial downscaling), we constructed the first gap-free, high-resolution air quality inventory for Shanghai's 16 administrative districts spanning 2014–2025.

## Key Features
- **Algorithm:** Automated Multi-Layer Stacked Generalization (AMLSG).
- **Architecture:** Hybrid ensemble integrating Deep Neural Networks (e.g., NeuralNetTorch) and Gradient Boosting Decision Trees (LightGBM, CatBoost).
- **Indicators:** Covers 7 key pollutants: $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{O}_3$, $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, and AQI.
- **Resolution:** Monthly temporal resolution at the district level (meso-scale).

## Repository Structure
The project is organized as follows:

```text
├── src/                # Core implementation of the reconstruction framework
│   ├── amlsg_engine.py # The AMLSG algorithm wrapper
│   ├── stage1_temp.py  # Stage I: Temporal Backcasting module
│   ├── stage2_spatial.py # Stage II: Spatial Downscaling module
│   └── stage3_hybrid.py  # Stage III: Hybrid Anchor module
├── data/               # Spatio-temporal inventory datasets
│   ├── city_level_complete_inventory.csv     # Reconstructed city-level indicators
│   ├── district_level_complete_inventory.csv # Reconstructed district-level indicators
│   └── raw_dataset.xlsx                      # Original raw monitoring records
├── main_pipeline.py    # Execution script for peer review verification
└── requirements.txt    # Python dependencies

## Getting Started

Prerequisites
Python 3.8+
AutoGluon 0.8.2+
PyTorch
Pandas, Numpy

## Installation

git clone [https://github.com/Shanghai-AirQuality-Research/Shanghai-AQ-Reconstruction-AMLSG.git](https://github.com/Shanghai-AirQuality-Research/Shanghai-AQ-Reconstruction-AMLSG.git)
cd Shanghai-AQ-Reconstruction-AMLSG
pip install -r requirements.txt

## Usage

Run the demonstration script to initialize the model pipeline:
python main_pipeline.py

## Citation
If you utilize this code or dataset, please cite our paper:
[Your Name], et al. "A High-Resolution Spatio-Temporal Inventory of Air Quality in Shanghai (2014–2025) Reconstructed via Hierarchical Multi-Layer Stacked Generalization." [Journal Name], 2025.

## License
This project is licensed under the MIT License - see the LICENSE file for details.



