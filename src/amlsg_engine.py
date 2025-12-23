import pandas as pd
from autogluon.tabular import TabularPredictor
import os
import logging

class AMLSGEngine:
    """
    The core implementation of the Automated Multi-Layer Stacked Generalization (AMLSG) algorithm.
    This class manages the hierarchical ensemble training and inference process.
    """
    def __init__(self, target_label, output_dir='models/', mode='high_fidelity'):
        self.target = target_label
        self.output_dir = output_dir
        # Map academic configurations to internal hyperparameters
        self.preset = 'best_quality' if mode == 'high_fidelity' else 'medium_quality'
        self.predictor = None
        self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"AMLSG-{self.target}")

    def fit(self, train_data, time_limit=600):
        """
        Executes the stacking strategy.
        Args:
            train_data (pd.DataFrame): Training corpus.
            time_limit (int): Optimization budget in seconds.
        """
        self.logger.info(f"Initializing AMLSG training for target: {self.target}...")
        
        # Define the hyperparameter search space for base learners
        # NN_TORCH: Deep Neural Networks
        # GBM/CAT/XGB: Gradient Boosting Trees
        hyperparameters = {
            'NN_TORCH': {}, 
            'GBM': {},      
            'CAT': {},      
            'XGB': {}        
        }
        
        save_path = os.path.join(self.output_dir, self.target)
        
        self.predictor = TabularPredictor(
            label=self.target, 
            path=save_path,
            eval_metric='root_mean_squared_error'
        ).fit(
            train_data,
            presets=self.preset, 
            auto_stack=True,      # Enables Multi-Layer Stacking
            num_bag_folds=5,      # Enables Repeated k-fold Bagging
            hyperparameters=hyperparameters,
            time_limit=time_limit
        )
        self.logger.info(f"Model ensemble established. Leaderboard available in {save_path}")

    def predict(self, data):
        """Inference utilizing the full weighted ensemble."""
        if not self.predictor:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.predictor.predict(data)
