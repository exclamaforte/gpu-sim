#!/usr/bin/env python3
"""Gradient-boosted trees baseline (XGBoost/LightGBM) over hand-engineered features.

Tree ensembles are frequently competitive with deep models on structured/tabular
performance-prediction tasks. This determines whether the transformer's
sequence understanding of code is actually earning anything beyond numeric features.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    
    def train_test_split(X, y, test_size=0.2, random_state=42):
        """Simple train/test split."""
        np.random.seed(random_state)
        n = len(X)
        indices = np.random.permutation(n)
        split = int(n * (1 - test_size))
        train_idx, val_idx = indices[:split], indices[split:]
        return X[train_idx], X[val_idx], y[train_idx], y[val_idx]
    
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true))
    
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0


try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


@dataclass
class TreeBaselineConfig:
    """Configuration for tree baseline."""
    model_type: str = 'xgboost'  # 'xgboost', 'lightgbm', 'sklearn'
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    early_stopping_rounds: int = 50
    target_transform: str = 'log'  # 'log', 'none'
    n_jobs: int = -1


class TreeBaseline:
    """Gradient-boosted trees baseline for CUDA runtime prediction."""
    
    def __init__(self, config: Optional[TreeBaselineConfig] = None):
        self.config = config or TreeBaselineConfig()
        self.model = None
        self.feature_names = None
        self.is_fitted = False
    
    def _get_model(self):
        """Get the appropriate model instance."""
        if self.config.model_type == 'xgboost' and HAS_XGB:
            return xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                objective='reg:squarederror',
                tree_method='hist',
                early_stopping_rounds=self.config.early_stopping_rounds,
            )
        elif self.config.model_type == 'lightgbm' and HAS_LGB:
            return lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                objective='regression',
                verbosity=-1,
            )
        else:
            # Fallback to sklearn
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                random_state=self.config.random_state,
            )
    
    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        """Transform target variable."""
        if self.config.target_transform == 'log':
            return np.log(y + 1e-6)
        return y
    
    def _inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform target."""
        if self.config.target_transform == 'log':
            return np.exp(y)
        return y
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
    ) -> 'TreeBaseline':
        """Fit the model."""
        self.feature_names = feature_names or [f'feat_{i}' for i in range(X.shape[1])]
        
        # Transform target
        y_trans = self._transform_target(y)
        
        # Prepare validation set
        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_trans = self._transform_target(y_val)
            eval_set = [(X_val, y_val_trans)]
        
        # Get model
        self.model = self._get_model()
        
        # Fit
        if HAS_XGB and self.config.model_type == 'xgboost':
            self.model.fit(
                X, y_trans,
                eval_set=eval_set,
                verbose=False,
            )
        elif HAS_LGB and self.config.model_type == 'lightgbm':
            self.model.fit(
                X, y_trans,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)],
            )
        else:
            self.model.fit(X, y_trans)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict runtime in milliseconds."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        pred_trans = self.model.predict(X)
        return self._inverse_transform_target(pred_trans)
    
    def predict_log(self, X: np.ndarray) -> np.ndarray:
        """Predict log runtime."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)
    
    def feature_importances(self) -> dict:
        """Get feature importances."""
        if not self.is_fitted:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'get_score'):
            # XGBoost
            score = self.model.get_score(importance_type='gain')
            importances = np.zeros(len(self.feature_names))
            for i, name in enumerate(self.feature_names):
                importances[i] = score.get(name, 0)
        elif hasattr(self.model, 'feature_importance'):
            # LightGBM
            importances = self.model.feature_importance(importance_type='gain')
        else:
            return {}
        
        # Normalize
        total = importances.sum()
        if total > 0:
            importances = importances / total
        
        return dict(zip(self.feature_names, importances))
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        pred = self.predict(X)
        
        # Metrics
        mape = mean_absolute_percentage_error(y, pred) * 100
        mse = mean_squared_error(y, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, pred)
        
        # Log-scale metrics
        log_y = np.log(y + 1e-6)
        log_pred = np.log(pred + 1e-6)
        rmsle = np.sqrt(np.mean((log_y - log_pred) ** 2))
        
        return {
            'mape': mape,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'rmsle': rmsle,
        }


def train_tree_baseline(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: Optional[list[str]] = None,
    val_split: float = 0.2,
    config: Optional[TreeBaselineConfig] = None,
) -> tuple[TreeBaseline, dict]:
    """Train a tree baseline and return model + metrics."""
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, targets, test_size=val_split, random_state=42
    )
    
    # Train
    config = config or TreeBaselineConfig()
    model = TreeBaseline(config)
    model.fit(X_train, y_train, X_val, y_val, feature_names)
    
    # Evaluate
    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    
    metrics = {
        'train': train_metrics,
        'val': val_metrics,
    }
    
    return model, metrics


def compare_tree_models(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> dict:
    """Compare different tree model implementations."""
    results = {}
    
    for model_type in ['xgboost', 'lightgbm', 'sklearn']:
        if model_type == 'xgboost' and not HAS_XGB:
            results[model_type] = {'error': 'XGBoost not installed'}
            continue
        if model_type == 'lightgbm' and not HAS_LGB:
            results[model_type] = {'error': 'LightGBM not installed'}
            continue
        
        config = TreeBaselineConfig(model_type=model_type)
        model, metrics = train_tree_baseline(features, targets, feature_names, config=config)
        results[model_type] = {
            'metrics': metrics,
            'feature_importances': model.feature_importances(),
        }
    
    return results


if __name__ == '__main__':
    # Demo with synthetic data
    import sys
    sys.path.insert(0, '/home/gabe/Documents/gpu-sim')
    
    from epic_u2o.features import extract_normalized_features
    from epic_u2o.generator_stage1 import generate_kernels
    from epic_u2o.structural_split import enrich_kernels_with_families
    
    print("Generating synthetic dataset...")
    kernels = list(generate_kernels(max_kernels=500))
    kernels = enrich_kernels_with_families([k.to_dict() for k in kernels])
    
    # Extract features
    X = []
    y = []
    for k in kernels:
        # Use normalized features
        feat_vec = extract_normalized_features(k['source'], None)
        X.append(feat_vec)
        # Synthetic target: simple model based on bytes moved
        import random
        base_time = 0.01 + k['metadata'].get('problem_size', 1024) * 1e-5 * random.lognormvariate(0, 0.1)
        y.append(base_time)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Feature names
    feature_names = [
        'grid_x', 'grid_y', 'grid_z', 'block_x', 'block_y', 'block_z',
        'shared_mem', 'registers', 'num_args', 'num_ptr_args', 'num_scalar_args',
        'arg_bytes', 'global_reads', 'global_writes', 'shared_reads', 'shared_writes',
        'bytes_moved', 'flops', 'int_ops', 'has_loops', 'has_cond', 'loop_depth',
        'has_sync', 'num_sync', 'uses_shared', 'uses_const', 'uses_vector',
    ]
    
    # Compare models
    print("\nComparing tree models...")
    results = compare_tree_models(X, y, feature_names)
    
    for model_type, result in results.items():
        if 'error' in result:
            print(f"  {model_type}: {result['error']}")
        else:
            val = result['metrics']['val']
            print(f"  {model_type}: MAPE={val['mape']:.2f}%, R2={val['r2']:.3f}, RMSLE={val['rmsle']:.3f}")
    
    # Train best model
    print("\nTraining XGBoost model...")
    if HAS_XGB:
        config = TreeBaselineConfig(model_type='xgboost')
        model, metrics = train_tree_baseline(X, y, feature_names, config=config)
        
        print(f"Train MAPE: {metrics['train']['mape']:.2f}%")
        print(f"Val MAPE: {metrics['val']['mape']:.2f}%")
        
        # Feature importances
        importances = model.feature_importances()
        print("\nTop 10 features:")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
            print(f"  {name}: {imp:.4f}")