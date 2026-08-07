import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple

class OutlierDetector:
    """Detect outliers using IQR, Z-score, and Isolation Forest."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.outlier_results = {}
    
    def iqr_outliers(self, col: str, multiplier: float = 1.5) -> Dict[str, Any]:
        """Detect outliers using Interquartile Range method."""
        series = self.df[col].dropna()
        if len(series) == 0:
            return {}
        
        q25 = float(np.percentile(series, 25))
        q75 = float(np.percentile(series, 75))
        iqr = q75 - q25
        
        lower_fence = q25 - multiplier * iqr
        upper_fence = q75 + multiplier * iqr
        
        outliers = series[(series < lower_fence) | (series > upper_fence)]
        
        return {
            'method': 'IQR',
            'column': col,
            'q25': q25,
            'q75': q75,
            'iqr': float(iqr),
            'lower_fence': float(lower_fence),
            'upper_fence': float(upper_fence),
            'n_outliers': int(len(outliers)),
            'pct_outliers': float(len(outliers) / len(series) * 100),
            'outlier_values': outliers.sort_values().tolist()[:20],
            'outlier_indices': outliers.index.tolist()[:50],
        }
    
    def zscore_outliers(self, col: str, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        series = self.df[col].dropna()
        if len(series) == 0:
            return {}
        
        mean = float(np.mean(series))
        std = float(np.std(series, ddof=1))
        
        if std == 0:
            return {
                'method': 'Z-score',
                'column': col,
                'n_outliers': 0,
                'pct_outliers': 0.0,
            }
        
        z_scores = np.abs((series - mean) / std)
        outliers = series[z_scores > threshold]
        
        return {
            'method': 'Z-score',
            'column': col,
            'mean': mean,
            'std': std,
            'threshold': threshold,
            'n_outliers': int(len(outliers)),
            'pct_outliers': float(len(outliers) / len(series) * 100),
            'outlier_values': outliers.sort_values().tolist()[:20],
            'outlier_indices': outliers.index.tolist()[:50],
            'max_zscore': float(z_scores.max()),
        }
    
    def modified_zscore_outliers(self, col: str, threshold: float = 3.5) -> Dict[str, Any]:
        """Detect outliers using Modified Z-score (MAD-based, robust to outliers)."""
        series = self.df[col].dropna()
        if len(series) == 0:
            return {}
        
        median = float(np.median(series))
        mad = float(np.median(np.abs(series - median)))
        
        if mad == 0:
            return {
                'method': 'Modified Z-score',
                'column': col,
                'n_outliers': 0,
                'pct_outliers': 0.0,
            }
        
        modified_z = 0.6745 * (series - median) / mad
        outliers = series[np.abs(modified_z) > threshold]
        
        return {
            'method': 'Modified Z-score',
            'column': col,
            'median': median,
            'mad': mad,
            'threshold': threshold,
            'n_outliers': int(len(outliers)),
            'pct_outliers': float(len(outliers) / len(series) * 100),
            'outlier_values': outliers.sort_values().tolist()[:20],
            'outlier_indices': outliers.index.tolist()[:50],
        }
    
    def isolation_forest_outliers(self, col: str, contamination: float = 0.05) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            return {'method': 'Isolation Forest', 'error': 'sklearn not installed'}
        
        series = self.df[col].dropna()
        if len(series) < 10:
            return {}
        
        X = series.values.reshape(-1, 1)
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(X)
        scores = iso_forest.decision_function(X)
        
        outlier_mask = predictions == -1
        outliers = series[outlier_mask]
        
        return {
            'method': 'Isolation Forest',
            'column': col,
            'contamination': contamination,
            'n_outliers': int(outlier_mask.sum()),
            'pct_outliers': float(outlier_mask.sum() / len(series) * 100),
            'outlier_values': outliers.sort_values().tolist()[:20],
            'outlier_indices': outliers.index.tolist()[:50],
            'mean_anomaly_score': float(scores.mean()),
            'min_anomaly_score': float(scores.min()),
        }
    
    def detect_all_outliers(self, col: str) -> Dict[str, Any]:
        """Run all outlier detection methods on a column."""
        results = {
            'iqr': self.iqr_outliers(col),
            'zscore': self.zscore_outliers(col),
            'modified_zscore': self.modified_zscore_outliers(col),
            'isolation_forest': self.isolation_forest_outliers(col),
        }
        
        # Consensus: outliers flagged by majority of methods
        all_indices = set()
        method_counts = {}
        
        for method_name, result in results.items():
            if 'outlier_indices' in result:
                for idx in result['outlier_indices']:
                    all_indices.add(idx)
                    method_counts[idx] = method_counts.get(idx, 0) + 1
        
        consensus_outliers = [idx for idx, count in method_counts.items() if count >= 3]
        
        results['consensus'] = {
            'n_outliers': len(consensus_outliers),
            'outlier_indices': sorted(consensus_outliers)[:50],
            'methods_agreeing': method_counts,
        }
        
        self.outlier_results[col] = results
        return results
    
    def detect_all_columns(self) -> Dict[str, Dict]:
        """Run outlier detection on all numeric columns."""
        for col in self.numeric_cols:
            self.detect_all_outliers(col)
        return self.outlier_results
    
    def outlier_summary(self) -> pd.DataFrame:
        """Summary of outliers across all columns."""
        if not self.outlier_results:
            self.detect_all_columns()
        
        results = []
        for col in self.numeric_cols:
            if col in self.outlier_results:
                col_results = self.outlier_results[col]
                results.append({
                    'column': col,
                    'iqr_outliers': col_results.get('iqr', {}).get('n_outliers', 0),
                    'zscore_outliers': col_results.get('zscore', {}).get('n_outliers', 0),
                    'mad_outliers': col_results.get('modified_zscore', {}).get('n_outliers', 0),
                    'iforest_outliers': col_results.get('isolation_forest', {}).get('n_outliers', 0),
                    'consensus_outliers': col_results.get('consensus', {}).get('n_outliers', 0),
                })
        
        return pd.DataFrame(results)
    
    def outlier_overlap(self, col: str) -> Dict[str, Any]:
        """Analyze overlap between different outlier detection methods."""
        if col not in self.outlier_results:
            self.detect_all_outliers(col)
        
        results = self.outlier_results[col]
        
        sets = {}
        for method in ['iqr', 'zscore', 'modified_zscore', 'isolation_forest']:
            if method in results and 'outlier_indices' in results[method]:
                sets[method] = set(results[method]['outlier_indices'])
        
        if len(sets) < 2:
            return {'error': 'Not enough methods with results'}
        
        overlap_matrix = {}
        methods = list(sets.keys())
        
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i <= j:
                    intersection = len(sets[m1] & sets[m2])
                    union = len(sets[m1] | sets[m2])
                    jaccard = intersection / union if union > 0 else 0
                    overlap_matrix[f'{m1} vs {m2}'] = {
                        'intersection': intersection,
                        'union': union,
                        'jaccard': float(jaccard),
                    }
        
        return overlap_matrix
