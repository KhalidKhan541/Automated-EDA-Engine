import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional

class ColumnAnalyzer:
    """Compute 30+ statistics per column with auto type detection."""
    
    TYPE_NUMERIC = 'numeric'
    TYPE_CATEGORICAL = 'categorical'
    TYPE_DATETIME = 'datetime'
    TYPE_BOOLEAN = 'boolean'
    TYPE_TEXT = 'text'
    TYPE_CONSTANT = 'constant'
    TYPE_UNSUPPORTED = 'unsupported'
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_types = {}
        self.column_stats = {}
    
    def detect_column_types(self) -> Dict[str, str]:
        """Auto-detect column types based on dtype and cardinality."""
        for col in self.df.columns:
            series = self.df[col]
            
            if series.nunique() <= 1:
                self.column_types[col] = self.TYPE_CONSTANT
            elif series.dtype == bool or (series.nunique() == 2 and set(series.dropna().unique()).issubset({0, 1, True, False, '0', '1', 'True', 'False'})):
                self.column_types[col] = self.TYPE_BOOLEAN
            elif pd.api.types.is_datetime64_any_dtype(series):
                self.column_types[col] = self.TYPE_DATETIME
            elif pd.api.types.is_numeric_dtype(series):
                if series.nunique() / max(len(series.dropna()), 1) < 0.05 and series.nunique() <= 20:
                    self.column_types[col] = self.TYPE_CATEGORICAL
                else:
                    self.column_types[col] = self.TYPE_NUMERIC
            elif series.dtype == object:
                nunique = series.nunique()
                if nunique <= min(50, len(series) * 0.1):
                    self.column_types[col] = self.TYPE_CATEGORICAL
                elif series.str.len().mean() > 50:
                    self.column_types[col] = self.TYPE_TEXT
                else:
                    self.column_types[col] = self.TYPE_CATEGORICAL
            else:
                self.column_types[col] = self.TYPE_UNSUPPORTED
        
        return self.column_types
    
    def analyze_all(self) -> Dict[str, Dict]:
        """Run analysis on all columns."""
        if not self.column_types:
            self.detect_column_types()
        
        for col in self.df.columns:
            col_type = self.column_types.get(col, self.TYPE_UNSUPPORTED)
            self.column_stats[col] = {
                'name': col,
                'type': col_type,
                **self._base_stats(col),
            }
            
            if col_type == self.TYPE_NUMERIC:
                self.column_stats[col].update(self._numeric_stats(col))
            elif col_type == self.TYPE_CATEGORICAL:
                self.column_stats[col].update(self._categorical_stats(col))
            elif col_type == self.TYPE_DATETIME:
                self.column_stats[col].update(self._datetime_stats(col))
            elif col_type == self.TYPE_BOOLEAN:
                self.column_stats[col].update(self._boolean_stats(col))
            elif col_type == self.TYPE_TEXT:
                self.column_stats[col].update(self._text_stats(col))
        
        return self.column_stats
    
    def _base_stats(self, col: str) -> Dict[str, Any]:
        """Base statistics for all column types."""
        series = self.df[col]
        n = len(series)
        null_count = series.isna().sum()
        
        return {
            'n_total': n,
            'n_null': int(null_count),
            'n_non_null': int(n - null_count),
            'null_pct': float(null_count / n * 100) if n > 0 else 0,
            'n_unique': int(series.nunique()),
            'unique_pct': float(series.nunique() / max(n - null_count, 1) * 100),
            'memory_bytes': int(series.memory_usage(deep=True)),
            'sample_values': series.dropna().head(5).tolist(),
        }
    
    def _numeric_stats(self, col: str) -> Dict[str, Any]:
        """30+ statistics for numeric columns."""
        series = self.df[col].dropna()
        if len(series) == 0:
            return {}
        
        values = series.values.astype(float)
        
        # Central tendency
        mean = float(np.mean(values))
        median = float(np.median(values))
        mode_result = stats.mode(values, keepdims=True)
        mode_val = float(mode_result.mode[0]) if len(mode_result.mode) > 0 else np.nan
        
        # Dispersion
        std = float(np.std(values, ddof=1))
        variance = float(np.var(values, ddof=1))
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        range_val = float(np.max(values) - np.min(values))
        cv = float(std / mean) if mean != 0 else np.nan
        
        # Quantiles
        q25 = float(np.percentile(values, 25))
        q75 = float(np.percentile(values, 75))
        q10 = float(np.percentile(values, 10))
        q90 = float(np.percentile(values, 90))
        
        # Shape
        skewness = float(stats.skew(values))
        kurtosis = float(stats.kurtosis(values))
        
        # Normality tests
        if len(values) >= 8 and len(values) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(values)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        if len(values) >= 20:
            ks_stat, ks_p = stats.kstest(values, 'norm', args=(mean, std))
        else:
            ks_stat, ks_p = np.nan, np.nan
        
        # Outlier detection
        z_scores = np.abs(stats.zscore(values))
        n_outliers_zscore = int(np.sum(z_scores > 3))
        
        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr
        n_outliers_iqr = int(np.sum((values < lower_fence) | (values > upper_fence)))
        
        # Tail heaviness
        mad = float(np.median(np.abs(values - median)))
        
        # Percentiles
        percentiles = {}
        for p in [1, 5, 25, 50, 75, 95, 99]:
            percentiles[f'p{p}'] = float(np.percentile(values, p))
        
        return {
            'mean': mean,
            'median': median,
            'mode': mode_val,
            'std': std,
            'variance': variance,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'range': range_val,
            'iqr': iqr,
            'cv': cv,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'q25': q25,
            'q75': q75,
            'q10': q10,
            'q90': q90,
            'mad': mad,
            'percentiles': percentiles,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'is_normal': float(shapiro_p) > 0.05 if not np.isnan(shapiro_p) else None,
            'n_outliers_zscore': n_outliers_zscore,
            'n_outliers_iqr': n_outliers_iqr,
            'outlier_pct_iqr': float(n_outliers_iqr / len(values) * 100),
            'lower_fence': lower_fence,
            'upper_fence': upper_fence,
        }
    
    def _categorical_stats(self, col: str) -> Dict[str, Any]:
        """Statistics for categorical columns."""
        series = self.df[col].dropna()
        if len(series) == 0:
            return {}
        
        value_counts = series.value_counts()
        total = len(series)
        
        categories = []
        for val, count in value_counts.items():
            categories.append({
                'value': str(val),
                'count': int(count),
                'pct': float(count / total * 100),
                'cumulative_pct': float(value_counts[:val].sum() / total * 100) if False else 0,
            })
        
        # Recalculate cumulative properly
        cumsum = 0
        for cat in categories:
            cumsum += cat['pct']
            cat['cumulative_pct'] = cumsum
        
        return {
            'n_categories': int(value_counts.shape[0]),
            'top_category': str(value_counts.index[0]),
            'top_category_pct': float(value_counts.iloc[0] / total * 100),
            'bottom_category': str(value_counts.index[-1]),
            'entropy': float(stats.entropy(value_counts.values)),
            'normalized_entropy': float(stats.entropy(value_counts.values) / np.log(len(value_counts))) if len(value_counts) > 1 else 0,
            'value_counts': value_counts.to_dict(),
            'categories': categories[:20],  # Top 20
        }
    
    def _datetime_stats(self, col: str) -> Dict[str, Any]:
        """Statistics for datetime columns."""
        series = pd.to_datetime(self.df[col], errors='coerce').dropna()
        if len(series) == 0:
            return {}
        
        return {
            'min_date': str(series.min()),
            'max_date': str(series.max()),
            'range_days': int((series.max() - series.min()).days),
            'n_unique_dates': int(series.nunique()),
            'hour_distribution': series.dt.hour.value_counts().to_dict() if hasattr(series.dt, 'hour') else {},
        }
    
    def _boolean_stats(self, col: str) -> Dict[str, Any]:
        """Statistics for boolean columns."""
        series = self.df[col].dropna()
        if len(series) == 0:
            return {}
        
        value_counts = series.value_counts()
        total = len(series)
        
        return {
            'true_count': int(value_counts.get(True, value_counts.get(1, 0))),
            'false_count': int(value_counts.get(False, value_counts.get(0, 0))),
            'true_pct': float(value_counts.get(True, value_counts.get(1, 0)) / total * 100),
            'false_pct': float(value_counts.get(False, value_counts.get(0, 0)) / total * 100),
            'imbalance_ratio': float(max(value_counts) / min(value_counts)) if min(value_counts) > 0 else float('inf'),
        }
    
    def _text_stats(self, col: str) -> Dict[str, Any]:
        """Statistics for text columns."""
        series = self.df[col].dropna()
        if len(series) == 0:
            return {}
        
        lengths = series.str.len()
        words = series.str.split().str.len()
        
        return {
            'avg_length': float(lengths.mean()),
            'median_length': float(lengths.median()),
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max()),
            'avg_words': float(words.mean()),
            'median_words': float(words.median()),
            'unique_ratio': float(series.nunique() / len(series)),
            'sample_texts': series.head(3).tolist(),
        }
