import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple

class CorrelationAnalyzer:
    """Compute Pearson, Spearman, and Cramér's V correlations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def pearson_correlation(self) -> Dict[str, Any]:
        """Compute Pearson correlation matrix for numeric columns."""
        if len(self.numeric_cols) < 2:
            return {'matrix': {}, 'p_values': {}}
        
        corr_matrix = self.df[self.numeric_cols].corr(method='pearson')
        p_value_matrix = pd.DataFrame(
            np.ones((len(self.numeric_cols), len(self.numeric_cols))),
            index=self.numeric_cols,
            columns=self.numeric_cols
        )
        
        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols):
                if i != j:
                    valid = self.df[[col1, col2]].dropna()
                    if len(valid) > 2:
                        corr, p_val = stats.pearsonr(valid[col1], valid[col2])
                        p_value_matrix.loc[col1, col2] = p_val
        
        return {
            'method': 'Pearson',
            'matrix': corr_matrix.to_dict(),
            'p_values': p_value_matrix.to_dict(),
            'columns': self.numeric_cols,
        }
    
    def spearman_correlation(self) -> Dict[str, Any]:
        """Compute Spearman rank correlation matrix for numeric columns."""
        if len(self.numeric_cols) < 2:
            return {'matrix': {}, 'p_values': {}}
        
        corr_matrix = self.df[self.numeric_cols].corr(method='spearman')
        p_value_matrix = pd.DataFrame(
            np.ones((len(self.numeric_cols), len(self.numeric_cols))),
            index=self.numeric_cols,
            columns=self.numeric_cols
        )
        
        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols):
                if i != j:
                    valid = self.df[[col1, col2]].dropna()
                    if len(valid) > 2:
                        corr, p_val = stats.spearmanr(valid[col1], valid[col2])
                        p_value_matrix.loc[col1, col2] = p_val
        
        return {
            'method': 'Spearman',
            'matrix': corr_matrix.to_dict(),
            'p_values': p_value_matrix.to_dict(),
            'columns': self.numeric_cols,
        }
    
    def cramers_v(self, col1: str, col2: str) -> Dict[str, float]:
        """Compute Cramér's V for two categorical columns."""
        contingency = pd.crosstab(self.df[col1], self.df[col2])
        
        chi2 = stats.chi2_contingency(contingency)[0]
        n = len(self.df.dropna(subset=[col1, col2]))
        min_dim = min(contingency.shape) - 1
        
        if min_dim == 0 or n == 0:
            return {'cramers_v': 0.0, 'chi2': 0.0, 'p_value': 1.0}
        
        cramers_v_val = np.sqrt(chi2 / (n * min_dim))
        
        _, p_value, _, _ = stats.chi2_contingency(contingency)
        
        return {
            'cramers_v': float(cramers_v_val),
            'chi2': float(chi2),
            'p_value': float(p_value),
        }
    
    def cramers_v_matrix(self) -> Dict[str, Any]:
        """Compute Cramér's V matrix for all categorical columns."""
        if len(self.categorical_cols) < 2:
            return {'matrix': {}, 'columns': []}
        
        n_cols = len(self.categorical_cols)
        matrix = pd.DataFrame(
            np.eye(n_cols),
            index=self.categorical_cols,
            columns=self.categorical_cols
        )
        
        for i, col1 in enumerate(self.categorical_cols):
            for j, col2 in enumerate(self.categorical_cols):
                if i < j:
                    result = self.cramers_v(col1, col2)
                    matrix.loc[col1, col2] = result['cramers_v']
                    matrix.loc[col2, col1] = result['cramers_v']
        
        return {
            'method': "Cramér's V",
            'matrix': matrix.to_dict(),
            'columns': self.categorical_cols,
        }
    
    def mixed_correlation(self) -> Dict[str, Any]:
        """Correlation matrix including both numeric and categorical columns."""
        all_cols = self.numeric_cols + self.categorical_cols
        if len(all_cols) < 2:
            return {'matrix': {}, 'columns': []}
        
        n = len(all_cols)
        matrix = pd.DataFrame(np.eye(n), index=all_cols, columns=all_cols)
        
        # Numeric vs Numeric (Pearson)
        for i, col1 in enumerate(self.numeric_cols):
            for j, col2 in enumerate(self.numeric_cols):
                if i < j:
                    valid = self.df[[col1, col2]].dropna()
                    if len(valid) > 2:
                        corr, _ = stats.pearsonr(valid[col1], valid[col2])
                        matrix.loc[col1, col2] = corr
                        matrix.loc[col2, col1] = corr
        
        # Categorical vs Categorical (Cramér's V)
        for i, col1 in enumerate(self.categorical_cols):
            for j, col2 in enumerate(self.categorical_cols):
                if i < j:
                    result = self.cramers_v(col1, col2)
                    matrix.loc[col1, col2] = result['cramers_v']
                    matrix.loc[col2, col1] = result['cramers_v']
        
        # Numeric vs Categorical (Eta squared)
        for num_col in self.numeric_cols:
            for cat_col in self.categorical_cols:
                eta2 = self._eta_squared(num_col, cat_col)
                matrix.loc[num_col, cat_col] = eta2
                matrix.loc[cat_col, num_col] = eta2
        
        return {
            'method': 'Mixed (Pearson + Cramér\'s V + Eta²)',
            'matrix': matrix.to_dict(),
            'columns': all_cols,
        }
    
    def _eta_squared(self, numeric_col: str, categorical_col: str) -> float:
        """Compute eta squared (effect size) between numeric and categorical."""
        groups = []
        for name, group in self.df.groupby(categorical_col)[numeric_col]:
            valid = group.dropna()
            if len(valid) > 0:
                groups.append(valid.values)
        
        if len(groups) < 2:
            return 0.0
        
        f_val, p_val = stats.f_oneway(*groups)
        
        total_n = sum(len(g) for g in groups)
        k = len(groups)
        
        if total_n <= k:
            return 0.0
        
        eta2 = (f_val * (k - 1)) / (f_val * (k - 1) + (total_n - k))
        
        return float(eta2)
    
    def top_correlations(self, n: int = 10) -> Dict[str, Any]:
        """Get top N strongest correlations."""
        pearson = self.pearson_correlation()
        corr_df = pd.DataFrame(pearson['matrix'])
        
        pairs = []
        seen = set()
        
        for col1 in corr_df.columns:
            for col2 in corr_df.columns:
                if col1 != col2:
                    pair = tuple(sorted([col1, col2]))
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append({
                            'var1': col1,
                            'var2': col2,
                            'correlation': float(corr_df.loc[col1, col2]),
                            'abs_correlation': float(abs(corr_df.loc[col1, col2])),
                        })
        
        pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        return {
            'top_positive': [p for p in pairs if p['correlation'] > 0][:n],
            'top_negative': [p for p in pairs if p['correlation'] < 0][:n],
            'all_pairs': pairs[:n * 2],
        }
