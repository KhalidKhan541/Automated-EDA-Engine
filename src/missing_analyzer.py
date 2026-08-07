import numpy as np
import pandas as pd
from typing import Dict, Any, List

class MissingAnalyzer:
    """Analyze missing data patterns and generate heatmap data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.missing_matrix = df.isna()
    
    def overall_missing(self) -> Dict[str, Any]:
        """Overall missing data summary."""
        n_rows, n_cols = self.df.shape
        total_cells = n_rows * n_cols
        total_missing = self.missing_matrix.sum().sum()
        
        return {
            'total_cells': total_cells,
            'total_missing': int(total_missing),
            'total_present': int(total_cells - total_missing),
            'overall_missing_pct': float(total_missing / total_cells * 100),
            'n_complete_rows': int((~self.missing_matrix.any(axis=1)).sum()),
            'n_incomplete_rows': int(self.missing_matrix.any(axis=1).sum()),
            'complete_row_pct': float((~self.missing_matrix.any(axis=1)).sum() / n_rows * 100),
            'n_rows_with_missing': int(self.missing_matrix.any(axis=1).sum()),
        }
    
    def column_missing(self) -> pd.DataFrame:
        """Missing data per column."""
        results = []
        
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            null_pct = null_count / len(self.df) * 100
            
            results.append({
                'column': col,
                'n_missing': int(null_count),
                'pct_missing': float(null_pct),
                'n_complete': int(len(self.df) - null_count),
                'pct_complete': float(100 - null_pct),
                'dtype': str(self.df[col].dtype),
            })
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('pct_missing', ascending=False)
        
        return result_df
    
    def row_missing(self) -> Dict[str, Any]:
        """Missing data per row analysis."""
        row_missing = self.missing_matrix.sum(axis=1)
        
        return {
            'mean_missing_per_row': float(row_missing.mean()),
            'median_missing_per_row': float(row_missing.median()),
            'max_missing_per_row': int(row_missing.max()),
            'rows_with_no_missing': int((row_missing == 0).sum()),
            'rows_with_1_missing': int((row_missing == 1).sum()),
            'rows_with_2_missing': int((row_missing == 2).sum()),
            'rows_with_3plus_missing': int((row_missing >= 3).sum()),
            'distribution': row_missing.value_counts().sort_index().to_dict(),
        }
    
    def missing_patterns(self) -> Dict[str, Any]:
        """Identify missing data patterns."""
        pattern_df = self.missing_matrix.astype(int)
        pattern_str = pattern_df.apply(lambda row: ''.join(row.astype(str)), axis=1)
        
        pattern_counts = pattern_str.value_counts().head(20)
        
        patterns = []
        for pattern, count in pattern_counts.items():
            cols_missing = [self.df.columns[i] for i, c in enumerate(pattern) if c == '1']
            patterns.append({
                'pattern': pattern,
                'n_rows': int(count),
                'pct': float(count / len(self.df) * 100),
                'columns_missing': cols_missing,
                'n_cols_missing': len(cols_missing),
            })
        
        return {
            'n_unique_patterns': int(pattern_str.nunique()),
            'top_patterns': patterns,
            'most_common_pattern': pattern_counts.index[0] if len(pattern_counts) > 0 else '',
        }
    
    def pairwise_missing(self) -> Dict[str, Any]:
        """Pairwise missing correlations between columns."""
        cols_with_missing = [col for col in self.df.columns if self.df[col].isna().any()]
        
        if len(cols_with_missing) < 2:
            return {'matrix': {}, 'columns': []}
        
        n = len(cols_with_missing)
        matrix = np.eye(n)
        
        for i, col1 in enumerate(cols_with_missing):
            for j, col2 in enumerate(cols_with_missing):
                if i < j:
                    both_missing = (self.df[col1].isna() & self.df[col2].isna()).sum()
                    either_missing = (self.df[col1].isna() | self.df[col2].isna()).sum()
                    
                    if either_missing > 0:
                        corr = both_missing / either_missing
                    else:
                        corr = 0.0
                    
                    matrix[i, j] = corr
                    matrix[j, i] = corr
        
        result_df = pd.DataFrame(matrix, index=cols_with_missing, columns=cols_with_missing)
        
        return {
            'matrix': result_df.to_dict(),
            'columns': cols_with_missing,
        }
    
    def missing_summary_table(self) -> pd.DataFrame:
        """Summary table for all columns with missing data."""
        col_missing = self.column_missing()
        
        summary = col_missing[col_missing['n_missing'] > 0].copy()
        summary = summary.sort_values('pct_missing', ascending=False)
        
        summary['severity'] = pd.cut(
            summary['pct_missing'],
            bins=[0, 1, 5, 20, 50, 100],
            labels=['Minimal', 'Low', 'Moderate', 'High', 'Critical']
        )
        
        return summary
    
    def heatmap_data(self) -> Dict[str, Any]:
        """Generate data for missing heatmap visualization."""
        matrix = self.missing_matrix.astype(int).values
        
        return {
            'matrix': matrix.tolist(),
            'columns': self.df.columns.tolist(),
            'n_rows': len(self.df),
            'n_cols': len(self.df.columns),
            'row_labels': list(range(len(self.df))),
        }
    
    def drop_impact_analysis(self) -> pd.DataFrame:
        """Analyze impact of different drop thresholds."""
        thresholds = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
        results = []
        
        for threshold in thresholds:
            if threshold == 0:
                cols_to_drop = []
            else:
                cols_to_drop = [col for col in self.df.columns 
                               if self.df[col].isna().mean() > threshold]
            
            df_reduced = self.df.drop(columns=cols_to_drop)
            rows_remaining = len(df_reduced.dropna())
            
            results.append({
                'threshold': threshold,
                'n_cols_dropped': len(cols_to_drop),
                'cols_dropped': cols_to_drop,
                'n_cols_remaining': len(df_reduced.columns),
                'n_rows_complete': rows_remaining,
                'pct_rows_remaining': float(rows_remaining / len(self.df) * 100),
            })
        
        return pd.DataFrame(results)
