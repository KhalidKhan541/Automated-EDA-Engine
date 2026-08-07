import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any, Optional

from .column_analyzer import ColumnAnalyzer
from .correlation import CorrelationAnalyzer
from .missing_analyzer import MissingAnalyzer
from .outlier_detector import OutlierDetector
from .plotter import EDAPlotter
from .report_generator import HTMLReportGenerator

class EDAPipeline:
    """Main pipeline orchestrating all EDA components."""
    
    def __init__(self, output_dir: str = 'output', style: str = 'dark'):
        self.output_dir = output_dir
        self.style = style
        self.logger = logging.getLogger(__name__)
        
        self.plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)
        
        self.results = {}
        self.plot_paths = {}
    
    def run_full_analysis(self, df: pd.DataFrame,
                         dataset_name: str = 'Dataset') -> str:
        """Run complete EDA analysis and generate HTML report."""
        self.logger.info("=" * 60)
        self.logger.info(f"AUTOMATED EDA: {dataset_name}")
        self.logger.info("=" * 60)
        
        # Dataset info
        self.results['dataset_info'] = self._get_dataset_info(df, dataset_name)
        
        # Column analysis
        self.logger.info("Analyzing columns...")
        col_analyzer = ColumnAnalyzer(df)
        self.results['column_stats'] = col_analyzer.analyze_all()
        self.results['column_types'] = col_analyzer.column_types
        
        # Correlations
        self.logger.info("Computing correlations...")
        corr_analyzer = CorrelationAnalyzer(df)
        self.results['correlations'] = self._compute_correlations(corr_analyzer)
        
        # Missing data
        self.logger.info("Analyzing missing data...")
        missing_analyzer = MissingAnalyzer(df)
        self.results['missing'] = self._analyze_missing(missing_analyzer)
        
        # Outliers
        self.logger.info("Detecting outliers...")
        outlier_detector = OutlierDetector(df)
        self.results['outliers'] = self._detect_outliers(outlier_detector)
        
        # Generate plots
        self.logger.info("Generating plots...")
        self._generate_plots(df, col_analyzer, corr_analyzer, missing_analyzer, outlier_detector)
        
        # Generate report
        self.logger.info("Generating HTML report...")
        report_gen = HTMLReportGenerator(self.output_dir)
        report_path = report_gen.generate_report(self.results, self.plot_paths)
        
        self.logger.info(f"Report saved to: {report_path}")
        return report_path
    
    def _get_dataset_info(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            'name': name,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'total_cells': len(df) * len(df.columns),
            'missing_cells': int(df.isna().sum().sum()),
            'missing_pct': float(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100),
            'memory_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
        }
    
    def _compute_correlations(self, analyzer: CorrelationAnalyzer) -> Dict[str, Any]:
        """Compute all correlation types."""
        results = {}
        
        try:
            results['pearson'] = analyzer.pearson_correlation()
        except Exception as e:
            self.logger.warning(f"Pearson correlation failed: {e}")
        
        try:
            results['spearman'] = analyzer.spearman_correlation()
        except Exception as e:
            self.logger.warning(f"Spearman correlation failed: {e}")
        
        try:
            results['cramers_v'] = analyzer.cramers_v_matrix()
        except Exception as e:
            self.logger.warning(f"Cramér's V failed: {e}")
        
        try:
            results['top'] = analyzer.top_correlations()
        except Exception as e:
            self.logger.warning(f"Top correlations failed: {e}")
        
        return results
    
    def _analyze_missing(self, analyzer: MissingAnalyzer) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        results = {}
        
        try:
            results['overall'] = analyzer.overall_missing()
        except Exception as e:
            self.logger.warning(f"Overall missing failed: {e}")
        
        try:
            col_missing = analyzer.column_missing()
            results['by_column'] = col_missing.to_dict('records')
        except Exception as e:
            self.logger.warning(f"Column missing failed: {e}")
        
        try:
            results['row_analysis'] = analyzer.row_missing()
        except Exception as e:
            self.logger.warning(f"Row missing failed: {e}")
        
        try:
            results['patterns'] = analyzer.missing_patterns()
        except Exception as e:
            self.logger.warning(f"Missing patterns failed: {e}")
        
        try:
            results['pairwise'] = analyzer.pairwise_missing()
        except Exception as e:
            self.logger.warning(f"Pairwise missing failed: {e}")
        
        try:
            results['drop_impact'] = analyzer.drop_impact_analysis().to_dict('records')
        except Exception as e:
            self.logger.warning(f"Drop impact failed: {e}")
        
        return results
    
    def _detect_outliers(self, detector: OutlierDetector) -> Dict[str, Any]:
        """Detect outliers across all columns."""
        results = {}
        
        try:
            detector.detect_all_columns()
            results['by_column'] = detector.outlier_results
        except Exception as e:
            self.logger.warning(f"Outlier detection failed: {e}")
        
        try:
            summary = detector.outlier_summary()
            results['summary'] = summary.to_dict('records')
        except Exception as e:
            self.logger.warning(f"Outlier summary failed: {e}")
        
        return results
    
    def _generate_plots(self, df: pd.DataFrame, col_analyzer: ColumnAnalyzer,
                       corr_analyzer: CorrelationAnalyzer,
                       missing_analyzer: MissingAnalyzer,
                       outlier_detector: OutlierDetector):
        """Generate all plots."""
        plotter = EDAPlotter(style=self.style)
        
        # Overview grid
        try:
            path = os.path.join(self.plot_dir, 'overview.png')
            plotter.overview_grid(df, save_path=path)
            self.plot_paths['overview'] = path
        except Exception as e:
            self.logger.warning(f"Overview plot failed: {e}")
        
        # Distribution plots
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:10]:
            try:
                path = os.path.join(self.plot_dir, f'dist_{col}.png')
                plotter.distribution_plot(df[col], col, save_path=path)
                self.plot_paths[f'distribution_{col}'] = path
            except Exception as e:
                self.logger.warning(f"Distribution plot for {col} failed: {e}")
        
        # Categorical plots
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols[:5]:
            try:
                path = os.path.join(self.plot_dir, f'cat_{col}.png')
                plotter.categorical_plot(df[col], col, save_path=path)
                self.plot_paths[f'categorical_{col}'] = path
            except Exception as e:
                self.logger.warning(f"Categorical plot for {col} failed: {e}")
        
        # Correlation heatmaps
        if 'pearson' in self.results.get('correlations', {}):
            try:
                corr_data = self.results['correlations']['pearson']
                corr_df = pd.DataFrame(corr_data['matrix'])
                path = os.path.join(self.plot_dir, 'correlation_pearson.png')
                plotter.correlation_heatmap(corr_df, 'Pearson Correlation', save_path=path)
                self.plot_paths['correlation_pearson'] = path
            except Exception as e:
                self.logger.warning(f"Pearson heatmap failed: {e}")
        
        if 'cramers_v' in self.results.get('correlations', {}):
            try:
                cramers_data = self.results['correlations']['cramers_v']
                cramers_df = pd.DataFrame(cramers_data['matrix'])
                path = os.path.join(self.plot_dir, 'correlation_cramers.png')
                plotter.correlation_heatmap(cramers_df, "Cramér's V (Categorical)", save_path=path)
                self.plot_paths['correlation_cramers'] = path
            except Exception as e:
                self.logger.warning(f"Cramér's V heatmap failed: {e}")
        
        # Missing heatmap
        try:
            path = os.path.join(self.plot_dir, 'missing_heatmap.png')
            plotter.missing_heatmap(df, save_path=path)
            self.plot_paths['missing_heatmap'] = path
        except Exception as e:
            self.logger.warning(f"Missing heatmap failed: {e}")
        
        # Missing bar chart
        try:
            missing_df = pd.DataFrame(self.results['missing']['by_column'])
            path = os.path.join(self.plot_dir, 'missing_bars.png')
            plotter.missing_bar_chart(missing_df, save_path=path)
            self.plot_paths['missing_bars'] = path
        except Exception as e:
            self.logger.warning(f"Missing bar chart failed: {e}")
        
        # Outlier boxplots
        if len(numeric_cols) > 0:
            try:
                path = os.path.join(self.plot_dir, 'outlier_boxplots.png')
                plotter.outlier_boxplots(df, numeric_cols[:12].tolist(), save_path=path)
                self.plot_paths['outlier_boxplots'] = path
            except Exception as e:
                self.logger.warning(f"Outlier boxplots failed: {e}")
