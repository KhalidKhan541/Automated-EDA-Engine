import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from typing import Dict, Any, Optional
import io
import base64

class EDAPlotter:
    """Generate distribution plots, correlation heatmaps, and missing data heatmaps."""
    
    COLORS = {
        'primary': '#6366f1',
        'secondary': '#8b5cf6',
        'accent': '#ec4899',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'dark': '#1e293b',
        'light': '#f1f5f9',
    }
    
    def __init__(self, style: str = 'dark'):
        if style == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#0f172a'
            self.text_color = '#e2e8f0'
            self.grid_color = '#334155'
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.bg_color = '#ffffff'
            self.text_color = '#1e293b'
            self.grid_color = '#e2e8f0'
    
    def distribution_plot(self, series: pd.Series, col_name: str,
                         save_path: str = None) -> str:
        """Histogram + KDE + Q-Q plot for a numeric column."""
        data = series.dropna()
        if len(data) < 5:
            return ''
        
        fig = plt.figure(figsize=(14, 5))
        gs = gridspec.GridSpec(1, 3, wspace=0.3)
        
        # Histogram + KDE
        ax1 = fig.add_subplot(gs[0])
        ax1.hist(data, bins=50, density=True, alpha=0.7, color=self.COLORS['primary'],
                edgecolor='white', linewidth=0.5)
        
        try:
            kde_x = np.linspace(data.min(), data.max(), 200)
            kde = stats.gaussian_kde(data)
            ax1.plot(kde_x, kde(kde_x), color=self.COLORS['accent'], linewidth=2, label='KDE')
        except Exception:
            pass
        
        ax1.axvline(data.mean(), color=self.COLORS['warning'], linestyle='--',
                   linewidth=1.5, label=f'Mean: {data.mean():.2f}')
        ax1.axvline(data.median(), color=self.COLORS['success'], linestyle='--',
                   linewidth=1.5, label=f'Median: {data.median():.2f}')
        ax1.set_xlabel(col_name, fontsize=11, color=self.text_color)
        ax1.set_ylabel('Density', fontsize=11, color=self.text_color)
        ax1.set_title('Distribution', fontsize=13, fontweight='bold', color=self.text_color)
        ax1.legend(fontsize=9)
        ax1.set_facecolor(self.bg_color)
        ax1.grid(True, alpha=0.3, color=self.grid_color)
        
        # Box plot
        ax2 = fig.add_subplot(gs[1])
        bp = ax2.boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor=self.COLORS['primary'], alpha=0.7),
                        medianprops=dict(color=self.COLORS['accent'], linewidth=2),
                        whiskerprops=dict(color=self.text_color),
                        capprops=dict(color=self.text_color),
                        flierprops=dict(markerfacecolor=self.COLORS['danger'], markersize=4))
        ax2.set_title('Box Plot', fontsize=13, fontweight='bold', color=self.text_color)
        ax2.set_facecolor(self.bg_color)
        ax2.grid(True, alpha=0.3, color=self.grid_color, axis='y')
        
        # Q-Q plot
        ax3 = fig.add_subplot(gs[2])
        try:
            stats.probplot(data, dist="norm", plot=ax3)
            ax3.get_lines()[0].set(markerfacecolor=self.COLORS['primary'],
                                   markeredgecolor='white', markersize=5, alpha=0.7)
            ax3.get_lines()[1].set(color=self.COLORS['accent'], linewidth=2)
        except Exception:
            ax3.text(0.5, 0.5, 'Q-Q not available', ha='center', va='center',
                    transform=ax3.transAxes, color=self.text_color)
        
        ax3.set_title('Q-Q Plot', fontsize=13, fontweight='bold', color=self.text_color)
        ax3.set_facecolor(self.bg_color)
        ax3.grid(True, alpha=0.3, color=self.grid_color)
        
        fig.suptitle(f'{col_name}', fontsize=15, fontweight='bold',
                    color=self.text_color, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')
        plt.close()
    
    def categorical_plot(self, series: pd.Series, col_name: str,
                        save_path: str = None):
        """Bar chart for categorical columns."""
        data = series.dropna()
        if len(data) == 0:
            return
        
        value_counts = data.value_counts().head(20)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(value_counts)))
        
        bars = ax.barh(range(len(value_counts)), value_counts.values,
                      color=colors, edgecolor='white', linewidth=0.5)
        
        ax.set_yticks(range(len(value_counts)))
        ax.set_yticklabels(value_counts.index, fontsize=10, color=self.text_color)
        ax.set_xlabel('Count', fontsize=11, color=self.text_color)
        ax.set_title(f'{col_name} - Value Counts', fontsize=13,
                    fontweight='bold', color=self.text_color)
        ax.set_facecolor(self.bg_color)
        ax.grid(True, alpha=0.3, color=self.grid_color, axis='x')
        
        for bar, val in zip(bars, value_counts.values):
            ax.text(bar.get_width() + max(value_counts.values) * 0.01,
                   bar.get_y() + bar.get_height()/2,
                   f'{val:,}', va='center', fontsize=9, color=self.text_color)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')
        plt.close()
    
    def correlation_heatmap(self, corr_matrix: pd.DataFrame, title: str,
                           save_path: str = None):
        """Correlation heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        im = ax.imshow(corr_matrix.values, cmap='RdYlBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right',
                          fontsize=10, color=self.text_color)
        ax.set_yticklabels(corr_matrix.columns, fontsize=10, color=self.text_color)
        
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                if i <= j:
                    color = 'white' if abs(corr_matrix.values[i, j]) > 0.5 else 'black'
                    ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                           ha='center', va='center', fontsize=8, color=color)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation', fontsize=11, color=self.text_color)
        cbar.ax.tick_params(labelcolor=self.text_color)
        
        ax.set_title(title, fontsize=14, fontweight='bold', color=self.text_color, pad=20)
        ax.set_facecolor(self.bg_color)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')
        plt.close()
    
    def missing_heatmap(self, df: pd.DataFrame, save_path: str = None,
                       max_rows: int = 100):
        """Missing data heatmap."""
        sample_df = df.head(max_rows) if len(df) > max_rows else df
        missing_matrix = sample_df.isna().astype(int)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(missing_matrix.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha='right',
                          fontsize=10, color=self.text_color)
        ax.set_yticks(range(0, len(sample_df), max(1, len(sample_df) // 20)))
        ax.set_yticklabels(range(0, len(sample_df), max(1, len(sample_df) // 20)),
                          fontsize=9, color=self.text_color)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Present', 'Missing'])
        cbar.ax.tick_params(labelcolor=self.text_color)
        
        title = f'Missing Data Heatmap'
        if len(df) > max_rows:
            title += f' (showing first {max_rows} rows)'
        ax.set_title(title, fontsize=14, fontweight='bold', color=self.text_color, pad=15)
        ax.set_facecolor(self.bg_color)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')
        plt.close()
    
    def outlier_boxplots(self, df: pd.DataFrame, numeric_cols: list,
                        save_path: str = None):
        """Combined boxplots for outlier visualization."""
        n_cols = min(len(numeric_cols), 6)
        n_rows = (len(numeric_cols) - 1) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, col in enumerate(numeric_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            
            ax = axes[row, col_idx]
            data = df[col].dropna()
            
            bp = ax.boxplot(data, vert=True, patch_artist=True,
                          boxprops=dict(facecolor=self.COLORS['primary'], alpha=0.7),
                          medianprops=dict(color=self.COLORS['accent'], linewidth=2),
                          flierprops=dict(markerfacecolor=self.COLORS['danger'], markersize=3))
            
            ax.set_title(col[:15], fontsize=10, fontweight='bold', color=self.text_color)
            ax.set_facecolor(self.bg_color)
            ax.grid(True, alpha=0.3, color=self.grid_color, axis='y')
        
        for idx in range(len(numeric_cols), n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            axes[row, col_idx].set_visible(False)
        
        fig.suptitle('Outlier Detection - Boxplots', fontsize=14,
                    fontweight='bold', color=self.text_color)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')
        plt.close()
    
    def missing_bar_chart(self, missing_df: pd.DataFrame, save_path: str = None):
        """Bar chart of missing data per column."""
        data = missing_df[missing_df['n_missing'] > 0].head(20)
        
        if len(data) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = [self.COLORS['danger'] if pct > 50 else
                 self.COLORS['warning'] if pct > 20 else
                 self.COLORS['success'] for pct in data['pct_missing']]
        
        bars = ax.barh(range(len(data)), data['pct_missing'],
                      color=colors, edgecolor='white', linewidth=0.5, height=0.7)
        
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['column'], fontsize=10, color=self.text_color)
        ax.set_xlabel('% Missing', fontsize=11, color=self.text_color)
        ax.set_title('Missing Data by Column', fontsize=13,
                    fontweight='bold', color=self.text_color)
        ax.set_facecolor(self.bg_color)
        ax.grid(True, alpha=0.3, color=self.grid_color, axis='x')
        
        for bar, pct in zip(bars, data['pct_missing']):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', va='center', fontsize=9, color=self.text_color)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')
        plt.close()
    
    def overview_grid(self, df: pd.DataFrame, save_path: str = None):
        """Overview grid with dataset summary stats."""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3)
        
        # 1. Missing data summary
        ax1 = fig.add_subplot(gs[0, 0])
        missing_pct = df.isna().mean() * 100
        colors = ['#10b981' if p < 5 else '#f59e0b' if p < 20 else '#ef4444' for p in missing_pct]
        ax1.barh(range(len(missing_pct)), missing_pct, color=colors, height=0.7)
        ax1.set_yticks(range(len(missing_pct)))
        ax1.set_yticklabels(missing_pct.index, fontsize=8, color=self.text_color)
        ax1.set_title('Missing %', fontsize=12, fontweight='bold', color=self.text_color)
        ax1.set_facecolor(self.bg_color)
        ax1.grid(True, alpha=0.3, color=self.grid_color, axis='x')
        
        # 2. Column types
        ax2 = fig.add_subplot(gs[0, 1])
        dtypes = df.dtypes.value_counts()
        ax2.pie(dtypes.values, labels=dtypes.index.astype(str),
               colors=plt.cm.Set3(np.linspace(0, 1, len(dtypes))),
               autopct='%1.0f%%', textprops={'color': self.text_color})
        ax2.set_title('Column Types', fontsize=12, fontweight='bold', color=self.text_color)
        
        # 3. Dataset shape
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        info_text = f"Rows: {len(df):,}\nColumns: {len(df.columns)}\nCells: {len(df) * len(df.columns):,}\nMissing: {df.isna().sum().sum():,}\nComplete: {(~df.isna()).all(axis=1).sum():,}"
        ax3.text(0.5, 0.5, info_text, ha='center', va='center',
                transform=ax3.transAxes, fontsize=14, color=self.text_color,
                bbox=dict(boxstyle='round', facecolor=self.bg_color,
                         edgecolor=self.grid_color, alpha=0.8))
        ax3.set_title('Dataset Info', fontsize=12, fontweight='bold', color=self.text_color)
        
        # 4. Numeric distributions (first 6 columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
        ax4 = fig.add_subplot(gs[1, :])
        for i, col in enumerate(numeric_cols):
            data = df[col].dropna()
            if len(data) > 0:
                kde_x = np.linspace(data.min(), data.max(), 100)
                kde = stats.gaussian_kde(data)
                ax4.plot(kde_x, kde(kde_x), label=col, linewidth=2, alpha=0.8)
        ax4.set_title('Numeric Distributions (KDE)', fontsize=12,
                     fontweight='bold', color=self.text_color)
        ax4.legend(fontsize=9, facecolor=self.bg_color, edgecolor=self.grid_color,
                  labelcolor=self.text_color)
        ax4.set_facecolor(self.bg_color)
        ax4.grid(True, alpha=0.3, color=self.grid_color)
        
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold',
                    color=self.text_color, y=1.02)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')
        plt.close()
    
    def fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
