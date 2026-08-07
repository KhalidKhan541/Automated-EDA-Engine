import os
import base64
import json
from datetime import datetime
from typing import Dict, Any, List

class HTMLReportGenerator:
    """Generate comprehensive HTML report using Jinja2 templates."""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.template = self._get_template()
    
    def generate_report(self, analysis_results: Dict[str, Any],
                       plot_paths: Dict[str, str],
                       output_path: str = None) -> str:
        """Generate full HTML report."""
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'eda_report.html')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        html = self._render_report(analysis_results, plot_paths)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    def _render_report(self, results: Dict[str, Any],
                      plot_paths: Dict[str, str]) -> str:
        """Render the complete HTML report."""
        dataset_info = results.get('dataset_info', {})
        column_stats = results.get('column_stats', {})
        correlations = results.get('correlations', {})
        missing = results.get('missing', {})
        outliers = results.get('outliers', {})
        
        # Build column sections
        column_html = ''
        for col_name, stats in column_stats.items():
            col_type = stats.get('type', 'unknown')
            column_html += self._render_column_section(col_name, stats, col_type)
        
        # Build correlation section
        correlation_html = self._render_correlation_section(correlations)
        
        # Build missing section
        missing_html = self._render_missing_section(missing)
        
        # Build outlier section
        outlier_html = self._render_outlier_section(outliers)
        
        # Build plots section
        plots_html = self._render_plots_section(plot_paths)
        
        # Summary stats
        summary_html = self._render_summary(dataset_info, column_stats)
        
        html = self.template.format(
            title=f"EDA Report - {dataset_info.get('name', 'Dataset')}",
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            summary=summary_html,
            column_sections=column_html,
            correlation_section=correlation_html,
            missing_section=missing_html,
            outlier_section=outlier_html,
            plots_section=plots_html,
        )
        
        return html
    
    def _render_summary(self, dataset_info: Dict, column_stats: Dict) -> str:
        """Render dataset summary section."""
        n_cols = len(column_stats)
        col_types = {}
        for stats in column_stats.values():
            t = stats.get('type', 'unknown')
            col_types[t] = col_types.get(t, 0) + 1
        
        type_badges = ''.join([
            f'<span class="badge badge-{t}">{t}: {c}</span>'
            for t, c in col_types.items()
        ])
        
        return f'''
        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-value">{dataset_info.get("n_rows", 0):,}</div>
                <div class="stat-label">Rows</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{n_cols}</div>
                <div class="stat-label">Columns</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{dataset_info.get("total_cells", 0):,}</div>
                <div class="stat-label">Total Cells</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{dataset_info.get("missing_cells", 0):,}</div>
                <div class="stat-label">Missing Cells</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{dataset_info.get("missing_pct", 0):.1f}%</div>
                <div class="stat-label">Missing %</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{dataset_info.get("memory_mb", 0):.1f} MB</div>
                <div class="stat-label">Memory</div>
            </div>
        </div>
        <div class="type-badges">{type_badges}</div>
        '''
    
    def _render_column_section(self, col_name: str, stats: Dict,
                              col_type: str) -> str:
        """Render analysis section for a single column."""
        content = f'''
        <div class="column-card" id="col-{col_name}">
            <div class="column-header">
                <h3>{col_name}</h3>
                <span class="badge badge-{col_type}">{col_type}</span>
            </div>
            <div class="column-stats">
                <div class="stat-row">
                    <span>Non-null:</span>
                    <span>{stats.get("n_non_null", 0):,} / {stats.get("n_total", 0):,} ({100 - stats.get("null_pct", 0):.1f}%)</span>
                </div>
                <div class="stat-row">
                    <span>Unique:</span>
                    <span>{stats.get("n_unique", 0):,} ({stats.get("unique_pct", 0):.1f}%)</span>
                </div>
                <div class="stat-row">
                    <span>Memory:</span>
                    <span>{stats.get("memory_bytes", 0) / 1024:.1f} KB</span>
                </div>
        '''
        
        if col_type == 'numeric':
            content += self._render_numeric_stats(stats)
        elif col_type == 'categorical':
            content += self._render_categorical_stats(stats)
        
        content += '</div></div>'
        return content
    
    def _render_numeric_stats(self, stats: Dict) -> str:
        """Render numeric column statistics."""
        percentiles = stats.get('percentiles', {})
        
        return f'''
            <div class="stats-section">
                <h4>Central Tendency</h4>
                <div class="stat-row"><span>Mean:</span><span>{stats.get("mean", 0):.4f}</span></div>
                <div class="stat-row"><span>Median:</span><span>{stats.get("median", 0):.4f}</span></div>
                <div class="stat-row"><span>Mode:</span><span>{stats.get("mode", 0):.4f}</span></div>
            </div>
            <div class="stats-section">
                <h4>Dispersion</h4>
                <div class="stat-row"><span>Std Dev:</span><span>{stats.get("std", 0):.4f}</span></div>
                <div class="stat-row"><span>Variance:</span><span>{stats.get("variance", 0):.4f}</span></div>
                <div class="stat-row"><span>IQR:</span><span>{stats.get("iqr", 0):.4f}</span></div>
                <div class="stat-row"><span>Range:</span><span>{stats.get("range", 0):.4f}</span></div>
                <div class="stat-row"><span>CV:</span><span>{stats.get("cv", 0):.4f}</span></div>
            </div>
            <div class="stats-section">
                <h4>Shape</h4>
                <div class="stat-row"><span>Skewness:</span><span>{stats.get("skewness", 0):.4f}</span></div>
                <div class="stat-row"><span>Kurtosis:</span><span>{stats.get("kurtosis", 0):.4f}</span></div>
                <div class="stat-row"><span>Normal:</span><span>{"Yes" if stats.get("is_normal") else "No"}</span></div>
            </div>
            <div class="stats-section">
                <h4>Quantiles</h4>
                <div class="stat-row"><span>P1:</span><span>{percentiles.get("p1", 0):.4f}</span></div>
                <div class="stat-row"><span>P5:</span><span>{percentiles.get("p5", 0):.4f}</span></div>
                <div class="stat-row"><span>P25:</span><span>{stats.get("q25", 0):.4f}</span></div>
                <div class="stat-row"><span>P50:</span><span>{stats.get("median", 0):.4f}</span></div>
                <div class="stat-row"><span>P75:</span><span>{stats.get("q75", 0):.4f}</span></div>
                <div class="stat-row"><span>P95:</span><span>{percentiles.get("p95", 0):.4f}</span></div>
                <div class="stat-row"><span>P99:</span><span>{percentiles.get("p99", 0):.4f}</span></div>
            </div>
            <div class="stats-section">
                <h4>Outliers</h4>
                <div class="stat-row"><span>IQR Outliers:</span><span>{stats.get("n_outliers_iqr", 0):,} ({stats.get("outlier_pct_iqr", 0):.1f}%)</span></div>
                <div class="stat-row"><span>Z-Score Outliers:</span><span>{stats.get("n_outliers_zscore", 0):,}</span></div>
            </div>
        '''
    
    def _render_categorical_stats(self, stats: Dict) -> str:
        """Render categorical column statistics."""
        categories = stats.get('categories', [])
        category_bars = ''
        
        max_count = max([c['count'] for c in categories]) if categories else 1
        
        for cat in categories[:10]:
            pct = cat['pct']
            bar_width = (cat['count'] / max_count) * 100 if max_count > 0 else 0
            category_bars += f'''
                <div class="category-row">
                    <span class="cat-name">{cat["value"][:30]}</span>
                    <div class="cat-bar-container">
                        <div class="cat-bar" style="width: {bar_width}%"></div>
                    </div>
                    <span class="cat-count">{cat["count"]:,} ({pct:.1f}%)</span>
                </div>
            '''
        
        return f'''
            <div class="stats-section">
                <h4>Categories</h4>
                <div class="stat-row"><span>Count:</span><span>{stats.get("n_categories", 0)}</span></div>
                <div class="stat-row"><span>Top:</span><span>{stats.get("top_category", "")}</span></div>
                <div class="stat-row"><span>Top %:</span><span>{stats.get("top_category_pct", 0):.1f}%</span></div>
                <div class="stat-row"><span>Entropy:</span><span>{stats.get("entropy", 0):.4f}</span></div>
                <div class="stat-row"><span>Norm Entropy:</span><span>{stats.get("normalized_entropy", 0):.4f}</span></div>
            </div>
            <div class="stats-section">
                <h4>Value Distribution</h4>
                {category_bars}
            </div>
        '''
    
    def _render_correlation_section(self, correlations: Dict) -> str:
        """Render correlation analysis section."""
        pearson = correlations.get('pearson', {})
        spearman = correlations.get('spearman', {})
        cramers = correlations.get('cramers_v', {})
        
        return f'''
        <div class="section">
            <h2>Correlation Analysis</h2>
            <div class="correlation-tabs">
                <div class="tab">
                    <h3>Pearson Correlation</h3>
                    <div class="correlation-info">
                        <p>Measures linear relationship between numeric variables. Range: [-1, 1]</p>
                        <div class="corr-legend">
                            <span class="corr-strong">|r| > 0.7: Strong</span>
                            <span class="corr-moderate">0.4 < |r| < 0.7: Moderate</span>
                            <span class="corr-weak">|r| < 0.4: Weak</span>
                        </div>
                    </div>
                </div>
                <div class="tab">
                    <h3>Spearman Correlation</h3>
                    <p>Measures monotonic relationship (rank-based). Robust to non-linearity.</p>
                </div>
                <div class="tab">
                    <h3>Cramér's V (Categorical)</h3>
                    <p>Measures association between categorical variables. Range: [0, 1]</p>
                </div>
            </div>
        </div>
        '''
    
    def _render_missing_section(self, missing: Dict) -> str:
        """Render missing data section."""
        overall = missing.get('overall', {})
        col_missing = missing.get('by_column', [])
        
        missing_rows = ''
        if isinstance(col_missing, list):
            for item in col_missing[:20]:
                if isinstance(item, dict) and item.get('n_missing', 0) > 0:
                    pct = item.get('pct_missing', 0)
                    severity = 'critical' if pct > 50 else 'warning' if pct > 20 else 'normal'
                    missing_rows += f'''
                        <div class="stat-row missing-{severity}">
                            <span>{item.get("column", "")}</span>
                            <span>{item.get("n_missing", 0):,} ({pct:.1f}%)</span>
                        </div>
                    '''
        
        return f'''
        <div class="section">
            <h2>Missing Data Analysis</h2>
            <div class="summary-grid">
                <div class="stat-card">
                    <div class="stat-value">{overall.get("total_missing", 0):,}</div>
                    <div class="stat-label">Total Missing</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overall.get("overall_missing_pct", 0):.1f}%</div>
                    <div class="stat-label">Missing %</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overall.get("n_complete_rows", 0):,}</div>
                    <div class="stat-label">Complete Rows</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overall.get("complete_row_pct", 0):.1f}%</div>
                    <div class="stat-label">Complete %</div>
                </div>
            </div>
            <div class="missing-columns">
                <h3>Missing by Column</h3>
                {missing_rows}
            </div>
        </div>
        '''
    
    def _render_outlier_section(self, outliers: Dict) -> str:
        """Render outlier analysis section."""
        summary = outliers.get('summary', [])
        
        rows = ''
        if isinstance(summary, list):
            for item in summary:
                if isinstance(item, dict):
                    rows += f'''
                        <tr>
                            <td>{item.get("column", "")}</td>
                            <td>{item.get("iqr_outliers", 0)}</td>
                            <td>{item.get("zscore_outliers", 0)}</td>
                            <td>{item.get("mad_outliers", 0)}</td>
                            <td>{item.get("iforest_outliers", 0)}</td>
                            <td>{item.get("consensus_outliers", 0)}</td>
                        </tr>
                    '''
        
        return f'''
        <div class="section">
            <h2>Outlier Detection</h2>
            <p>Detects outliers using multiple methods: IQR, Z-Score, Modified Z-Score (MAD), and Isolation Forest.</p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>IQR</th>
                        <th>Z-Score</th>
                        <th>MAD</th>
                        <th>Isolation Forest</th>
                        <th>Consensus</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        '''
    
    def _render_plots_section(self, plot_paths: Dict[str, str]) -> str:
        """Render plots section with images."""
        plot_html = ''
        
        for name, path in plot_paths.items():
            if path and os.path.exists(path):
                img_data = self._encode_image(path)
                plot_html += f'''
                <div class="plot-card">
                    <h3>{name.replace('_', ' ').title()}</h3>
                    <img src="data:image/png;base64,{img_data}" alt="{name}">
                </div>
                '''
        
        return f'''
        <div class="section">
            <h2>Visualizations</h2>
            <div class="plots-grid">
                {plot_html}
            </div>
        </div>
        '''
    
    def _encode_image(self, path: str) -> str:
        """Encode image file to base64."""
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _get_template(self) -> str:
        """Return the HTML template with dark theme CSS."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent: #6366f1;
            --accent-secondary: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --border: #334155;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--accent), var(--accent-secondary));
            padding: 3rem 2rem;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        header .subtitle {{
            opacity: 0.9;
            font-size: 1rem;
        }}
        
        .section {{
            margin: 2rem 0;
            padding: 1.5rem;
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        
        .section h2 {{
            color: var(--accent);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border);
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .stat-card {{
            background: var(--bg-primary);
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            border: 1px solid var(--border);
        }}
        
        .stat-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--accent);
        }}
        
        .stat-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.3rem;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .badge-numeric {{ background: rgba(99, 102, 241, 0.2); color: var(--accent); }}
        .badge-categorical {{ background: rgba(139, 92, 246, 0.2); color: var(--accent-secondary); }}
        .badge-datetime {{ background: rgba(16, 185, 129, 0.2); color: var(--success); }}
        .badge-boolean {{ background: rgba(245, 158, 11, 0.2); color: var(--warning); }}
        .badge-text {{ background: rgba(236, 72, 153, 0.2); color: #ec4899; }}
        
        .column-card {{
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
        }}
        
        .column-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
        }}
        
        .column-header h3 {{
            font-family: monospace;
            font-size: 1.1rem;
        }}
        
        .column-stats {{
            padding: 1rem 1.5rem;
        }}
        
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.4rem 0;
            border-bottom: 1px solid rgba(51, 65, 85, 0.5);
        }}
        
        .stat-row:last-child {{
            border-bottom: none;
        }}
        
        .stats-section {{
            margin: 1rem 0;
            padding: 1rem;
            background: rgba(30, 41, 59, 0.5);
            border-radius: 6px;
        }}
        
        .stats-section h4 {{
            color: var(--accent);
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }}
        
        .category-row {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0.3rem 0;
        }}
        
        .cat-name {{
            width: 200px;
            font-size: 0.85rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .cat-bar-container {{
            flex: 1;
            height: 8px;
            background: var(--bg-primary);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .cat-bar {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--accent-secondary));
            border-radius: 4px;
        }}
        
        .cat-count {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            width: 120px;
            text-align: right;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        .data-table th,
        .data-table td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        .data-table th {{
            background: var(--bg-primary);
            color: var(--accent);
            font-weight: 600;
        }}
        
        .data-table tr:hover {{
            background: rgba(99, 102, 241, 0.05);
        }}
        
        .missing-critical {{ background: rgba(239, 68, 68, 0.1); }}
        .missing-warning {{ background: rgba(245, 158, 11, 0.1); }}
        
        .plots-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 1.5rem;
        }}
        
        .plot-card {{
            background: var(--bg-primary);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        
        .plot-card h3 {{
            margin-bottom: 1rem;
            color: var(--accent);
            font-size: 1rem;
        }}
        
        .plot-card img {{
            width: 100%;
            border-radius: 4px;
        }}
        
        .correlation-tabs {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}
        
        .tab {{
            background: var(--bg-primary);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        
        .tab h3 {{
            color: var(--accent);
            margin-bottom: 0.5rem;
        }}
        
        .corr-legend {{
            display: flex;
            gap: 1rem;
            margin-top: 0.5rem;
            font-size: 0.8rem;
        }}
        
        .corr-strong {{ color: var(--danger); }}
        .corr-moderate {{ color: var(--warning); }}
        .corr-weak {{ color: var(--success); }}
        
        .type-badges {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin: 1rem 0;
        }}
        
        .nav {{
            position: sticky;
            top: 0;
            background: var(--bg-primary);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border);
            z-index: 100;
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }}
        
        .nav a {{
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.9rem;
            padding: 0.3rem 0.8rem;
            border-radius: 6px;
            transition: all 0.2s;
        }}
        
        .nav a:hover {{
            background: var(--accent);
            color: white;
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}
            .summary-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .plots-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>{title}</h1>
        <p class="subtitle">Generated on {generated_at} | Automated EDA Engine</p>
    </header>
    
    <nav class="nav">
        <a href="#summary">Summary</a>
        <a href="#columns">Columns</a>
        <a href="#correlations">Correlations</a>
        <a href="#missing">Missing Data</a>
        <a href="#outliers">Outliers</a>
        <a href="#plots">Plots</a>
    </nav>
    
    <div class="container">
        <div id="summary" class="section">
            <h2>Dataset Overview</h2>
            {summary}
        </div>
        
        <div id="columns" class="section">
            <h2>Column Analysis</h2>
            {column_sections}
        </div>
        
        <div id="correlations">
            {correlation_section}
        </div>
        
        <div id="missing">
            {missing_section}
        </div>
        
        <div id="outliers">
            {outlier_section}
        </div>
        
        <div id="plots">
            {plots_section}
        </div>
    </div>
    
    <footer>
        <p>Automated EDA Engine v1.0 | Built with Python, Pandas, Matplotlib, SciPy</p>
    </footer>
</body>
</html>'''
