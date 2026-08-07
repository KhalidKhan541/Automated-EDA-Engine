# Automated EDA Report Engine

A from-scratch EDA profiling tool that generates comprehensive HTML reports with dark theme, 30+ statistics per column, distribution plots, correlation analysis, and outlier detection.

## Features

| Feature | Description |
|---------|-------------|
| **Auto Type Detection** | Numeric, categorical, datetime, boolean, text, constant |
| **30+ Stats/Column** | Mean, median, mode, std, variance, IQR, skewness, kurtosis, quantiles, percentiles, normality tests |
| **Distribution Plots** | Histogram + KDE + Box Plot + Q-Q Plot |
| **Correlation Matrix** | Pearson + Spearman + Cramér's V (categorical) + Eta² |
| **Missing Heatmap** | Visual missing data patterns with row/column analysis |
| **Outlier Detection** | IQR + Z-Score + Modified Z-Score (MAD) + Isolation Forest |
| **Jinja2 HTML Report** | Dark theme, responsive, embedded charts |

## Quick Start

```bash
pip install -r requirements.txt

# Generate EDA report
python run.py data.csv

# With options
python run.py data.csv --output reports --name "Customer Data" --style dark --sample 10000
```

## Architecture

```
Automated-EDA-Engine/
├── src/
│   ├── column_analyzer.py    # 30+ stats per column with auto type detection
│   ├── correlation.py        # Pearson, Spearman, Cramér's V, Eta²
│   ├── missing_analyzer.py   # Missing patterns, heatmap, pairwise
│   ├── outlier_detector.py   # IQR, Z-score, MAD, Isolation Forest
│   ├── plotter.py            # Distribution, correlation, missing heatmaps
│   ├── report_generator.py   # Jinja2 HTML template with dark theme
│   └── pipeline.py           # Main orchestrator
├── run.py                    # CLI entry point
├── requirements.txt
└── README.md
```

## Dependencies

- numpy, pandas, scipy - Core data analysis
- scikit-learn - Isolation Forest outlier detection
- matplotlib - Visualization
- jinja2 - HTML report templating
