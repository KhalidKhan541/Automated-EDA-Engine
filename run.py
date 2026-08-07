import argparse
import pandas as pd
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Automated EDA Report Engine')
    parser.add_argument('input', help='CSV file path')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--name', '-n', default='Dataset', help='Dataset name')
    parser.add_argument('--style', choices=['dark', 'light'], default='dark',
                       help='Report style theme')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample N rows for faster analysis')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    if not os.path.exists(args.input):
        logger.error(f"File not found: {args.input}")
        sys.exit(1)

    logger.info(f"Loading {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        sys.exit(1)

    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=42)
        logger.info(f"Sampled {len(df)} rows")

    logger.info(f"Dataset shape: {df.shape}")

    from src.pipeline import EDAPipeline

    pipeline = EDAPipeline(output_dir=args.output, style=args.style)
    report_path = pipeline.run_full_analysis(df, dataset_name=args.name)

    print(f"\nEDA Report generated: {report_path}")
    print(f"Open in browser to view the report.")

if __name__ == '__main__':
    main()
