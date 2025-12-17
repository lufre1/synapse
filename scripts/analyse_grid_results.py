import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import re
import os
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze grid search results from CSV file')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                       help='Path to the input CSV file containing grid search results')
    parser.add_argument('--output-dir', '-o', type=str, default='./analysis_results',
                       help='Directory to save analysis results and figures (default: ./analysis_results)')
    return parser.parse_args()

def extract_hyperparams(dataset_name):
    """Extract hyperparameters from dataset name."""
    patterns = {
        'seed': r'sd(\d+)',
        'ft': r'ft(\d+)',
        'bt': r'bt(\d+)',
    }
    
    params = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, dataset_name)
        if match:
            value = match.group(1)
            # Convert numeric values appropriately
            if key in ['seed', 'ft', 'bt']:
                try:
                    value = float(value) if 'e' in value.lower() else int(value)
                except:
                    pass
            params[key] = value
    
    return params

def load_and_process_data(input_file):
    """Load CSV data and extract hyperparameters."""
    df = pd.read_csv(input_file)
    
    # Extract hyperparameters from dataset names
    hyperparams_df = df['dataset'].apply(extract_hyperparams).apply(pd.Series)
    df_with_params = pd.concat([df, hyperparams_df], axis=1)
    
    return df, df_with_params

def generate_basic_statistics(df):
    """Generate basic statistics about the results."""
    print("=== GRID SEARCH RESULTS ANALYSIS ===\n")
    
    print("Dataset count:", len(df))
    print("Average F1-Score:", df['f1-score'].mean())
    print("Average Precision:", df['precision'].mean())
    print("Average Recall:", df['recall'].mean())
    print("Average SBD Score:", df['SBD score'].mean())
    
    print("\n=== PERFORMANCE DISTRIBUTION ===")
    print("F1-Score Statistics:")
    print(df['f1-score'].describe())
    
    print("\nSBD Score Statistics:")
    print(df['SBD score'].describe())
    
    return {
        'dataset_count': len(df),
        'avg_f1': df['f1-score'].mean(),
        'avg_precision': df['precision'].mean(),
        'avg_recall': df['recall'].mean(),
        'avg_sbd': df['SBD score'].mean()
    }

def analyze_by_parameters(df_with_params, output_dir):
    """Analyze performance by different hyperparameters."""
    analysis_results = {}
    
    # Analyze by seed (sd) parameter
    if 'seed' in df_with_params.columns:
        print("\n=== PERFORMANCE BY SEED ===")
        seed_performance = df_with_params.groupby('seed')[['f1-score', 'SBD score']].mean()
        print(seed_performance)
        analysis_results['seed_performance'] = seed_performance
    
    # Analyze by fine-tuning parameter (ft)
    if 'ft' in df_with_params.columns:
        print("\n=== PERFORMANCE BY FOREGROUND THRESHOLD PARAMETER ===")
        ft_performance = df_with_params.groupby('ft')[['f1-score', 'SBD score']].mean()
        print(ft_performance)
        analysis_results['ft_performance'] = ft_performance
    
    # Analyze by boundary threshold parameter (bt)
    if 'bt' in df_with_params.columns:
        print("\n=== PERFORMANCE BY BOUNDARY THRESHOLD PARAMETER ===")
        bt_performance = df_with_params.groupby('bt')[['f1-score', 'SBD score']].mean()
        print(bt_performance)
        analysis_results['bt_performance'] = bt_performance
    
    return analysis_results

def create_parameter_influence_analysis(df_with_params, output_dir):
    """Create comprehensive analysis of how parameters influence performance."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create figure with subplots for parameter influence analysis
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: F1 Score vs Seed
    if 'seed' in df_with_params.columns:
        plt.subplot(3, 4, 1)
        seed_data = df_with_params.groupby('seed')[['f1-score', 'SBD score']].mean()
        x = range(len(seed_data))
        plt.plot(x, seed_data['f1-score'], marker='o', linewidth=2, markersize=8, label='F1-Score')
        plt.plot(x, seed_data['SBD score'], marker='s', linewidth=2, markersize=8, label='SBD Score')
        plt.xticks(x, seed_data.index)
        plt.xlabel('Seed')
        plt.ylabel('Score')
        plt.title('F1 Score & SBD Score vs Seed')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 2: F1 Score vs Foreground Threshold
    if 'ft' in df_with_params.columns:
        plt.subplot(3, 4, 2)
        ft_data = df_with_params.groupby('ft')[['f1-score', 'SBD score']].mean()
        x = range(len(ft_data))
        plt.plot(x, ft_data['f1-score'], marker='o', linewidth=2, markersize=8, label='F1-Score')
        plt.plot(x, ft_data['SBD score'], marker='s', linewidth=2, markersize=8, label='SBD Score')
        plt.xticks(x, ft_data.index)
        plt.xlabel('Foreground Threshold')
        plt.ylabel('Score')
        plt.title('F1 Score & SBD Score vs Foreground Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 3: F1 Score vs Boundary Threshold
    if 'bt' in df_with_params.columns:
        plt.subplot(3, 4, 3)
        bt_data = df_with_params.groupby('bt')[['f1-score', 'SBD score']].mean()
        x = range(len(bt_data))
        plt.plot(x, bt_data['f1-score'], marker='o', linewidth=2, markersize=8, label='F1-Score')
        plt.plot(x, bt_data['SBD score'], marker='s', linewidth=2, markersize=8, label='SBD Score')
        plt.xticks(x, bt_data.index)
        plt.xlabel('Boundary Threshold')
        plt.ylabel('Score')
        plt.title('F1 Score & SBD Score vs Boundary Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot of F1 vs SBD by Seed
    if 'seed' in df_with_params.columns:
        plt.subplot(3, 4, 4)
        scatter = plt.scatter(df_with_params['seed'], df_with_params['f1-score'], 
                            c=df_with_params['SBD score'], cmap='viridis', alpha=0.7, s=100)
        plt.xlabel('Seed')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Seed (Color: SBD Score)')
        plt.colorbar(scatter, label='SBD Score')
        plt.grid(True, alpha=0.3)
    
    # Plot 5: Scatter plot of F1 vs SBD by Foreground Threshold
    if 'ft' in df_with_params.columns:
        plt.subplot(3, 4, 5)
        scatter = plt.scatter(df_with_params['ft'], df_with_params['f1-score'], 
                            c=df_with_params['SBD score'], cmap='viridis', alpha=0.7, s=100)
        plt.xlabel('Foreground Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Foreground Threshold (Color: SBD Score)')
        plt.colorbar(scatter, label='SBD Score')
        plt.grid(True, alpha=0.3)
    
    # Plot 6: Scatter plot of F1 vs SBD by Boundary Threshold
    if 'bt' in df_with_params.columns:
        plt.subplot(3, 4, 6)
        scatter = plt.scatter(df_with_params['bt'], df_with_params['f1-score'], 
                            c=df_with_params['SBD score'], cmap='viridis', alpha=0.7, s=100)
        plt.xlabel('Boundary Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Boundary Threshold (Color: SBD Score)')
        plt.colorbar(scatter, label='SBD Score')
        plt.grid(True, alpha=0.3)
    
    # Plot 7: Box plot of F1 scores by Seed
    if 'seed' in df_with_params.columns:
        plt.subplot(3, 4, 7)
        df_with_params.boxplot(column='f1-score', by='seed', ax=plt.gca())
        plt.title('F1 Score Distribution by Seed')
        plt.suptitle('')  # Remove automatic title
        plt.grid(True, alpha=0.3)
    
    # Plot 8: Box plot of F1 scores by Foreground Threshold
    if 'ft' in df_with_params.columns:
        plt.subplot(3, 4, 8)
        df_with_params.boxplot(column='f1-score', by='ft', ax=plt.gca())
        plt.title('F1 Score Distribution by Foreground Threshold')
        plt.suptitle('')  # Remove automatic title
        plt.grid(True, alpha=0.3)
    
    # Plot 9: Box plot of F1 scores by Boundary Threshold
    if 'bt' in df_with_params.columns:
        plt.subplot(3, 4, 9)
        df_with_params.boxplot(column='f1-score', by='bt', ax=plt.gca())
        plt.title('F1 Score Distribution by Boundary Threshold')
        plt.suptitle('')  # Remove automatic title
        plt.grid(True, alpha=0.3)
    
    # Plot 10: Combined parameter influence heatmap (if we have all three parameters)
    if all(param in df_with_params.columns for param in ['seed', 'ft', 'bt']):
        plt.subplot(3, 4, 10)
        # Create pivot table for heatmap
        pivot_data = df_with_params.pivot_table(values='f1-score', index='seed', columns='ft')
        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', cbar_kws={'label': 'F1 Score'})
        plt.title('F1 Score Heatmap (Seed vs Foreground Threshold)')
        plt.xlabel('Foreground Threshold')
        plt.ylabel('Seed')
    
    # Plot 11: Combined parameter influence heatmap (Seed vs BT)
    if all(param in df_with_params.columns for param in ['seed', 'bt']):
        plt.subplot(3, 4, 11)
        pivot_data = df_with_params.pivot_table(values='f1-score', index='seed', columns='bt')
        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', cbar_kws={'label': 'F1 Score'})
        plt.title('F1 Score Heatmap (Seed vs Boundary Threshold)')
        plt.xlabel('Boundary Threshold')
        plt.ylabel('Seed')
    
    # Plot 12: Combined parameter influence heatmap (FT vs BT)
    if all(param in df_with_params.columns for param in ['ft', 'bt']):
        plt.subplot(3, 4, 12)
        pivot_data = df_with_params.pivot_table(values='f1-score', index='ft', columns='bt')
        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', cbar_kws={'label': 'F1 Score'})
        plt.title('F1 Score Heatmap (Foreground vs Boundary Threshold)')
        plt.xlabel('Boundary Threshold')
        plt.ylabel('Foreground Threshold')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'parameter_influence_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved parameter influence analysis to: {output_path}")

def analyze_parameter_relationships(df_with_params, output_dir):
    """Analyze relationships between parameters and performance metrics."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n=== PARAMETER RELATIONSHIP ANALYSIS ===")
    
    # Analyze each parameter's effect on both metrics
    if 'seed' in df_with_params.columns:
        print("\n--- SEED EFFECT ANALYSIS ---")
        seed_corr_f1 = df_with_params['seed'].corr(df_with_params['f1-score'])
        seed_corr_sbd = df_with_params['seed'].corr(df_with_params['SBD score'])
        print(f"Correlation between Seed and F1-Score: {seed_corr_f1:.4f}")
        print(f"Correlation between Seed and SBD Score: {seed_corr_sbd:.4f}")
    
    if 'ft' in df_with_params.columns:
        print("\n--- FOREGROUND THRESHOLD EFFECT ANALYSIS ---")
        ft_corr_f1 = df_with_params['ft'].corr(df_with_params['f1-score'])
        ft_corr_sbd = df_with_params['ft'].corr(df_with_params['SBD score'])
        print(f"Correlation between Foreground Threshold and F1-Score: {ft_corr_f1:.4f}")
        print(f"Correlation between Foreground Threshold and SBD Score: {ft_corr_sbd:.4f}")
    
    if 'bt' in df_with_params.columns:
        print("\n--- BOUNDARY THRESHOLD EFFECT ANALYSIS ---")
        bt_corr_f1 = df_with_params['bt'].corr(df_with_params['f1-score'])
        bt_corr_sbd = df_with_params['bt'].corr(df_with_params['SBD score'])
        print(f"Correlation between Boundary Threshold and F1-Score: {bt_corr_f1:.4f}")
        print(f"Correlation between Boundary Threshold and SBD Score: {bt_corr_sbd:.4f}")
    
    # Create detailed parameter analysis
    param_analysis = []
    
    if 'seed' in df_with_params.columns:
        seed_stats = df_with_params.groupby('seed').agg({
            'f1-score': ['mean', 'std', 'count'],
            'SBD score': ['mean', 'std']
        }).round(4)
        print("\n--- SEED STATISTICS ---")
        print(seed_stats)
        param_analysis.append(('seed', seed_stats))
    
    if 'ft' in df_with_params.columns:
        ft_stats = df_with_params.groupby('ft').agg({
            'f1-score': ['mean', 'std', 'count'],
            'SBD score': ['mean', 'std']
        }).round(4)
        print("\n--- FOREGROUND THRESHOLD STATISTICS ---")
        print(ft_stats)
        param_analysis.append(('ft', ft_stats))
    
    if 'bt' in df_with_params.columns:
        bt_stats = df_with_params.groupby('bt').agg({
            'f1-score': ['mean', 'std', 'count'],
            'SBD score': ['mean', 'std']
        }).round(4)
        print("\n--- BOUNDARY THRESHOLD STATISTICS ---")
        print(bt_stats)
        param_analysis.append(('bt', bt_stats))
    
    # Save parameter analysis to CSV
    if param_analysis:
        # Combine all parameter statistics into one CSV
        combined_stats = []
        for param_name, stats in param_analysis:
            # Flatten multi-index columns
            stats_flat = stats.copy()
            stats_flat.columns = ['_'.join(col).strip() if col[1] else col[0] for col in stats_flat.columns.values]
            stats_flat['parameter'] = param_name
            combined_stats.append(stats_flat.reset_index())
        
        if combined_stats:
            final_df = pd.concat(combined_stats, ignore_index=True)
            output_path = os.path.join(output_dir, 'parameter_statistics.csv')
            final_df.to_csv(output_path, index=False)
            print(f"\nSaved parameter statistics to: {output_path}")

def create_correlation_heatmap(df, output_dir):
    """Create and save correlation heatmap."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Metrics')
    plt.tight_layout()
    
    # Save the heatmap
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation heatmap to: {output_path}")
    
    return correlation_matrix

def analyze_top_models(df, output_dir):
    """Analyze and display top performing models."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get top 10 performing models
    top_models = df.nlargest(10, 'f1-score')
    
    # Save top models to CSV
    top_models_output = os.path.join(output_dir, 'top_performing_models.csv')
    top_models.to_csv(top_models_output, index=False)
    print(f"Saved top performing models to: {top_models_output}")
    
    # Print detailed information
    print("\n=== TOP 10 PERFORMING MODELS ===")
    for idx, row in top_models.iterrows():
        print(f"F1: {row['f1-score']:.4f} | SBD: {row['SBD score']:.4f} | "
              f"Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f}")
        print(f"Dataset: {row['dataset']}")
        print(f"Predictions/Actual: {row['#pred / #actual']}")
        print("-" * 80)

def check_anomalies(df):
    """Check for anomalies or special cases."""
    print("\n=== ANOMALY CHECK ===")
    perfect_f1 = len(df[df['f1-score'] == 1.0])
    perfect_sbd = len(df[df['SBD score'] > 0.9999])
    
    print(f"Models with perfect scores (F1=1.0): {perfect_f1}")
    print(f"Models with near-perfect SBD scores (>0.9999): {perfect_sbd}")
    
    # Check for any models with very high performance
    high_performance = df[df['f1-score'] > 0.99]
    print(f"Models with F1-score > 0.99: {len(high_performance)}")
    
    return {
        'perfect_f1_count': perfect_f1,
        'perfect_sbd_count': perfect_sbd,
        'high_performance_count': len(high_performance)
    }

def summarize_parameter_combinations(df_with_params):
    """Summarize parameter combinations."""
    print("\n=== PARAMETER COMBINATIONS ===")
    param_cols = [col for col in df_with_params.columns if col not in ['dataset', 'f1-score', 'precision', 'recall', 'SBD score', '#pred / #actual']]
    
    if param_cols:
        for param in param_cols:
            unique_vals = df_with_params[param].unique()
            print(f"{param}: {sorted(unique_vals)}")
    else:
        print("No hyperparameters found in dataset names.")

def save_summary_stats(stats, correlations, anomalies, output_dir):
    """Save summary statistics to text file."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    summary_file = os.path.join(output_dir, 'summary_statistics.txt')
    
    with open(summary_file, 'w') as f:
        f.write("GRID SEARCH RESULTS SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("BASIC STATISTICS:\n")
        f.write(f"Dataset count: {stats['dataset_count']}\n")
        f.write(f"Average F1-Score: {stats['avg_f1']:.4f}\n")
        f.write(f"Average Precision: {stats['avg_precision']:.4f}\n")
        f.write(f"Average Recall: {stats['avg_recall']:.4f}\n")
        f.write(f"Average SBD Score: {stats['avg_sbd']:.4f}\n\n")
        
        f.write("ANOMALY CHECK:\n")
        f.write(f"Perfect F1 scores: {anomalies['perfect_f1_count']}\n")
        f.write(f"Near-perfect SBD scores: {anomalies['perfect_sbd_count']}\n")
        f.write(f"High performance models: {anomalies['high_performance_count']}\n\n")
        
        f.write("CORRELATION MATRIX:\n")
        f.write(str(correlations))
    
    print(f"Saved summary statistics to: {summary_file}")

def main():
    """Main function to run the analysis."""
    # Parse arguments
    args = parse_arguments()
    
    # Load and process data
    print(f"Loading data from: {args.input_file}")
    df, df_with_params = load_and_process_data(args.input_file)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate basic statistics
    stats = generate_basic_statistics(df)
    
    # Analyze by parameters
    analysis_results = analyze_by_parameters(df_with_params, args.output_dir)
    
    # Create parameter influence analysis
    create_parameter_influence_analysis(df_with_params, args.output_dir)
    
    # Analyze parameter relationships
    analyze_parameter_relationships(df_with_params, args.output_dir)
    
    # Create correlation heatmap
    correlations = create_correlation_heatmap(df, args.output_dir)
    
    # Analyze top models
    analyze_top_models(df, args.output_dir)
    
    # Check for anomalies
    anomalies = check_anomalies(df)
    
    # Summarize parameter combinations
    summarize_parameter_combinations(df_with_params)
    
    # Save summary statistics
    save_summary_stats(stats, correlations, anomalies, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()