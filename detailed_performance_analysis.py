import xarray as xr
import numpy as np
import time
import os
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import json

def create_test_scenarios():
    """Create comprehensive test scenarios for different use cases"""
    
    scenarios = {
        'point_access': {
            'name': 'Point Access (Weather Station)',
            'description': 'Single point queries (e.g., weather station data)',
            'test': lambda ds, var: [
                ds[var].isel(north_south=i, east_west=j).values 
                for i, j in np.random.randint(0, [3000, 7200], size=(1000, 2))
            ]
        },
        'small_region': {
            'name': 'Small Region (City-scale)',
            'description': '100x100 pixel regions',
            'test': lambda ds, var: [
                ds[var].isel(
                    north_south=slice(i, i+100),
                    east_west=slice(j, j+100)
                ).values
                for i, j in np.random.randint(0, [2900, 7100], size=(50, 2))
            ]
        },
        'medium_region': {
            'name': 'Medium Region (State-scale)',
            'description': '500x500 pixel regions',
            'test': lambda ds, var: [
                ds[var].isel(
                    north_south=slice(i, i+500),
                    east_west=slice(j, j+500)
                ).values
                for i, j in np.random.randint(0, [2500, 6700], size=(20, 2))
            ]
        },
        'large_region': {
            'name': 'Large Region (Country-scale)',
            'description': '1500x1500 pixel regions',
            'test': lambda ds, var: [
                ds[var].isel(
                    north_south=slice(i, i+1500),
                    east_west=slice(j, j+1500)
                ).values
                for i, j in [(0, 0), (1500, 3600), (0, 5700)]
            ]
        },
        'lat_bands': {
            'name': 'Latitude Bands',
            'description': 'Full latitude strips (climate zones)',
            'test': lambda ds, var: [
                ds[var].isel(north_south=slice(i, i+100)).values
                for i in range(0, 3000, 300)
            ]
        },
        'lon_bands': {
            'name': 'Longitude Bands',
            'description': 'Full longitude strips (time zones)',
            'test': lambda ds, var: [
                ds[var].isel(east_west=slice(j, j+100)).values
                for j in range(0, 7200, 720)
            ]
        },
        'diagonal_transect': {
            'name': 'Diagonal Transect',
            'description': 'Diagonal line across dataset',
            'test': lambda ds, var: [
                ds[var].values[i, int(i * 7200/3000)]
                for i in range(0, 3000, 10)
            ]
        },
        'random_scatter': {
            'name': 'Random Scatter',
            'description': '1000 random pixels across dataset',
            'test': lambda ds, var: [
                ds[var].values[i, j]
                for i, j in np.random.randint(0, [3000, 7200], size=(1000, 2))
            ]
        },
        'checkerboard': {
            'name': 'Checkerboard Pattern',
            'description': 'Every other chunk access pattern',
            'test': lambda ds, var: [
                ds[var].isel(
                    north_south=slice(i, i+250),
                    east_west=slice(j, j+250)
                ).values
                for i in range(0, 3000, 500)
                for j in range(0, 7200, 500)
                if (i//500 + j//500) % 2 == 0
            ]
        },
        'full_scan': {
            'name': 'Full Dataset Scan',
            'description': 'Complete dataset read',
            'test': lambda ds, var: ds[var].values
        }
    }
    
    return scenarios

def get_additional_chunking_strategies():
    """Define additional chunking strategies to test"""
    
    strategies = [
        # Original strategies
        {
            'name': 'Cloud-Small (500x500)',
            'chunks': {'north_south': 500, 'east_west': 500},
            'rationale': 'Small chunks for random access'
        },
        {
            'name': 'Cloud-Medium (1000x1000)',
            'chunks': {'north_south': 1000, 'east_west': 1000},
            'rationale': 'Balanced cloud performance'
        },
        {
            'name': 'NLDAS-style (750x1800)',
            'chunks': {'north_south': 750, 'east_west': 1800},
            'rationale': 'Operational forecast patterns'
        },
        
        # New strategies
        {
            'name': 'Stripe-Horizontal (100x7200)',
            'chunks': {'north_south': 100, 'east_west': 7200},
            'rationale': 'Optimized for latitude band analysis'
        },
        {
            'name': 'Stripe-Vertical (3000x100)',
            'chunks': {'north_south': 3000, 'east_west': 100},
            'rationale': 'Optimized for longitude band analysis'
        },
        {
            'name': 'Golden-Ratio (1854x1146)',
            'chunks': {'north_south': 1854, 'east_west': 1146},
            'rationale': 'Based on golden ratio for aesthetic/balanced access'
        },
        {
            'name': 'Power-of-2 (512x512)',
            'chunks': {'north_south': 512, 'east_west': 512},
            'rationale': 'Binary-aligned for computing efficiency'
        },
        {
            'name': 'Power-of-2 Large (2048x2048)',
            'chunks': {'north_south': 2048, 'east_west': 2048},
            'rationale': 'Large binary-aligned chunks'
        },
        {
            'name': 'Adaptive-Climate (600x1200)',
            'chunks': {'north_south': 600, 'east_west': 1200},
            'rationale': 'Sized for typical climate model grids'
        },
        {
            'name': 'Memory-Page (375x450)',
            'chunks': {'north_south': 375, 'east_west': 450},
            'rationale': 'Aligned with typical OS memory pages'
        },
        {
            'name': 'Ultra-Small (128x128)',
            'chunks': {'north_south': 128, 'east_west': 128},
            'rationale': 'Maximum granularity for random access'
        },
        {
            'name': 'Asymmetric (300x3600)',
            'chunks': {'north_south': 300, 'east_west': 3600},
            'rationale': 'Optimized for scanning in preferred direction'
        },
        {
            'name': 'Prime-Numbers (503x719)',
            'chunks': {'north_south': 503, 'east_west': 719},
            'rationale': 'Prime numbers to avoid alignment issues'
        },
        {
            'name': 'GPU-Optimized (384x384)',
            'chunks': {'north_south': 384, 'east_west': 384},
            'rationale': 'Multiple of 32 for GPU processing'
        },
        {
            'name': 'Network-Optimized (1460x1460)',
            'chunks': {'north_south': 1460, 'east_west': 1460},
            'rationale': 'Based on typical MTU size considerations'
        }
    ]
    
    return strategies

def benchmark_strategy(ds, strategy: Dict, scenarios: Dict, var_name: str = 'Swnet_tavg') -> Dict:
    """Benchmark a single chunking strategy across all scenarios"""
    
    results = {
        'strategy': strategy['name'],
        'chunk_size_mb': calculate_chunk_size(strategy['chunks']),
        'num_chunks': calculate_num_chunks(strategy['chunks']),
        'scenarios': {}
    }
    
    for scenario_name, scenario in scenarios.items():
        try:
            # Warm-up run
            _ = scenario['test'](ds, var_name)
            
            # Timed runs (3 iterations)
            times = []
            for _ in range(3):
                start = time.time()
                _ = scenario['test'](ds, var_name)
                times.append(time.time() - start)
            
            results['scenarios'][scenario_name] = {
                'time': np.mean(times),
                'std': np.std(times)
            }
            
        except Exception as e:
            results['scenarios'][scenario_name] = {
                'time': float('inf'),
                'std': 0,
                'error': str(e)
            }
    
    return results

def calculate_chunk_size(chunks: Dict) -> float:
    """Calculate chunk size in MB"""
    elements = chunks.get('north_south', 1) * chunks.get('east_west', 1)
    return (elements * 4) / (1024 * 1024)  # 4 bytes per float32

def calculate_num_chunks(chunks: Dict) -> int:
    """Calculate total number of chunks"""
    ns_chunks = np.ceil(3000 / chunks.get('north_south', 3000))
    ew_chunks = np.ceil(7200 / chunks.get('east_west', 7200))
    return int(ns_chunks * ew_chunks)

def create_performance_matrix(results: List[Dict]) -> pd.DataFrame:
    """Create a performance matrix DataFrame"""
    
    # Extract scenario names
    scenario_names = list(results[0]['scenarios'].keys())
    
    # Create matrix
    data = []
    for result in results:
        row = {'Strategy': result['strategy']}
        row['Chunk Size (MB)'] = result['chunk_size_mb']
        row['Num Chunks'] = result['num_chunks']
        
        for scenario in scenario_names:
            if scenario in result['scenarios']:
                row[scenario] = result['scenarios'][scenario]['time']
            else:
                row[scenario] = float('inf')
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def analyze_results(df: pd.DataFrame, scenarios: Dict):
    """Analyze results and generate insights"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*100)
    
    # Best strategy for each scenario
    print("\n1. BEST STRATEGY FOR EACH USE CASE:")
    print("-"*80)
    
    scenario_cols = [col for col in df.columns if col not in ['Strategy', 'Chunk Size (MB)', 'Num Chunks']]
    
    for scenario in scenario_cols:
        best_idx = df[scenario].idxmin()
        best_strategy = df.loc[best_idx, 'Strategy']
        best_time = df.loc[best_idx, scenario]
        
        worst_idx = df[scenario].idxmax()
        worst_time = df.loc[worst_idx, scenario]
        
        speedup = worst_time / best_time if best_time > 0 else float('inf')
        
        print(f"\n{scenarios[scenario]['name']}:")
        print(f"  Best: {best_strategy} ({best_time:.3f}s)")
        print(f"  Speedup vs worst: {speedup:.1f}x")
    
    # Overall rankings
    print("\n\n2. OVERALL PERFORMANCE RANKINGS:")
    print("-"*80)
    
    # Calculate average rank for each strategy
    rankings = {}
    for strategy in df['Strategy']:
        ranks = []
        for scenario in scenario_cols:
            sorted_df = df.sort_values(scenario)
            rank = sorted_df[sorted_df['Strategy'] == strategy].index[0] + 1
            ranks.append(rank)
        rankings[strategy] = np.mean(ranks)
    
    # Sort by average rank
    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1])
    
    print("\nStrategy Rankings (lower is better):")
    for i, (strategy, avg_rank) in enumerate(sorted_rankings[:10], 1):
        chunk_info = df[df['Strategy'] == strategy].iloc[0]
        print(f"{i:2d}. {strategy:<30} (Avg Rank: {avg_rank:.1f}, "
              f"Chunk Size: {chunk_info['Chunk Size (MB)']:.1f}MB, "
              f"Chunks: {chunk_info['Num Chunks']})")
    
    # Category-specific recommendations
    print("\n\n3. CATEGORY-SPECIFIC RECOMMENDATIONS:")
    print("-"*80)
    
    categories = {
        'Random Access': ['point_access', 'random_scatter', 'small_region'],
        'Regional Analysis': ['medium_region', 'large_region'],
        'Linear Scans': ['lat_bands', 'lon_bands', 'diagonal_transect'],
        'Full Processing': ['full_scan'],
        'Mixed Patterns': ['checkerboard']
    }
    
    for category, scenario_list in categories.items():
        print(f"\n{category}:")
        
        # Calculate average time for each strategy in this category
        category_performance = {}
        for _, row in df.iterrows():
            times = [row[s] for s in scenario_list if s in row]
            if times:
                category_performance[row['Strategy']] = np.mean(times)
        
        # Get top 3
        sorted_perf = sorted(category_performance.items(), key=lambda x: x[1])[:3]
        for i, (strategy, avg_time) in enumerate(sorted_perf, 1):
            print(f"  {i}. {strategy} (Avg: {avg_time:.3f}s)")

def generate_visualizations(df: pd.DataFrame, scenarios: Dict):
    """Generate performance visualizations"""
    
    # Create output directory
    os.makedirs('performance_plots', exist_ok=True)
    
    # 1. Heatmap of performance
    plt.figure(figsize=(16, 10))
    
    # Prepare data for heatmap
    scenario_cols = [col for col in df.columns if col not in ['Strategy', 'Chunk Size (MB)', 'Num Chunks']]
    heatmap_data = df[scenario_cols].values
    
    # Normalize by row (best = 1, others relative)
    for i, row in enumerate(heatmap_data):
        min_val = np.min(row)
        if min_val > 0:
            heatmap_data[i] = min_val / row
    
    plt.imshow(heatmap_data, aspect='auto', cmap='RdYlGn')
    plt.colorbar(label='Relative Performance (1.0 = best)')
    
    plt.yticks(range(len(df)), df['Strategy'])
    plt.xticks(range(len(scenario_cols)), [scenarios[s]['name'] for s in scenario_cols], rotation=45, ha='right')
    
    plt.title('Chunking Strategy Performance Heatmap\n(Green = Better, Red = Worse)')
    plt.tight_layout()
    plt.savefig('performance_plots/performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot: Chunk Size vs Performance
    plt.figure(figsize=(12, 8))
    
    # Average performance across all scenarios
    avg_performance = df[scenario_cols].mean(axis=1)
    
    plt.scatter(df['Chunk Size (MB)'], avg_performance, s=100, alpha=0.6)
    
    # Label best performers
    best_indices = avg_performance.nsmallest(5).index
    for idx in best_indices:
        plt.annotate(df.loc[idx, 'Strategy'], 
                    (df.loc[idx, 'Chunk Size (MB)'], avg_performance[idx]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Chunk Size (MB)')
    plt.ylabel('Average Time (seconds)')
    plt.title('Chunk Size vs Average Performance')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('performance_plots/chunk_size_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_comprehensive_analysis(input_file: str):
    """Run the comprehensive performance analysis"""
    
    print("Starting comprehensive chunking performance analysis...")
    
    # Get test scenarios and strategies
    scenarios = create_test_scenarios()
    strategies = get_additional_chunking_strategies()
    
    # Results storage
    all_results = []
    
    # Test each strategy
    for i, strategy in enumerate(strategies, 1):
        print(f"\nTesting strategy {i}/{len(strategies)}: {strategy['name']}...")
        
        try:
            # Create temporary chunked file
            with tempfile.TemporaryDirectory() as temp_dir:
                # Open and chunk the dataset
                ds = xr.open_dataset(input_file)
                chunks = strategy['chunks'].copy()
                chunks['time'] = 1  # Add time dimension
                
                # Apply chunking and save as zarr
                zarr_path = os.path.join(temp_dir, 'data.zarr')
                ds.chunk(chunks).to_zarr(zarr_path, mode='w')
                ds.close()
                
                # Open chunked dataset
                ds_chunked = xr.open_zarr(zarr_path)
                
                # Benchmark
                results = benchmark_strategy(ds_chunked, strategy, scenarios)
                all_results.append(results)
                
                ds_chunked.close()
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Create performance matrix
    df = create_performance_matrix(all_results)
    
    # Save raw results
    df.to_csv('chunking_performance_results.csv', index=False)
    
    # Save detailed results as JSON
    with open('chunking_performance_detailed.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Analyze and visualize
    analyze_results(df, scenarios)
    generate_visualizations(df, scenarios)
    
    print("\n\nAnalysis complete! Results saved to:")
    print("  - chunking_performance_results.csv")
    print("  - chunking_performance_detailed.json")
    print("  - performance_plots/")

if __name__ == "__main__":
    input_file = '/Users/akulkarn/Desktop/chuncking_experiment/LIS_HIST_200901070000.d01.nc'
    run_comprehensive_analysis(input_file)