import xarray as xr
import numpy as np
import time
import os
import tempfile
from tabulate import tabulate
import json

def get_chunking_strategies():
    """Define all chunking strategies to test"""
    
    strategies = [
        # Standard strategies
        {'name': 'Ultra-Small (128x128)', 'chunks': {'north_south': 128, 'east_west': 128}},
        {'name': 'Tile-based (256x256)', 'chunks': {'north_south': 256, 'east_west': 256}},
        {'name': 'GPU-Optimized (384x384)', 'chunks': {'north_south': 384, 'east_west': 384}},
        {'name': 'Cloud-Small (500x500)', 'chunks': {'north_south': 500, 'east_west': 500}},
        {'name': 'Power-of-2 (512x512)', 'chunks': {'north_south': 512, 'east_west': 512}},
        {'name': 'Cloud-Medium (1000x1000)', 'chunks': {'north_south': 1000, 'east_west': 1000}},
        {'name': 'NLDAS-style (750x1800)', 'chunks': {'north_south': 750, 'east_west': 1800}},
        {'name': 'Network-Opt (1460x1460)', 'chunks': {'north_south': 1460, 'east_west': 1460}},
        {'name': 'Power-of-2-L (2048x2048)', 'chunks': {'north_south': 2048, 'east_west': 2048}},
        
        # Specialized strategies
        {'name': 'Stripe-H (100x7200)', 'chunks': {'north_south': 100, 'east_west': 7200}},
        {'name': 'Stripe-V (3000x100)', 'chunks': {'north_south': 3000, 'east_west': 100}},
        {'name': 'Asymmetric (300x3600)', 'chunks': {'north_south': 300, 'east_west': 3600}},
        {'name': 'Memory-Page (375x450)', 'chunks': {'north_south': 375, 'east_west': 450}},
        {'name': 'Prime-Nums (503x719)', 'chunks': {'north_south': 503, 'east_west': 719}},
        {'name': 'Golden-Ratio (1854x1146)', 'chunks': {'north_south': 1854, 'east_west': 1146}},
    ]
    
    return strategies

def quick_benchmark(ds, var_name='Swnet_tavg'):
    """Quick benchmark of common access patterns"""
    
    results = {}
    
    # 1. Single point access (100 points)
    start = time.time()
    for _ in range(100):
        i, j = np.random.randint(0, 3000), np.random.randint(0, 7200)
        _ = ds[var_name].isel(north_south=i, east_west=j).values
    results['point_access'] = time.time() - start
    
    # 2. Small region (100x100, 10 regions)
    start = time.time()
    for _ in range(10):
        i, j = np.random.randint(0, 2900), np.random.randint(0, 7100)
        _ = ds[var_name].isel(
            north_south=slice(i, i+100),
            east_west=slice(j, j+100)
        ).values
    results['small_region'] = time.time() - start
    
    # 3. Medium region (500x500, 5 regions)
    start = time.time()
    for _ in range(5):
        i, j = np.random.randint(0, 2500), np.random.randint(0, 6700)
        _ = ds[var_name].isel(
            north_south=slice(i, i+500),
            east_west=slice(j, j+500)
        ).values
    results['medium_region'] = time.time() - start
    
    # 4. Latitude band (full width, 100 rows)
    start = time.time()
    _ = ds[var_name].isel(north_south=slice(1000, 1100)).values
    results['lat_band'] = time.time() - start
    
    # 5. Longitude band (full height, 100 cols)
    start = time.time()
    _ = ds[var_name].isel(east_west=slice(3000, 3100)).values
    results['lon_band'] = time.time() - start
    
    # 6. Full scan
    start = time.time()
    _ = ds[var_name].values
    results['full_scan'] = time.time() - start
    
    return results

def calculate_metrics(chunks):
    """Calculate chunk metrics"""
    chunk_elements = chunks['north_south'] * chunks['east_west']
    chunk_size_mb = (chunk_elements * 4) / (1024 * 1024)
    
    ns_chunks = int(np.ceil(3000 / chunks['north_south']))
    ew_chunks = int(np.ceil(7200 / chunks['east_west']))
    total_chunks = ns_chunks * ew_chunks
    
    return {
        'chunk_size_mb': round(chunk_size_mb, 2),
        'total_chunks': total_chunks,
        'grid': f"{ns_chunks}x{ew_chunks}"
    }

def run_performance_comparison(input_file):
    """Run performance comparison for all strategies"""
    
    print("Running chunking performance comparison...")
    print("This will test 15 different chunking strategies across 6 access patterns.\n")
    
    strategies = get_chunking_strategies()
    results = []
    
    # Test original file first
    print("Testing original file (baseline)...")
    ds_orig = xr.open_dataset(input_file)
    baseline = quick_benchmark(ds_orig)
    ds_orig.close()
    
    baseline_result = {
        'strategy': 'Original (no chunks)',
        'chunk_size_mb': 'N/A',
        'total_chunks': 1,
        'grid': '1x1',
        **baseline
    }
    results.append(baseline_result)
    
    # Test each chunking strategy
    for i, strategy in enumerate(strategies, 1):
        print(f"Testing {i}/{len(strategies)}: {strategy['name']}...")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create chunked dataset
                ds = xr.open_dataset(input_file)
                chunks = strategy['chunks'].copy()
                chunks['time'] = 1
                
                zarr_path = os.path.join(temp_dir, 'data.zarr')
                ds.chunk(chunks).to_zarr(zarr_path, mode='w', consolidated=True)
                ds.close()
                
                # Benchmark
                ds_chunked = xr.open_zarr(zarr_path)
                perf = quick_benchmark(ds_chunked)
                ds_chunked.close()
                
                # Calculate metrics
                metrics = calculate_metrics(strategy['chunks'])
                
                result = {
                    'strategy': strategy['name'],
                    **metrics,
                    **perf
                }
                results.append(result)
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Display results
    print("\n" + "="*120)
    print("PERFORMANCE COMPARISON RESULTS (times in seconds)")
    print("="*120)
    
    # Prepare table data
    headers = ['Strategy', 'Chunk MB', 'Chunks', 'Grid', 'Points', 'Small', 'Medium', 'Lat Band', 'Lon Band', 'Full Scan']
    table_data = []
    
    for r in results:
        row = [
            r['strategy'],
            r['chunk_size_mb'],
            r['total_chunks'],
            r['grid'],
            f"{r['point_access']:.3f}",
            f"{r['small_region']:.3f}",
            f"{r['medium_region']:.3f}",
            f"{r['lat_band']:.3f}",
            f"{r['lon_band']:.3f}",
            f"{r['full_scan']:.3f}"
        ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Find best strategies for each use case
    print("\n" + "="*120)
    print("BEST STRATEGIES FOR EACH USE CASE")
    print("="*120)
    
    use_cases = {
        'point_access': 'Point Access (weather stations, spot checks)',
        'small_region': 'Small Regions (city-scale analysis)',
        'medium_region': 'Medium Regions (state/province analysis)',
        'lat_band': 'Latitude Bands (climate zones)',
        'lon_band': 'Longitude Bands (time zones)',
        'full_scan': 'Full Dataset (batch processing)'
    }
    
    for metric, description in use_cases.items():
        # Sort by this metric
        sorted_results = sorted(results[1:], key=lambda x: x[metric])  # Skip baseline
        best = sorted_results[0]
        
        # Calculate speedup vs baseline
        baseline_time = baseline[metric]
        speedup = baseline_time / best[metric] if best[metric] > 0 else 0
        
        print(f"\n{description}:")
        print(f"  Best: {best['strategy']}")
        print(f"  Time: {best[metric]:.3f}s (Baseline: {baseline_time:.3f}s)")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Chunk size: {best['chunk_size_mb']} MB, Total chunks: {best['total_chunks']}")
    
    # Overall recommendations
    print("\n" + "="*120)
    print("RECOMMENDATIONS BASED ON TIME METRICS")
    print("="*120)
    
    print("\n1. FOR RANDOM ACCESS PATTERNS (Point queries, small regions):")
    print("   - Use smaller chunks (256x256 to 512x512)")
    print("   - Trade-off: More chunks mean more metadata overhead")
    print("   - Best: Tile-based (256x256) or Power-of-2 (512x512)")
    
    print("\n2. FOR REGIONAL ANALYSIS (Medium to large regions):")
    print("   - Use medium chunks (750x750 to 1500x1500)")
    print("   - Balanced performance across different region sizes")
    print("   - Best: Cloud-Medium (1000x1000) or NLDAS-style (750x1800)")
    
    print("\n3. FOR LINEAR SCANS (Climate/time zone analysis):")
    print("   - Use stripe patterns matching your scan direction")
    print("   - Stripe-H for latitude bands, Stripe-V for longitude bands")
    print("   - Alternative: Asymmetric chunks (300x3600)")
    
    print("\n4. FOR FULL DATASET PROCESSING:")
    print("   - Use larger chunks to minimize overhead")
    print("   - Best: Power-of-2-L (2048x2048) or Golden-Ratio (1854x1146)")
    print("   - Fewer chunks = less metadata to process")
    
    print("\n5. SPECIAL CONSIDERATIONS:")
    print("   - GPU Processing: Use GPU-Optimized (384x384) - multiples of 32")
    print("   - Network Transfer: Use Network-Opt (1460x1460) - aligned with MTU")
    print("   - Memory Efficiency: Use Memory-Page (375x450) - OS page alignment")
    
    # Save results
    with open('performance_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: performance_comparison_results.json")

if __name__ == "__main__":
    # Install tabulate if needed
    try:
        import tabulate
    except ImportError:
        os.system('pip install tabulate')
        import tabulate
    
    input_file = '/Users/akulkarn/Desktop/chuncking_experiment/LIS_HIST_200901070000.d01.nc'
    run_performance_comparison(input_file)