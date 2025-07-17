import xarray as xr
import numpy as np
import time
import os
import zarr
from typing import Dict, List
import tempfile
import shutil

def benchmark_read_patterns(ds, name: str) -> Dict:
    """Benchmark different read patterns on a dataset"""
    
    results = {
        'name': name,
        'full_read': 0,
        'point_queries': 0,
        'regional_read': 0,
        'line_scan_lat': 0,
        'line_scan_lon': 0
    }
    
    # Get a data variable for testing
    data_var = 'Swnet_tavg'  # One of the main variables
    
    # 1. Full dataset read
    start = time.time()
    data = ds[data_var].values
    results['full_read'] = time.time() - start
    
    # 2. Point queries (100 random points)
    start = time.time()
    for _ in range(100):
        i = np.random.randint(0, 3000)
        j = np.random.randint(0, 7200)
        val = ds[data_var].isel(north_south=i, east_west=j).values
    results['point_queries'] = time.time() - start
    
    # 3. Regional read (10% of area - 1000x1000 region)
    start = time.time()
    i_start = 1000
    j_start = 3000
    region = ds[data_var].isel(
        north_south=slice(i_start, i_start+1000),
        east_west=slice(j_start, j_start+1000)
    ).values
    results['regional_read'] = time.time() - start
    
    # 4. Line scan - full latitude
    start = time.time()
    lat_line = ds[data_var].isel(north_south=1500).values
    results['line_scan_lat'] = time.time() - start
    
    # 5. Line scan - full longitude
    start = time.time()
    lon_line = ds[data_var].isel(east_west=3600).values
    results['line_scan_lon'] = time.time() - start
    
    return results

def create_chunked_dataset(input_file: str, chunks: Dict, output_path: str, 
                          format: str = 'zarr') -> str:
    """Create a chunked version of the dataset"""
    
    ds = xr.open_dataset(input_file)
    
    # Apply chunking
    ds_chunked = ds.chunk(chunks)
    
    if format == 'zarr':
        # Save as Zarr
        zarr_path = os.path.join(output_path, 'data.zarr')
        ds_chunked.to_zarr(zarr_path, mode='w')
        ds.close()
        return zarr_path
    else:
        # Save as NetCDF with compression
        nc_path = os.path.join(output_path, 'data.nc')
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
        ds_chunked.to_netcdf(nc_path, encoding=encoding)
        ds.close()
        return nc_path

def run_benchmarks(input_file: str):
    """Run benchmarks for different chunking strategies"""
    
    print("="*80)
    print("CHUNKING PERFORMANCE BENCHMARKS")
    print("="*80)
    
    # Define chunking strategies to test
    strategies = [
        {
            'name': 'Original (no chunks)',
            'chunks': None,
            'format': 'netcdf'
        },
        {
            'name': 'Cloud-Small (500x500)',
            'chunks': {'north_south': 500, 'east_west': 500, 'time': 1},
            'format': 'zarr'
        },
        {
            'name': 'Cloud-Medium (1000x1000)',
            'chunks': {'north_south': 1000, 'east_west': 1000, 'time': 1},
            'format': 'zarr'
        },
        {
            'name': 'NLDAS-style (750x1800)',
            'chunks': {'north_south': 750, 'east_west': 1800, 'time': 1},
            'format': 'zarr'
        },
        {
            'name': 'Row-optimized (1820x7200)',
            'chunks': {'north_south': 1820, 'east_west': 7200, 'time': 1},
            'format': 'zarr'
        },
        {
            'name': 'Tile-based (256x256)',
            'chunks': {'north_south': 256, 'east_west': 256, 'time': 1},
            'format': 'zarr'
        }
    ]
    
    results = []
    
    for strategy in strategies:
        print(f"\nTesting: {strategy['name']}...")
        
        try:
            if strategy['chunks'] is None:
                # Test original file
                ds = xr.open_dataset(input_file)
                bench_results = benchmark_read_patterns(ds, strategy['name'])
                ds.close()
            else:
                # Create temporary directory for chunked data
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Create chunked dataset
                    output_path = create_chunked_dataset(
                        input_file, 
                        strategy['chunks'], 
                        temp_dir,
                        strategy['format']
                    )
                    
                    # Open and benchmark
                    if strategy['format'] == 'zarr':
                        ds = xr.open_zarr(output_path)
                    else:
                        ds = xr.open_dataset(output_path)
                    
                    bench_results = benchmark_read_patterns(ds, strategy['name'])
                    ds.close()
            
            results.append(bench_results)
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Display results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS (times in seconds)")
    print("="*80)
    
    print(f"\n{'Strategy':<25} {'Full Read':<12} {'100 Points':<12} {'Regional':<12} {'Lat Line':<12} {'Lon Line':<12}")
    print("-"*85)
    
    for result in results:
        print(f"{result['name']:<25} "
              f"{result['full_read']:<12.3f} "
              f"{result['point_queries']:<12.3f} "
              f"{result['regional_read']:<12.3f} "
              f"{result['line_scan_lat']:<12.3f} "
              f"{result['line_scan_lon']:<12.3f}")
    
    # Analysis
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Find best performers for each pattern
    patterns = ['full_read', 'point_queries', 'regional_read', 'line_scan_lat', 'line_scan_lon']
    
    for pattern in patterns:
        best = min(results, key=lambda x: x[pattern])
        worst = max(results, key=lambda x: x[pattern])
        
        pattern_name = pattern.replace('_', ' ').title()
        print(f"\n{pattern_name}:")
        print(f"  Best: {best['name']} ({best[pattern]:.3f}s)")
        print(f"  Worst: {worst['name']} ({worst[pattern]:.3f}s)")
        
        if worst[pattern] > 0:
            speedup = worst[pattern] / best[pattern]
            print(f"  Speedup: {speedup:.1f}x")

def generate_recommendations():
    """Generate final recommendations based on analysis"""
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. GENERAL CLOUD OPTIMIZATION:")
    print("   - For most use cases, 1000x1000 chunks provide the best balance")
    print("   - This size (~4MB per chunk) works well with cloud storage block sizes")
    print("   - Provides good performance for both random and sequential access")
    
    print("\n2. SPECIFIC USE CASE RECOMMENDATIONS:")
    print("   a) Web Applications / Visualization:")
    print("      - Use 256x256 or 512x512 chunks")
    print("      - Aligns with web map tile standards")
    print("      - Enables efficient zoom/pan operations")
    
    print("   b) Time Series Analysis (when you have multiple time steps):")
    print("      - Chunk size: time=24-168, lat=all, lon=all")
    print("      - Optimizes for temporal access patterns")
    
    print("   c) Spatial Analysis:")
    print("      - Regional studies: 1000x1000 chunks")
    print("      - Continental scale: 2000x2000 chunks")
    print("      - Global analysis: consider row or column optimization")
    
    print("\n3. COMPRESSION SETTINGS:")
    print("   - Use zlib compression level 4 for good balance")
    print("   - Consider lossy compression for visualization (scale/offset encoding)")
    print("   - Shuffle filter can improve compression ratios")
    
    print("\n4. FORMAT CHOICE:")
    print("   - Zarr: Best for cloud storage, parallel access")
    print("   - NetCDF4: Better compatibility, single file convenience")
    print("   - Consider COG (Cloud Optimized GeoTIFF) for raster visualization")

if __name__ == "__main__":
    input_file = '/Users/akulkarn/Desktop/chuncking_experiment/LIS_HIST_200901070000.d01.nc'
    
    # Run the analysis
    run_benchmarks(input_file)
    generate_recommendations()