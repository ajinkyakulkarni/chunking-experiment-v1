import xarray as xr
import numpy as np
import zarr
from rechunker import rechunk
import os
import time
from typing import Dict, List, Tuple

def analyze_dataset(file_path: str) -> Dict:
    """Analyze NetCDF dataset structure"""
    ds = xr.open_dataset(file_path)
    
    # Get actual dimensions
    dims = dict(ds.sizes)
    
    # Since this dataset has spatial dimensions as east_west/north_south
    # and only 1 time step, we'll focus on spatial chunking
    
    analysis = {
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
        'dimensions': dims,
        'spatial_shape': (dims.get('north_south', 0), dims.get('east_west', 0)),
        'time_steps': dims.get('time', 1),
        'variables': len(list(ds.data_vars)),
        'main_vars': [v for v in ds.data_vars if 'north_south' in ds[v].dims and 'east_west' in ds[v].dims]
    }
    
    ds.close()
    return analysis

def generate_chunking_strategies(analysis: Dict) -> List[Dict]:
    """Generate different chunking strategies for spatial data"""
    
    north_south = analysis['dimensions']['north_south']  # 3000
    east_west = analysis['dimensions']['east_west']      # 7200
    
    strategies = []
    
    # Strategy 1: Square-like chunks (balanced)
    # Target ~50MB chunks for float32 data
    target_mb = 50
    bytes_per_element = 4
    elements_per_chunk = (target_mb * 1024 * 1024) / bytes_per_element
    
    # Calculate roughly square chunks
    chunk_side = int(np.sqrt(elements_per_chunk))
    
    strategies.append({
        'name': 'Balanced Square',
        'chunks': {
            'north_south': min(chunk_side, north_south),
            'east_west': min(chunk_side, east_west),
            'time': 1
        },
        'description': 'Square-like chunks for balanced spatial access',
        'use_case': 'General purpose, balanced read patterns'
    })
    
    # Strategy 2: Row-optimized (full width, chunked height)
    rows_per_chunk = int(elements_per_chunk / east_west)
    strategies.append({
        'name': 'Row-optimized',
        'chunks': {
            'north_south': max(1, min(rows_per_chunk, north_south)),
            'east_west': east_west,
            'time': 1
        },
        'description': 'Full width rows for east-west scanning',
        'use_case': 'Latitude-based analysis, horizontal scanning'
    })
    
    # Strategy 3: Column-optimized (full height, chunked width)
    cols_per_chunk = int(elements_per_chunk / north_south)
    strategies.append({
        'name': 'Column-optimized',
        'chunks': {
            'north_south': north_south,
            'east_west': max(1, min(cols_per_chunk, east_west)),
            'time': 1
        },
        'description': 'Full height columns for north-south scanning',
        'use_case': 'Longitude-based analysis, vertical scanning'
    })
    
    # Strategy 4: Small chunks for cloud optimization
    strategies.append({
        'name': 'Cloud-optimized Small',
        'chunks': {
            'north_south': 500,
            'east_west': 500,
            'time': 1
        },
        'description': 'Small chunks for cloud storage and partial reads',
        'use_case': 'Cloud storage, random access patterns'
    })
    
    # Strategy 5: Medium chunks
    strategies.append({
        'name': 'Cloud-optimized Medium',
        'chunks': {
            'north_south': 1000,
            'east_west': 1000,
            'time': 1
        },
        'description': 'Medium chunks balancing size and granularity',
        'use_case': 'Regional analysis, moderate data access'
    })
    
    # Strategy 6: Large chunks
    strategies.append({
        'name': 'Large Chunks',
        'chunks': {
            'north_south': 1500,
            'east_west': 2400,
            'time': 1
        },
        'description': 'Large chunks for sequential processing',
        'use_case': 'Batch processing, full dataset operations'
    })
    
    # Strategy 7: NLDAS-inspired (adapted for this grid)
    strategies.append({
        'name': 'NLDAS-inspired',
        'chunks': {
            'north_south': 750,  # 3000/4
            'east_west': 1800,   # 7200/4
            'time': 1
        },
        'description': 'Inspired by NLDAS chunking patterns',
        'use_case': 'Operational forecasting patterns'
    })
    
    # Strategy 8: Tile-based (for visualization)
    strategies.append({
        'name': 'Tile-based',
        'chunks': {
            'north_south': 256,
            'east_west': 256,
            'time': 1
        },
        'description': 'Web map tile-like chunks',
        'use_case': 'Visualization, web mapping applications'
    })
    
    return strategies

def calculate_chunk_metrics(dims: Dict, chunks: Dict) -> Dict:
    """Calculate performance metrics for a chunking strategy"""
    
    # Calculate chunk size in MB (assuming float32)
    bytes_per_element = 4
    chunk_elements = 1
    
    for dim in ['north_south', 'east_west']:
        if dim in chunks:
            chunk_elements *= chunks[dim]
    
    chunk_size_mb = (chunk_elements * bytes_per_element) / (1024 * 1024)
    
    # Calculate number of chunks
    num_chunks_ns = np.ceil(dims['north_south'] / chunks['north_south'])
    num_chunks_ew = np.ceil(dims['east_west'] / chunks['east_west'])
    total_chunks = int(num_chunks_ns * num_chunks_ew)
    
    # Estimate read patterns efficiency
    # Full scan: all chunks need to be read
    full_scan_chunks = total_chunks
    
    # Regional read (10% of area)
    regional_chunks = max(1, int(total_chunks * 0.1))
    
    # Point query (single value)
    point_query_chunks = 1
    
    # Line scan (full latitude)
    line_scan_ew = int(num_chunks_ew)
    
    # Line scan (full longitude)
    line_scan_ns = int(num_chunks_ns)
    
    return {
        'chunk_size_mb': round(chunk_size_mb, 2),
        'total_chunks': total_chunks,
        'chunks_grid': f"{int(num_chunks_ns)}x{int(num_chunks_ew)}",
        'access_patterns': {
            'full_scan': full_scan_chunks,
            'regional_10pct': regional_chunks,
            'point_query': point_query_chunks,
            'latitude_line': line_scan_ew,
            'longitude_line': line_scan_ns
        }
    }

def create_rechunking_report(file_path: str):
    """Create comprehensive rechunking report"""
    
    print("="*80)
    print("CLOUD-OPTIMIZED NETCDF RECHUNKING ANALYSIS")
    print("="*80)
    
    # Analyze dataset
    analysis = analyze_dataset(file_path)
    
    print(f"\nDataset: {os.path.basename(file_path)}")
    print(f"File size: {analysis['file_size_mb']:.2f} MB")
    print(f"Dimensions: {analysis['dimensions']}")
    print(f"Spatial shape: {analysis['spatial_shape']} (north_south x east_west)")
    print(f"Time steps: {analysis['time_steps']}")
    print(f"Number of variables: {analysis['variables']}")
    
    # Generate strategies
    strategies = generate_chunking_strategies(analysis)
    
    print("\n" + "="*80)
    print("CHUNKING STRATEGIES COMPARISON")
    print("="*80)
    
    # Create comparison table
    print(f"\n{'Strategy':<25} {'Chunks (NS x EW)':<20} {'Chunk Size':<15} {'Total Chunks':<15} {'Use Case':<40}")
    print("-"*120)
    
    for strategy in strategies:
        chunks = strategy['chunks']
        metrics = calculate_chunk_metrics(analysis['dimensions'], chunks)
        
        chunk_str = f"{chunks['north_south']} x {chunks['east_west']}"
        
        print(f"{strategy['name']:<25} {chunk_str:<20} {metrics['chunk_size_mb']:<15.1f} MB "
              f"{metrics['total_chunks']:<15} {strategy['use_case']:<40}")
    
    # Detailed analysis
    print("\n" + "="*80)
    print("DETAILED STRATEGY ANALYSIS")
    print("="*80)
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print(f"   Description: {strategy['description']}")
        print(f"   Chunks: north_south={strategy['chunks']['north_south']}, "
              f"east_west={strategy['chunks']['east_west']}")
        
        metrics = calculate_chunk_metrics(analysis['dimensions'], strategy['chunks'])
        
        print(f"   Chunk size: {metrics['chunk_size_mb']} MB")
        print(f"   Total chunks: {metrics['total_chunks']} ({metrics['chunks_grid']})")
        print(f"   Access patterns efficiency:")
        print(f"     - Full dataset scan: {metrics['access_patterns']['full_scan']} chunks")
        print(f"     - Regional query (10%): {metrics['access_patterns']['regional_10pct']} chunks")
        print(f"     - Point query: {metrics['access_patterns']['point_query']} chunk")
        print(f"     - Latitude line scan: {metrics['access_patterns']['latitude_line']} chunks")
        print(f"     - Longitude line scan: {metrics['access_patterns']['longitude_line']} chunks")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. FOR CLOUD STORAGE (S3, GCS, Azure):")
    print("   - Use 'Cloud-optimized Small' (500x500) or 'Cloud-optimized Medium' (1000x1000)")
    print("   - Smaller chunks enable efficient partial reads and parallel processing")
    print("   - Better for random access patterns and web applications")
    
    print("\n2. FOR LOCAL ANALYSIS:")
    print("   - Use 'Balanced Square' for general-purpose analysis")
    print("   - Use 'Row-optimized' for latitude-based processing")
    print("   - Use 'Column-optimized' for longitude-based processing")
    
    print("\n3. FOR VISUALIZATION:")
    print("   - Use 'Tile-based' (256x256) for web mapping applications")
    print("   - Compatible with standard web map tile systems")
    
    print("\n4. FOR BATCH PROCESSING:")
    print("   - Use 'Large Chunks' (1500x2400) to minimize overhead")
    print("   - Fewer chunks mean less metadata and faster sequential reads")
    
    # Implementation example
    print("\n" + "="*80)
    print("IMPLEMENTATION EXAMPLE")
    print("="*80)
    
    print("\n# Example: Convert to cloud-optimized format with medium chunks")
    print("```python")
    print("import xarray as xr")
    print("import zarr")
    print("")
    print("# Open dataset")
    print(f"ds = xr.open_dataset('{file_path}')")
    print("")
    print("# Define target chunks")
    print("target_chunks = {")
    print("    'north_south': 1000,")
    print("    'east_west': 1000,")
    print("    'time': 1")
    print("}")
    print("")
    print("# Save as Zarr (cloud-optimized)")
    print("ds.chunk(target_chunks).to_zarr('output.zarr', mode='w')")
    print("")
    print("# Or save as NetCDF4 with compression")
    print("encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}")
    print("ds.chunk(target_chunks).to_netcdf('output_chunked.nc', encoding=encoding)")
    print("```")

if __name__ == "__main__":
    file_path = '/Users/akulkarn/Desktop/chuncking_experiment/LIS_HIST_200901070000.d01.nc'
    create_rechunking_report(file_path)