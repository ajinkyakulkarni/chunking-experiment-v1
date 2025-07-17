import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os

def analyze_netcdf(file_path: str) -> Dict:
    """Analyze NetCDF file structure and dimensions"""
    ds = xr.open_dataset(file_path)
    
    analysis = {
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
        'dimensions': {},
        'variables': {},
        'coordinates': list(ds.coords.keys()),
        'data_vars': list(ds.data_vars.keys())
    }
    
    # Analyze dimensions
    for dim in ds.dims:
        analysis['dimensions'][dim] = ds.dims[dim]
    
    # Analyze variables
    for var in list(ds.coords.keys()) + list(ds.data_vars.keys()):
        var_info = {
            'shape': ds[var].shape,
            'dtype': str(ds[var].dtype),
            'size_mb': ds[var].nbytes / (1024 * 1024),
            'dimensions': list(ds[var].dims),
            'current_chunks': ds[var].chunks if ds[var].chunks else None
        }
        analysis['variables'][var] = var_info
    
    ds.close()
    return analysis

def calculate_chunk_options(dims: Dict[str, int], target_chunk_mb: float = 50.0) -> List[Dict]:
    """Calculate different chunking options based on dimensions"""
    chunk_options = []
    
    # Assuming typical float32 data (4 bytes per element)
    bytes_per_element = 4
    
    # Option 1: Balanced chunks (roughly equal in all dimensions)
    if 'time' in dims and 'lat' in dims and 'lon' in dims:
        time_size = dims['time']
        lat_size = dims['lat']
        lon_size = dims['lon']
        
        # Calculate balanced chunk sizes
        total_elements = target_chunk_mb * 1024 * 1024 / bytes_per_element
        chunk_size_per_dim = int(np.cbrt(total_elements))
        
        # Adjust for dimension limits
        time_chunk = min(time_size, max(1, chunk_size_per_dim))
        lat_chunk = min(lat_size, chunk_size_per_dim)
        lon_chunk = min(lon_size, chunk_size_per_dim)
        
        chunk_options.append({
            'name': 'Balanced',
            'chunks': {'time': time_chunk, 'lat': lat_chunk, 'lon': lon_chunk},
            'description': 'Roughly equal chunks in all dimensions'
        })
        
        # Option 2: Time-optimized (full spatial, chunked time)
        time_chunks_for_target = int(total_elements / (lat_size * lon_size))
        time_chunks_for_target = max(1, min(time_size, time_chunks_for_target))
        
        chunk_options.append({
            'name': 'Time-optimized',
            'chunks': {'time': time_chunks_for_target, 'lat': lat_size, 'lon': lon_size},
            'description': 'Full spatial coverage, chunked in time'
        })
        
        # Option 3: Spatial-optimized (full time, chunked spatial)
        spatial_elements = int(np.sqrt(total_elements / time_size))
        spatial_chunk = min(max(lat_size, lon_size), spatial_elements)
        
        chunk_options.append({
            'name': 'Spatial-optimized',
            'chunks': {'time': time_size, 'lat': spatial_chunk, 'lon': spatial_chunk},
            'description': 'Full temporal coverage, chunked spatially'
        })
        
        # Option 4: Single time step chunks (common for time series analysis)
        chunk_options.append({
            'name': 'Single-timestep',
            'chunks': {'time': 1, 'lat': lat_size, 'lon': lon_size},
            'description': 'One time step per chunk'
        })
        
        # Option 5: NLDAS-style chunks (based on the example)
        chunk_options.append({
            'name': 'NLDAS-style',
            'chunks': {'time': 24, 'lat': 224, 'lon': 464},
            'description': 'Based on NLDAS benchmarking example'
        })
    
    return chunk_options

def estimate_chunk_performance(dims: Dict[str, int], chunks: Dict[str, int]) -> Dict:
    """Estimate performance characteristics of a chunking strategy"""
    bytes_per_element = 4
    
    # Calculate chunk size
    chunk_elements = 1
    for dim, chunk_size in chunks.items():
        if dim in dims:
            chunk_elements *= chunk_size
    
    chunk_size_mb = (chunk_elements * bytes_per_element) / (1024 * 1024)
    
    # Calculate number of chunks
    num_chunks = 1
    for dim, size in dims.items():
        if dim in chunks:
            num_chunks *= np.ceil(size / chunks[dim])
    
    return {
        'chunk_size_mb': chunk_size_mb,
        'num_chunks': int(num_chunks),
        'chunks_per_dim': {dim: int(np.ceil(dims[dim] / chunks[dim])) 
                          for dim in chunks if dim in dims}
    }

# Main analysis
if __name__ == "__main__":
    file_path = '/Users/akulkarn/Desktop/chuncking_experiment/LIS_HIST_200901070000.d01.nc'
    
    print("Analyzing NetCDF file...")
    analysis = analyze_netcdf(file_path)
    
    print(f"\nFile: {file_path}")
    print(f"File size: {analysis['file_size_mb']:.2f} MB")
    print(f"\nDimensions:")
    for dim, size in analysis['dimensions'].items():
        print(f"  {dim}: {size}")
    
    print(f"\nCoordinates: {', '.join(analysis['coordinates'])}")
    print(f"Data variables: {', '.join(analysis['data_vars'])}")
    
    print("\nVariable details:")
    for var, info in analysis['variables'].items():
        print(f"\n{var}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Size: {info['size_mb']:.2f} MB")
        print(f"  Dimensions: {info['dimensions']}")
        
    # Generate chunking options
    print("\n" + "="*80)
    print("CHUNKING OPTIONS")
    print("="*80)
    
    chunk_options = calculate_chunk_options(analysis['dimensions'])
    
    for i, option in enumerate(chunk_options, 1):
        print(f"\nOption {i}: {option['name']}")
        print(f"Description: {option['description']}")
        print(f"Chunks: {option['chunks']}")
        
        if all(dim in analysis['dimensions'] for dim in option['chunks']):
            perf = estimate_chunk_performance(analysis['dimensions'], option['chunks'])
            print(f"Estimated chunk size: {perf['chunk_size_mb']:.2f} MB")
            print(f"Number of chunks: {perf['num_chunks']}")
            print(f"Chunks per dimension: {perf['chunks_per_dim']}")