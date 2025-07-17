# NetCDF Cloud-Optimized Chunking Experiment

## Overview

This repository contains a comprehensive analysis and benchmarking suite for optimizing NetCDF file chunking strategies for cloud storage and various access patterns. The project demonstrates how proper chunking can achieve up to **57x performance improvements** for specific data access patterns.

## Problem Statement

When storing large gridded datasets (like climate or weather data) in the cloud, the way data is chunked dramatically affects performance. Poor chunking choices can make data access prohibitively slow and expensive. This project analyzes 15+ different chunking strategies to find optimal configurations for different use cases.

## What We Did

### 1. Dataset Analysis
We analyzed a Land Information System (LIS) NetCDF file with:
- **Size**: 612 MB
- **Dimensions**: 3000 × 7200 spatial grid (north_south × east_west)
- **Variables**: 32 variables including temperature, moisture, radiation, and vegetation parameters
- **Challenge**: Finding optimal chunk sizes for different access patterns

### 2. Chunking Strategies Tested

We tested 15+ chunking strategies including:
- **Traditional approaches**: Square chunks (256×256, 500×500, 1000×1000)
- **NLDAS-inspired**: Based on operational weather forecasting systems (750×1800)
- **Innovative patterns**: 
  - Stripe patterns (100×7200 for latitude bands, 3000×100 for longitude bands)
  - GPU-optimized (384×384 - multiples of 32)
  - Network-optimized (1460×1460 - MTU aligned)
  - Power-of-2 sizes (512×512, 2048×2048)

### 3. Access Patterns Benchmarked

We tested six common access patterns:
1. **Point Queries**: Random individual pixel access (e.g., weather stations)
2. **Small Regions**: 100×100 pixel areas (city-scale analysis)
3. **Medium Regions**: 500×500 pixel areas (state/province scale)
4. **Latitude Bands**: Full-width horizontal strips (climate zones)
5. **Longitude Bands**: Full-height vertical strips (time zones)
6. **Full Dataset Scan**: Complete data read (batch processing)

## Key Findings

### Performance Results

| Access Pattern | Best Strategy | Time | Worst Strategy | Speedup |
|----------------|---------------|------|----------------|---------|
| Point Queries | Tile-256 | 1.10s | Large-2048 | **3.9x** |
| Small Regions | Tile-256 | 0.10s | Large-2048 | **4.0x** |
| Medium Regions | Small-500 | 0.06s | Stripe-H | **6.5x** |
| Latitude Bands | Stripe-H-100 | 0.02s | Stripe-V | **57.1x** |
| Longitude Bands | Stripe-V-100 | 0.01s | Stripe-H | **37.8x** |
| Full Scan | NLDAS-750×1800 | 0.32s | Tile-256 | **11.3x** |

### Major Discoveries

1. **Stripe patterns are game-changers**: For linear access patterns (climate zones, time zones), stripe chunking provides massive speedups (up to 57x).

2. **No one-size-fits-all**: The optimal chunking strategy varies dramatically based on access patterns.

3. **Chunk size matters**: 
   - Small chunks (256×256): Best for random access and web applications
   - Medium chunks (1000×1000): Good all-around performance
   - Large chunks (2048×2048): Best for sequential processing

## Repository Contents

### Analysis Scripts

1. **`rechunking_analysis.py`**: Analyzes NetCDF file structure and suggests chunking strategies
2. **`benchmark_chunks.py`**: Benchmarks different chunking strategies with actual data
3. **`chunking_time_metrics.py`**: Provides detailed time estimates for various access patterns
4. **`quick_performance_comparison.py`**: Rapid comparison of chunking strategies
5. **`detailed_performance_analysis.py`**: Advanced analysis with visualization capabilities
6. **`analyze_netcdf.py`**: Basic NetCDF file analysis tool

### Reports

1. **`CHUNKING_REPORT.md`**: Initial comprehensive analysis report
2. **`FINAL_CHUNKING_REPORT.md`**: Complete findings with time metrics and recommendations

## Usage

### Prerequisites

```bash
pip install xarray netcdf4 h5netcdf zarr rechunker numpy pandas matplotlib tabulate
```

### Basic Analysis

```python
# Analyze your NetCDF file
python rechunking_analysis.py

# Run performance benchmarks
python benchmark_chunks.py

# Get time metrics for different strategies
python chunking_time_metrics.py
```

### Apply Optimal Chunking

```python
import xarray as xr

# Open your dataset
ds = xr.open_dataset('your_file.nc')

# Apply recommended chunking (example: medium chunks for cloud storage)
chunks = {'north_south': 1000, 'east_west': 1000, 'time': 1}

# Save as Zarr (cloud-optimized)
ds.chunk(chunks).to_zarr('output.zarr', mode='w', consolidated=True)

# Or save as NetCDF4 with compression
encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
ds.chunk(chunks).to_netcdf('output_chunked.nc', encoding=encoding)
```

## Recommendations by Use Case

### 1. Web Mapping Applications
- **Use**: Tile-256 (256×256) or GPU-384 (384×384)
- **Why**: Small chunks enable fast pan/zoom operations

### 2. Climate Analysis (Latitude-based)
- **Use**: Stripe-H-100 (100×7200)
- **Why**: 57x faster for analyzing climate zones

### 3. Time Zone Analysis (Longitude-based)
- **Use**: Stripe-V-100 (3000×100)
- **Why**: 38x faster for meridional analysis

### 4. General Cloud Storage
- **Use**: Medium-1000 (1000×1000)
- **Why**: Best balance for unknown access patterns

### 5. Machine Learning
- **Use**: GPU-384 (384×384) or Small-500 (500×500)
- **Why**: GPU memory alignment or efficient random sampling

### 6. Batch Processing
- **Use**: Large-2048 (2048×2048) or NLDAS-750×1800
- **Why**: Minimizes chunk overhead for sequential reads

## Advanced Concepts

### Innovative Strategies Explored

1. **Hybrid Chunking**: Different chunk sizes for different variables
2. **Hierarchical Chunking**: Multiple resolution levels (like map pyramids)
3. **Access-Pattern Learning**: Dynamic rechunking based on usage logs
4. **Lifecycle-Based Rechunking**: Different chunks for hot/warm/cold data

### Cost Considerations

- **Request Costs**: Smaller chunks = more requests = higher cloud costs
- **Egress Charges**: Consider typical download patterns
- **Storage Overhead**: More chunks = more metadata

## Why This Matters

1. **Performance**: Up to 57x faster data access with proper chunking
2. **Cost Savings**: Reduced cloud egress and request charges
3. **User Experience**: Faster response times for applications
4. **Scalability**: Enables efficient processing of massive datasets
5. **Energy Efficiency**: Less compute time = lower carbon footprint

## Next Steps

1. **Test with your data**: Use the provided scripts to analyze your own NetCDF files
2. **Profile actual usage**: Monitor real access patterns for 1-2 weeks
3. **Implement caching**: Add a caching layer for frequently accessed chunks
4. **Consider your use case**: Choose chunking based on your specific access patterns

## Contributing

Feel free to:
- Test with different datasets
- Add new chunking strategies
- Improve benchmarking methods
- Share your results

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Inspired by [NASA-IMPACT NLDAS benchmarking](https://github.com/NASA-IMPACT/veda-odd/blob/nldas_benchmarking/nldas_benchmarking/01_rechunk/rechunk.ipynb)
- Built with xarray, zarr, and the scientific Python ecosystem

---

*This experiment demonstrates the critical importance of data organization for cloud-native geospatial workflows. Proper chunking is not just an optimization—it's essential for making large-scale data analysis feasible and cost-effective.*