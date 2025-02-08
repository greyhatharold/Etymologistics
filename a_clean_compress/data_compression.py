"""
Data compression module responsible for compressing and decompressing data files.
Supports multiple compression algorithms (gzip, bz2, lzma) and provides validation.
"""

import os
from pathlib import Path
from typing import Union, Optional, Literal
import gzip
import bz2
import lzma
import pandas as pd
import json


class CompressionError(Exception):
    """Custom exception for compression-related errors."""
    pass


CompressMethod = Literal['gzip', 'bz2', 'lzma']
COMPRESSION_EXTENSIONS = {
    'gzip': '.gz',
    'bz2': '.bz2',
    'lzma': '.xz'
}


def get_compressor(method: CompressMethod):
    """
    Get the appropriate compressor based on the method.
    
    Args:
        method: Compression method to use
        
    Returns:
        Compression module to use
        
    Raises:
        CompressionError: If compression method is invalid
    """
    compressors = {
        'gzip': gzip,
        'bz2': bz2,
        'lzma': lzma
    }
    
    if method not in compressors:
        raise CompressionError(f"Unsupported compression method: {method}")
    
    return compressors[method]


def validate_path(path: Union[str, Path]) -> Path:
    """
    Validate and create output directory if it doesn't exist.
    
    Args:
        path: Path to validate
        
    Returns:
        Validated Path object
        
    Raises:
        CompressionError: If path is invalid or inaccessible
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        raise CompressionError(f"Invalid path {path}: {str(e)}")


def compress_dataframe(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    method: CompressMethod = 'gzip',
    compression_level: Optional[int] = None
) -> Path:
    """
    Compress a pandas DataFrame to a file.
    
    Args:
        df: DataFrame to compress
        output_path: Path where compressed file will be saved
        method: Compression method to use ('gzip', 'bz2', 'lzma')
        compression_level: Optional compression level (algorithm-specific)
        
    Returns:
        Path to compressed file
        
    Raises:
        CompressionError: If compression fails
        
    Example:
        >>> df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        >>> compressed_path = compress_dataframe(df, 'data.csv.gz', method='gzip')
    """
    try:
        output_path = validate_path(output_path)
        compressor = get_compressor(method)
        
        # Ensure the file has the correct extension
        if not str(output_path).endswith(COMPRESSION_EXTENSIONS[method]):
            output_path = Path(str(output_path) + COMPRESSION_EXTENSIONS[method])
        
        # Convert DataFrame to CSV string
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        # Compress and write to file
        kwargs = {'compresslevel': compression_level} if compression_level is not None else {}
        with compressor.open(output_path, 'wb', **kwargs) as f:
            f.write(csv_data)
        
        return output_path
    
    except Exception as e:
        raise CompressionError(f"Failed to compress DataFrame: {str(e)}")


def decompress_to_dataframe(
    file_path: Union[str, Path],
    method: Optional[CompressMethod] = None
) -> pd.DataFrame:
    """
    Decompress a file to a pandas DataFrame.
    
    Args:
        file_path: Path to compressed file
        method: Optional compression method (if not specified, inferred from extension)
        
    Returns:
        Decompressed DataFrame
        
    Raises:
        CompressionError: If decompression fails
        
    Example:
        >>> df = decompress_to_dataframe('data.csv.gz')
    """
    try:
        file_path = Path(file_path)
        
        # Infer compression method from file extension if not specified
        if method is None:
            for m, ext in COMPRESSION_EXTENSIONS.items():
                if str(file_path).endswith(ext):
                    method = m
                    break
            if method is None:
                raise CompressionError(f"Could not infer compression method for {file_path}")
        
        compressor = get_compressor(method)
        
        # Decompress and read to DataFrame
        with compressor.open(file_path, 'rb') as f:
            return pd.read_csv(f)
    
    except Exception as e:
        raise CompressionError(f"Failed to decompress file {file_path}: {str(e)}")


def compress_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    method: CompressMethod = 'gzip',
    pattern: str = '*.csv',
    compression_level: Optional[int] = None
) -> dict:
    """
    Compress all matching files in a directory.
    
    Args:
        input_dir: Directory containing files to compress
        output_dir: Directory where compressed files will be saved
        method: Compression method to use
        pattern: Glob pattern for matching files
        compression_level: Optional compression level
        
    Returns:
        Dictionary mapping input files to their compressed outputs
        
    Raises:
        CompressionError: If directory compression fails
        
    Example:
        >>> results = compress_directory('raw_data', 'compressed_data', method='bz2')
    """
    try:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        for input_file in input_dir.glob(pattern):
            if input_file.is_file():
                # Read the input file
                df = pd.read_csv(input_file)
                
                # Create output path
                rel_path = input_file.relative_to(input_dir)
                output_path = output_dir / f"{rel_path}{COMPRESSION_EXTENSIONS[method]}"
                
                # Compress the file
                compressed_path = compress_dataframe(
                    df,
                    output_path,
                    method=method,
                    compression_level=compression_level
                )
                
                results[str(input_file)] = str(compressed_path)
        
        return results
    
    except Exception as e:
        raise CompressionError(f"Directory compression failed: {str(e)}")


def verify_compression(
    original_df: pd.DataFrame,
    compressed_path: Union[str, Path],
    method: Optional[CompressMethod] = None
) -> bool:
    """
    Verify that a compressed file can be decompressed to match the original DataFrame.
    
    Args:
        original_df: Original DataFrame to compare against
        compressed_path: Path to compressed file
        method: Optional compression method (if not specified, inferred from extension)
        
    Returns:
        True if verification succeeds, False otherwise
    """
    try:
        decompressed_df = decompress_to_dataframe(compressed_path, method)
        return original_df.equals(decompressed_df)
    except Exception as e:
        return False 