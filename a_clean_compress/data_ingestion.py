"""
Data ingestion module responsible for loading data from various sources.
Handles local files, remote URLs, and torrents, returning standardized pandas DataFrames.
"""

import os
import tempfile
from pathlib import Path
from typing import Union, List, Optional
from urllib.parse import urlparse
import magic
import qbittorrentapi
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataSourceError(Exception):
    """Custom exception for data source related errors."""
    pass


def validate_file_path(file_path: Union[str, Path]) -> Path:
    """
    Validate if a given file path exists and is accessible.
    
    Args:
        file_path: Path to the file as string or Path object
        
    Returns:
        Path object of validated file path
        
    Raises:
        DataSourceError: If path is invalid or file is inaccessible
    """
    path = Path(file_path)
    if not path.exists():
        raise DataSourceError(f"File not found: {file_path}")
    if not path.is_file():
        raise DataSourceError(f"Path is not a file: {file_path}")
    return path


def validate_url(url: str) -> str:
    """
    Validate if a given URL is properly formatted.
    
    Args:
        url: URL string to validate
        
    Returns:
        Validated URL string
        
    Raises:
        DataSourceError: If URL is invalid
    """
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise DataSourceError(f"Invalid URL format: {url}")
        return url
    except Exception as e:
        raise DataSourceError(f"URL validation failed: {str(e)}")


def is_torrent_file(file_path: Union[str, Path]) -> bool:
    """Check if file is a torrent file using magic numbers."""
    mime = magic.Magic(mime=True)
    return mime.from_file(str(file_path)) == 'application/x-bittorrent'


def is_magnet_link(url: str) -> bool:
    """Check if URL is a magnet link."""
    return url.startswith('magnet:?')


def setup_qbittorrent_client() -> qbittorrentapi.Client:
    """Setup and connect to qBittorrent client."""
    try:
        # Get configuration from environment variables or use defaults
        host = os.getenv('QBITTORRENT_HOST', 'localhost')
        port = int(os.getenv('QBITTORRENT_PORT', '8080'))
        username = os.getenv('QBITTORRENT_USERNAME', 'admin')
        password = os.getenv('QBITTORRENT_PASSWORD', 'adminadmin')
        
        # Try to connect to qBittorrent
        client = qbittorrentapi.Client(
            host=host,
            port=port,
            username=username,
            password=password
        )
        
        # Test connection
        try:
            client.auth_log_in()
        except qbittorrentapi.LoginFailed:
            raise DataSourceError(
                "Failed to login to qBittorrent. Please check credentials in .env file.\n"
                "Required environment variables:\n"
                "QBITTORRENT_HOST (default: localhost)\n"
                "QBITTORRENT_PORT (default: 8080)\n"
                "QBITTORRENT_USERNAME (default: admin)\n"
                "QBITTORRENT_PASSWORD (default: adminadmin)"
            )
        except qbittorrentapi.APIConnectionError:
            raise DataSourceError(
                f"Could not connect to qBittorrent at {host}:{port}.\n"
                "Please ensure qBittorrent is running and WebUI is enabled.\n"
                "To enable WebUI in qBittorrent:\n"
                "1. Open qBittorrent\n"
                "2. Go to Tools -> Preferences -> Web UI\n"
                "3. Check 'Web User Interface (Remote Control)'\n"
                "4. Set port (default: 8080)\n"
                "5. Set username and password\n"
                "6. Click Apply and OK"
            )
            
        return client
        
    except Exception as e:
        raise DataSourceError(
            f"Failed to setup qBittorrent client: {str(e)}\n"
            "Please ensure qBittorrent is installed and running."
        )


def download_from_torrent(
    torrent_path_or_magnet: Union[str, Path],
    save_path: Optional[Path] = None
) -> Path:
    """
    Download data from a torrent file or magnet link.
    
    Args:
        torrent_path_or_magnet: Path to torrent file or magnet link
        save_path: Optional path to save downloaded files
        
    Returns:
        Path to downloaded file
        
    Raises:
        DataSourceError: If download fails
    """
    try:
        client = setup_qbittorrent_client()
        
        # Use temporary directory if no save path provided
        if save_path is None:
            save_path = Path(tempfile.mkdtemp())
        
        # Add torrent to client
        if isinstance(torrent_path_or_magnet, str) and is_magnet_link(torrent_path_or_magnet):
            torrent = client.torrents_add(urls=torrent_path_or_magnet, save_path=str(save_path))
        else:
            with open(torrent_path_or_magnet, 'rb') as f:
                torrent = client.torrents_add(torrent_files=f, save_path=str(save_path))
        
        if not torrent:
            raise DataSourceError("Failed to add torrent")
        
        # Wait for download to complete
        while True:
            torrents = client.torrents_info()
            if not torrents:
                raise DataSourceError("Torrent not found in client")
            
            torrent = torrents[0]
            if torrent.state_enum.is_complete:
                break
            if torrent.state_enum.is_errored:
                raise DataSourceError(f"Torrent download failed: {torrent.state}")
        
        # Find the main data file
        content_path = Path(save_path) / torrent.content_path
        if content_path.is_file():
            return content_path
        
        # If it's a directory, look for data files
        if content_path.is_dir():
            data_files = list(content_path.glob('*.csv')) + \
                        list(content_path.glob('*.json')) + \
                        list(content_path.glob('*.xlsx'))
            if data_files:
                return data_files[0]
            
        raise DataSourceError("No suitable data file found in torrent")
        
    except Exception as e:
        raise DataSourceError(f"Failed to download torrent: {str(e)}")


def convert_to_dataframe(file_path: Path) -> pd.DataFrame:
    """Convert various file formats to DataFrame."""
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix == '.json':
            return pd.read_json(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise DataSourceError(f"Unsupported file format: {suffix}")
    except Exception as e:
        raise DataSourceError(f"Failed to convert file to DataFrame: {str(e)}")


def load_from_local(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from a local file into a pandas DataFrame.
    Now supports torrent files in addition to CSV, JSON, and Excel formats.
    """
    path = validate_file_path(file_path)
    
    try:
        # Check if it's a torrent file
        if is_torrent_file(path):
            downloaded_path = download_from_torrent(path)
            return convert_to_dataframe(downloaded_path)
            
        return convert_to_dataframe(path)
        
    except Exception as e:
        raise DataSourceError(f"Failed to load file {file_path}: {str(e)}")


def load_from_url(url: str, file_format: str = 'csv') -> pd.DataFrame:
    """
    Load data from a URL into a pandas DataFrame.
    Now supports magnet links in addition to direct downloads.
    """
    try:
        # Handle magnet links
        if is_magnet_link(url):
            downloaded_path = download_from_torrent(url)
            return convert_to_dataframe(downloaded_path)
            
        # Handle regular URLs
        validated_url = validate_url(url)
        response = requests.get(validated_url)
        response.raise_for_status()
        
        if file_format.lower() == 'csv':
            return pd.read_csv(pd.io.common.StringIO(response.text))
        elif file_format.lower() == 'json':
            return pd.read_json(pd.io.common.StringIO(response.text))
        else:
            raise DataSourceError(f"Unsupported format for URL loading: {file_format}")
            
    except requests.exceptions.RequestException as e:
        raise DataSourceError(f"Failed to download from URL {url}: {str(e)}")
    except Exception as e:
        raise DataSourceError(f"Failed to load data from URL {url}: {str(e)}")


def load_multiple_files(file_paths: List[Union[str, Path]]) -> pd.DataFrame:
    """
    Load and concatenate data from multiple local files.
    
    Args:
        file_paths: List of paths to local files
        
    Returns:
        Concatenated pandas DataFrame
        
    Raises:
        DataSourceError: If any file fails to load
    """
    dataframes = []
    
    for file_path in file_paths:
        df = load_from_local(file_path)
        dataframes.append(df)
    
    if not dataframes:
        raise DataSourceError("No data files were loaded")
    
    return pd.concat(dataframes, ignore_index=True) 