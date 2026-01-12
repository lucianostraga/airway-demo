"""
Dataset downloaders for public aviation data.

Provides utilities to download and cache public datasets:
- BTS (Bureau of Transportation Statistics) airline data
- Kaggle flight delay datasets
"""

from .bts_downloader import BTSDownloader

__all__ = ["BTSDownloader"]
