"""
Marketplace scraping modules for clothing price prediction.

This package contains scrapers for various second-hand clothing marketplaces.
"""

from .base_scraper import BaseScraper, ScrapedItem
from .poshmark_scraper import PoshmarkScraper
from .thredup_scraper import ThredUpScraper
from .depop_scraper import DepopScraper

__all__ = [
    'BaseScraper',
    'ScrapedItem',
    'PoshmarkScraper',
    'ThredUpScraper',
    'DepopScraper',
]
