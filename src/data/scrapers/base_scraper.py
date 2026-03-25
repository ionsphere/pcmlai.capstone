import time
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class ScrapedItem:
    platform: str
    item_id: str
    url: str
    title: str
    price: float
    currency: str
    
    brand: Optional[str] = None
    condition: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    size: Optional[str] = None
    color: Optional[str] = None
    
    image_urls: List[str] = None
    
    seller_id: Optional[str] = None
    seller_rating: Optional[float] = None
    num_likes: Optional[int] = None
    num_comments: Optional[int] = None
    listing_date: Optional[str] = None
    
    scraped_at: str = None
    raw_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.image_urls is None:
            self.image_urls = []
        if self.scraped_at is None:
            self.scraped_at = datetime.now().isoformat()
        if self.raw_data is None:
            self.raw_data = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_hash(self) -> str:
        unique_str = f"{self.platform}:{self.item_id}:{self.title}:{self.price}"
        return hashlib.md5(unique_str.encode()).hexdigest()


class BaseScraper(ABC):
    def __init__(
        self,
        output_dir: str,
        rate_limit: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
        user_agent: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_request_time = 0
        
        self.logger = logging.getLogger(self._class__.__name__)
        
        self.session = self.create_session(user_agent)
        
        self.stats = {
            'items_scraped': 0,
            'items_failed': 0,
            'requests_made': 0,
            'start_time': datetime.now().isoformat(),
        }
    
    def create_session(self, user_agent: Optional[str] = None) -> requests.Session:
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        default_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        session.headers.update({
            "User-Agent": user_agent or default_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
        })
        return session
    
    def rate_limit_wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            sleep_time = self.rate_limit - elapsed
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def make_request(
        self,
        url: str,
        method: str = 'GET',
        **kwargs
    ) -> Optional[requests.Response]:
        self.rate_limit_wait()
        self.stats['requests_made'] += 1
        
        try:
            self.logger.debug(f"Making {method} request to: {url}")
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None
    
    def save_item(self, item: ScrapedItem, save_json: bool = True):
        try:
            if save_json:
                platform_dir = self.output_dir / item.platform
                platform_dir.mkdir(exist_ok=True)
                
                filename = f"{item.item_id}.json"
                filepath = platform_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(item.to_dict(), f, indent=2, ensure_ascii=False)
                
                self.logger.debug(f"Saved item to: {filepath}")
            
            self.stats['items_scraped'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to save item {item.item_id}: {e}")
            self.stats['items_failed'] += 1
    
    def save_batch(self, items: List[ScrapedItem], batch_name: str = None):
        if not items:
            self.logger.warning("No items to save in batch")
            return
        
        try:
            platform = items[0].platform
            platform_dir = self.output_dir / platform / "batches"
            platform_dir.mkdir(parents=True, exist_ok=True)
            
            if batch_name is None:
                batch_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"batch_{batch_name}.json"
            filepath = platform_dir / filename
            
            batch_data = {
                'batch_name': batch_name,
                'num_items': len(items),
                'scraped_at': datetime.now().isoformat(),
                'items': [item.to_dict() for item in items]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved batch of {len(items)} items to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save batch: {e}")
    
    def save_stats(self):
        self.stats['end_time'] = datetime.now().isoformat()
        
        stats_file = self.output_dir / "scraping_stats.json"
        
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                all_stats = json.load(f)
        else:
            all_stats = []
        
        all_stats.append({
            'scraper': self._class__.__name__,
            **self.stats
        })
        
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        self.logger.info(f"Statistics saved to: {stats_file}")
    
    @abstractmethod
    def scrape_search(
        self,
        query: str,
        max_items: int = 100,
        **kwargs
    ) -> List[ScrapedItem]:
        pass
    
    @abstractmethod
    def scrape_item(self, item_url: str) -> Optional[ScrapedItem]:
        pass
    
    def close(self):
        self.session.close()
        self.save_stats()
