import re
import json
import time
from html import unescape
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, quote
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError

from .base_scraper import BaseScraper, ScrapedItem


class PoshmarkScraper(BaseScraper):
    BASE_URL = "https://poshmark.com"
    SEARCH_URL = "https://poshmark.com/search"
    CONDITION_MAP = {
        'nwt': 'New With Tags',
        'new_with_tags': 'New With Tags',
        'new_without_tags': 'New Without Tags',
        'new_with_defects': 'New With Defects',
        'excellent': 'Excellent',
        'good': 'Good',
        'fair': 'Fair',
        'poor': 'Poor',
    }
    
    def __init__(self, output_dir: str, **kwargs):
        super().__init__(output_dir, **kwargs)
        self.platform = "poshmark"
        self.playwright = None
        self.browser = None
        self.context = None
    
    def init_browser(self):
        if self.browser is None:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.context = self.browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={'width': 1920, 'height': 1080}
            )
            self.logger.info("Playwright browser initialized")
    
    def get_page_content(self, url: str, wait_for_selector: str = None, wait_time: int = 3000) -> Optional[str]:
        self.rate_limit_wait()
        self.stats['requests_made'] += 1
        
        try:
            self.init_browser()
            page = self.context.new_page()
            
            self.logger.debug(f"Navigating to: {url}")
            page.goto(url, wait_until='domcontentloaded', timeout=self.timeout * 1000)
            
            if wait_for_selector:
                try:
                    page.wait_for_selector(wait_for_selector, timeout=wait_time)
                except PlaywrightTimeoutError:
                    self.logger.warning(f"Timeout waiting for selector: {wait_for_selector}")
            else:
                page.wait_for_timeout(wait_time)
            
            content = page.content()
            page.close()
            
            self.logger.debug(f"Successfully fetched content from: {url}")
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to fetch page {url}: {e}")
            if 'page' in locals():
                page.close()
            return None
    
    def scrape_search(
        self,
        query: str,
        max_items: int = 100,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        condition: Optional[str] = None,
        **kwargs
    ) -> List[ScrapedItem]:
        self.logger.info(f"Starting Poshmark search scrape: '{query}' (max: {max_items})")
        items = []
        page = 1
        while len(items) < max_items:
            self.logger.info(f"Scraping page {page}...")
            search_params = self.build_search_params(
                query=query,
                page=page,
                category=category,
                min_price=min_price,
                max_price=max_price,
                condition=condition
            )
            
            from urllib.parse import urlencode
            url = f"{self.SEARCH_URL}?{urlencode(search_params)}"
            
            html_content = self.get_page_content(url, wait_for_selector='[class*="tile"], [class*="card"], [class*="item"]')
            if not html_content:
                self.logger.warning(f"Failed to fetch page {page}")
                break
            
            page_items = self.parse_search_results(html_content)
            
            if not page_items:
                self.logger.info("No more items found")
                break
            
            for item_data in page_items:
                if len(items) >= max_items:
                    break
                
                product_url = item_data.get('url')
                if product_url:
                    item = self.scrape_item(product_url)
                    if item:
                        items.append(item)
                        
                        if len(items) % 10 == 0:
                            self.logger.info(f"Scraped {len(items)} items so far...")
                else:
                    self.logger.warning(f"No URL found for item: {item_data}")
            
            page += 1
        
        self.logger.info(f"Completed scraping {len(items)} items from Poshmark")
        return items
    
    def scrape_item(self, item_url: str) -> Optional[ScrapedItem]:
        self.logger.info(f"Scraping item: {item_url}")
        
        html_content = self.get_page_content(item_url, wait_time=5000)
        if not html_content:
            return None
        
        try:
            item = self.parse_item_page(html_content, item_url)
            if item:
                self.save_item(item)
            return item
            
        except Exception as e:
            self.logger.error(f"Failed to parse item page: {e}")
            self.stats['items_failed'] += 1
            return None
    
    def build_search_params(
        self,
        query: str,
        page: int = 1,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        condition: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {
            'query': query,
            'type': 'listings',
            'src': 'dir',
        }
        
        if category:
            params['department'] = category
        
        if min_price is not None:
            params['price_min'] = int(min_price)
        
        if max_price is not None:
            params['price_max'] = int(max_price)
        
        if condition:
            params['condition'] = condition
        
        if page > 1:
            params['max_id'] = page * 48
        
        return params
    
    def parse_search_results(self, html: str) -> List[Dict[str, Any]]:
        items = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            item_cards = soup.find_all('div', class_=re.compile(r'tile|card|item'))
            for card in item_cards:
                try:
                    item_data = self.extract_search_card_data(card)
                    if item_data:
                        items.append(item_data)
                except Exception as e:
                    self.logger.debug(f"Failed to parse search card: {e}")
                    continue
            
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and data.get('@type') == 'Product':
                        items.append(self.extract_json_ld_data(data))
                except:
                    pass
            
        except Exception as e:
            self.logger.error(f"Failed to parse search results: {e}")
        
        return items
    
    def extract_search_card_data(self, card) -> Optional[Dict[str, Any]]:
        try:
            link = card.find('a', href=re.compile(r'/listing/'))
            if not link:
                return None
            
            item_url = urljoin(self.BASE_URL, link.get('href', ''))
            
            match = re.search(r'/listing/([^/?]+)', item_url)
            item_id = match.group(1) if match else None
            if not item_id:
                return None
            
            title_elem = card.find(['h2', 'h3', 'p'], class_=re.compile(r'title|name'))
            title = title_elem.get_text(strip=True) if title_elem else ''
            
            price_elem = card.find(text=re.compile(r'\$\d+'))
            price_text = price_elem.strip() if price_elem else '$0'
            price = self.parse_price(price_text)
            
            img = card.find('img')
            image_url = img.get('src') or img.get('data-src') if img else None
            
            return {
                'item_id': item_id,
                'url': item_url,
                'title': title,
                'price': price,
                'image_url': image_url,
            }
            
        except Exception as e:
            self.logger.debug(f"Failed to extract card data: {e}")
            return None
    
    def extract_json_ld_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'title': data.get('name', ''),
            'price': float(data.get('offers', {}).get('price', 0)),
            'image_url': data.get('image', [''])[0] if isinstance(data.get('image'), list) else data.get('image', ''),
            'description': data.get('description', ''),
            'brand': data.get('brand', {}).get('name', ''),
            'url': data.get('url', ''),
        }
    
    def parse_item_page(self, html: str, url: str) -> Optional[ScrapedItem]:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            match = re.search(r'/listing/[^/]+-([a-f0-9]+)$', url)
            if not match:
                match = re.search(r'/listing/(.+)$', url)
            
            item_id = match.group(1) if match else None
            
            if not item_id:
                self.logger.warning(f"Could not extract item ID from URL: {url}")
                return None
            
            scripts = soup.find_all('script', type='application/ld+json')
            product_data = {}
            breadcrumb_data = {}
            
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        if data.get('@type') == 'Product':
                            product_data = data
                            self.logger.debug(f"Found Product JSON-LD for {item_id}")
                        elif data.get('@type') == 'BreadcrumbList':
                            breadcrumb_data = data
                            self.logger.debug(f"Found BreadcrumbList JSON-LD for {item_id}")
                except Exception as e:
                    self.logger.debug(f"Failed to parse JSON-LD: {e}")
                    pass
            
            title = product_data.get('name', '')
            
            offers = product_data.get('offers', {})
            price = float(offers.get('price', 0))
            currency = offers.get('priceCurrency', 'USD')
            
            description = product_data.get('description', '')
            
            brand = None
            if isinstance(product_data.get('brand'), dict):
                brand = product_data['brand'].get('name')
            elif isinstance(product_data.get('brand'), str):
                brand = product_data['brand']
            
            category_raw = product_data.get('category', '')
            
            category = unescape(category_raw).replace('<', ' > ') if category_raw else None
            
            style = None
            if category and ' > ' in category:
                parts = category.split(' > ')
                if len(parts) >= 3:
                    style = parts[-1]
                    category = parts[-2]
            
            breadcrumb_path = []
            if breadcrumb_data.get('itemListElement'):
                for item in breadcrumb_data['itemListElement']:
                    item_name = item.get('name', '')
                    if item_name and item_name not in ['Home', brand]:
                        breadcrumb_path.append(item_name)
            
            image_urls = []
            if product_data.get('image'):
                imgs = product_data['image']
                image_urls = [imgs] if isinstance(imgs, str) else imgs
            
            condition_url = offers.get('itemCondition', '')
            condition = None
            if 'NewCondition' in condition_url:
                condition = 'New With Tags'
            elif 'UsedCondition' in condition_url:
                condition = 'Used'
            elif 'RefurbishedCondition' in condition_url:
                condition = 'Refurbished'
            
            if description:
                desc_lower = description.lower()
                if 'new with tags' in desc_lower or 'nwt' in desc_lower:
                    condition = 'New With Tags'
                elif 'new without tags' in desc_lower or 'nwot' in desc_lower:
                    condition = 'New Without Tags'
                elif 'excellent pre owned' in desc_lower or 'excellent condition' in desc_lower:
                    condition = 'Excellent'
                elif 'like new' in desc_lower:
                    condition = 'Like New'
                elif 'good pre owned' in desc_lower or 'good condition' in desc_lower:
                    condition = 'Good'
                elif 'fair condition' in desc_lower:
                    condition = 'Fair'
            
            size = None
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                desc_content = meta_desc.get('content', '')
                size_match = re.search(r'\bSize\s+(\w+)', desc_content, re.IGNORECASE)
                if size_match:
                    size = size_match.group(1)
                    self.logger.debug(f"Extracted size '{size}' from meta description")
            
            if not size and description:
                size_match = re.search(r'\bSize[:\s]+(\w+)', description, re.IGNORECASE)
                if size_match:
                    size = size_match.group(1)
            
            color = product_data.get('color')
            if not color and description:
                color_match = re.search(r'\bColor[:\s]+([^\n,.]+)', description, re.IGNORECASE)
                if color_match:
                    color = color_match.group(1).strip()
            
            item = ScrapedItem(
                platform=self.platform,
                item_id=item_id,
                url=url,
                title=title,
                price=price,
                currency=currency,
                brand=brand,
                condition=condition,
                description=description,
                category=category,
                size=size,
                color=color,
                image_urls=image_urls,
                raw_data={
                    'product_data': product_data,
                    'breadcrumb_data': breadcrumb_data,
                    'style': style
                }
            )
            
            self.stats['items_scraped'] += 1
            self.logger.info(f"Successfully parsed item {item_id}: {title}")
            
            return item
            
        except Exception as e:
            self.logger.error(f"Failed to parse item page: {e}")
            return None
    
    def create_item_from_search(self, item_data: Dict[str, Any]) -> Optional[ScrapedItem]:
        try:
            item_id = item_data.get('item_id')
            if not item_id:
                return None
            
            image_urls = []
            if item_data.get('image_url'):
                image_urls = [item_data['image_url']]
            
            item = ScrapedItem(
                platform=self.platform,
                item_id=item_id,
                url=item_data.get('url', ''),
                title=item_data.get('title', ''),
                price=item_data.get('price', 0.0),
                currency='USD',
                brand=item_data.get('brand'),
                condition=item_data.get('condition'),
                description=item_data.get('description'),
                image_urls=image_urls,
                raw_data=item_data
            )
            
            return item
            
        except Exception as e:
            self.logger.error(f"Failed to create item from search data: {e}")
            return None
    
    def extract_text(self, soup, selectors: List[str]) -> Optional[str]:
        for selector in selectors:
            if selector.startswith('['):
                elem = soup.select_one(selector)
            else:
                elem = soup.find(class_=selector) or soup.find(selector)
            
            if elem:
                text = elem.get_text(strip=True)
                if text:
                    return text
        
        return None
    
    def parse_price(self, price_text: str) -> float:
        try:
            price_str = re.sub(r'[^\d.]', '', price_text)
            return float(price_str) if price_str else 0.0
        except:
            return 0.0
    
    def close(self):
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        super().close()
