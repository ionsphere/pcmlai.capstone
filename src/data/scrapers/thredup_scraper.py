import re
import json
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from urllib.parse import urljoin, quote
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError

from .base_scraper import BaseScraper, ScrapedItem


class ThredUpScraper(BaseScraper):
    BASE_URL = "https://www.thredup.com"
    SEARCH_URL = "https://www.thredup.com/search"
    CONDITION_MAP = {
        'new_with_tags': 'New With Tags',
        'like_new': 'Like New',
        'gently_used': 'Gently Used',
        'good': 'Good',
        'fair': 'Fair',
    }
    
    def __init__(self, output_dir: str, **kwargs):
        super().__init__(output_dir, **kwargs)
        self.platform = "thredup"
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
    
        self.init_browser()
        page = self.context.new_page()
    
        try:
            self.logger.info(f"GET {url}")
            resp = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
            status = resp.status if resp else None
            title = page.title()
            self.logger.info(f"Status={status} Title={title}")
            page.wait_for_timeout(wait_time)
            html = page.content()
            is_cf = ("Just a moment" in html) or ("cf-" in html.lower()) or ("cloudflare" in html.lower())
            if is_cf or status in (403, 429):
                out = Path(self.output_dir) / "debug_thredup"
                out.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                page.screenshot(path=str(out / f"{ts}.png"), full_page=True)
                (out / f"{ts}.html").write_text(html, encoding="utf-8")
                (out / f"{ts}.meta.txt").write_text(
                    f"url={url}\nfinal_url={page.url}\nstatus={status}\ntitle={title}\n",
                    encoding="utf-8"
                )
                self.logger.warning(f"Saved Cloudflare debug artifacts to {out}")
    
            page.close()
            return html
    
        except Exception as e:
            self.logger.error(f"Failed to fetch page {url}: {e}")
            try: page.close()
            except: pass
            return None
    
    def scrape_search(
        self,
        query: str,
        max_items: int = 100,
        department: Optional[str] = None,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        sizes: Optional[List[str]] = None,
        **kwargs
    ) -> List[ScrapedItem]:
        self.logger.info(f"Starting ThredUp search scrape: '{query}' (max: {max_items})")
        items = []
        page = 1
        while len(items) < max_items:
            self.logger.info(f"Scraping page {page}...")
            search_params = self.build_search_params(
                query=query,
                page=page,
                department=department,
                category=category,
                min_price=min_price,
                max_price=max_price,
                sizes=sizes
            )
            
            from urllib.parse import urlencode
            url = f"{self.SEARCH_URL}/{query}?{urlencode(search_params, doseq=True)}"
            html_content = self.get_page_content(url, wait_for_selector='[class*="product"], [class*="item"]')
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
                
                item = self.create_item_from_search(item_data)
                if item:
                    items.append(item)
                    self.save_item(item)
                    if len(items) % 10 == 0:
                        self.logger.info(f"Scraped {len(items)} items so far...")
            
            page += 1
        
        self.logger.info(f"Completed scraping {len(items)} items from ThredUp")
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
        department: Optional[str] = None,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        sizes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        params = {
            'text': query,
            'page': page,
        }
        
        if department:
            params['department'] = department
        
        if category:
            params['category'] = category
        
        if min_price is not None:
            params['min_price'] = int(min_price)
        
        if max_price is not None:
            params['max_price'] = int(max_price)
        
        if sizes:
            params['sizes[]'] = sizes
        
        return params
    
    def parse_search_results(self, html: str) -> List[Dict[str, Any]]:
        items = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            product_cards = soup.find_all(['div', 'article'], class_=re.compile(r'product|item|card'))
            for card in product_cards:
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
                    products = []
                    if isinstance(data, list):
                        products = [d for d in data if d.get('@type') == 'Product']
                    elif isinstance(data, dict):
                        if data.get('@type') == 'Product':
                            products = [data]
                        elif data.get('@type') == 'ItemList':
                            products = data.get('itemListElement', [])
                    
                    for product in products:
                        item_data = self.extract_json_ld_data(product)
                        if item_data:
                            items.append(item_data)
                            
                except Exception as e:
                    self.logger.debug(f"Failed to parse JSON-LD: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to parse search results: {e}")
        
        return items
    
    def extract_search_card_data(self, card) -> Optional[Dict[str, Any]]:
        try:
            link = card.find('a', href=re.compile(r'/products/'))
            if not link:
                return None
            
            item_url = urljoin(self.BASE_URL, link.get('href', ''))
            match = re.search(r'/products/([^/?]+)', item_url)
            item_id = match.group(1) if match else None
            if not item_id:
                return None
            
            title_elem = card.find(['h2', 'h3', 'p'], class_=re.compile(r'title|name|product-name'))
            title = title_elem.get_text(strip=True) if title_elem else ''
            
            brand_elem = card.find(class_=re.compile(r'brand'))
            brand = brand_elem.get_text(strip=True) if brand_elem else None
            
            price_elem = card.find(class_=re.compile(r'price'))
            if not price_elem:
                price_elem = card.find(text=re.compile(r'\$\d+'))
            price_text = price_elem.get_text(strip=True) if price_elem else '$0'
            price = self.parse_price(price_text)
            
            img = card.find('img')
            image_url = None
            if img:
                image_url = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            
            condition_elem = card.find(class_=re.compile(r'condition'))
            condition = condition_elem.get_text(strip=True) if condition_elem else None
            
            size_elem = card.find(class_=re.compile(r'size'))
            size = size_elem.get_text(strip=True) if size_elem else None
            
            return {
                'item_id': item_id,
                'url': item_url,
                'title': title,
                'brand': brand,
                'price': price,
                'condition': condition,
                'size': size,
                'image_url': image_url,
            }
            
        except Exception as e:
            self.logger.debug(f"Failed to extract card data: {e}")
            return None
    
    def extract_json_ld_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            offers = data.get('offers', {})
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            
            images = data.get('image', [])
            if isinstance(images, str):
                images = [images]
            
            return {
                'title': data.get('name', ''),
                'brand': data.get('brand', {}).get('name') if isinstance(data.get('brand'), dict) else data.get('brand'),
                'price': float(offers.get('price', 0)),
                'image_urls': images,
                'description': data.get('description', ''),
                'condition': data.get('itemCondition', ''),
                'url': data.get('url', ''),
                'category': data.get('category', ''),
            }
        except Exception as e:
            self.logger.debug(f"Failed to extract JSON-LD data: {e}")
            return {}
    
    def parse_item_page(self, html: str, url: str) -> Optional[ScrapedItem]:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            match = re.search(r'/products/([^/?]+)', url)
            item_id = match.group(1) if match else None
            if not item_id:
                return None
            
            scripts = soup.find_all('script', type='application/ld+json')
            structured_data = {}
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and data.get('@type') == 'Product':
                        structured_data = data
                        break
                except:
                    pass
            
            title = structured_data.get('name') or self.extract_text(soup, ['h1', '.product-title', '[data-test="product-name"]'])
            
            offers = structured_data.get('offers', {})
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            price = float(offers.get('price', 0))
            
            if not price:
                price_elem = soup.find(class_=re.compile(r'price'))
                if price_elem:
                    price = self.parse_price(price_elem.get_text(strip=True))
            
            description = structured_data.get('description') or self.extract_text(soup, ['.description', '[data-test="description"]'])
            
            brand = None
            if isinstance(structured_data.get('brand'), dict):
                brand = structured_data['brand'].get('name')
            elif isinstance(structured_data.get('brand'), str):
                brand = structured_data['brand']
            
            if not brand:
                brand = self.extract_text(soup, ['.brand', '[data-test="brand"]'])
            
            image_urls = []
            if structured_data.get('image'):
                imgs = structured_data['image']
                image_urls = imgs if isinstance(imgs, list) else [imgs]
            
            if not image_urls:
                img_tags = soup.find_all('img', class_=re.compile(r'product-image'))
                image_urls = [img.get('src') or img.get('data-src') for img in img_tags if img.get('src') or img.get('data-src')]
            
            condition = structured_data.get('itemCondition') or self.extract_text(soup, ['.condition', '[data-test="condition"]'])
            if condition:
                condition_lower = condition.lower().replace(' ', '_')
                condition = self.CONDITION_MAP.get(condition_lower, condition)
            
            category = structured_data.get('category') or self.extract_text(soup, ['.category', '[data-test="category"]'])
            size = self.extract_text(soup, ['.size', '[data-test="size"]'])
            color = self.extract_text(soup, ['.color', '[data-test="color"]'])
            
            item = ScrapedItem(
                platform=self.platform,
                item_id=item_id,
                url=url,
                title=title,
                price=price,
                currency='USD',
                brand=brand,
                condition=condition,
                description=description,
                category=category,
                size=size,
                color=color,
                image_urls=image_urls,
                raw_data=structured_data or {}
            )
            
            return item
            
        except Exception as e:
            self.logger.error(f"Failed to parse item page: {e}")
            return None
    
    def create_item_from_search(self, item_data: Dict[str, Any]) -> Optional[ScrapedItem]:
        try:
            item_id = item_data.get('item_id')
            if not item_id:
                url = item_data.get('url', '')
                match = re.search(r'/products/([^/?]+)', url)
                item_id = match.group(1) if match else None
            
            if not item_id:
                return None
            
            image_urls = []
            if item_data.get('image_url'):
                image_urls = [item_data['image_url']]
            elif item_data.get('image_urls'):
                image_urls = item_data['image_urls']
            
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
                category=item_data.get('category'),
                size=item_data.get('size'),
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
            elif selector.startswith('.'):
                elem = soup.find(class_=selector[1:])
            else:
                elem = soup.find(selector)
            
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
