import re
import json
import time
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, quote
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError

from .base_scraper import BaseScraper, ScrapedItem


class DepopScraper(BaseScraper):
    BASE_URL = "https://www.depop.com"
    SEARCH_URL = "https://www.depop.com/search"
    API_BASE = "https://webapi.depop.com/api/v2"
    CONDITION_MAP = {
        'brand_new': 'Brand New',
        'like_new': 'Like New',
        'used_excellent': 'Used - Excellent',
        'used_good': 'Used - Good',
        'used_fair': 'Used - Fair',
    }
    
    def __init__(self, output_dir: str, **kwargs):
        super().__init__(output_dir, **kwargs)
        self.platform = "depop"
        self.playwright = None
        self.browser = None
        self.context = None
    
    def init_browser(self):
        self.run_browser_task(self._init_browser_impl)

    def _init_browser_impl(self):
        if self.browser is None:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled']
            )
            self.context = self.browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={'width': 1920, 'height': 1080},
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            self.logger.info("Playwright browser initialized with anti-bot detection avoidance")
    
    def get_page_content(self, url: str, wait_for_selector: str = None, wait_time: int = 3000) -> Optional[str]:
        self.rate_limit_wait()
        self.stats['requests_made'] += 1

        try:
            return self.run_browser_task(self._get_page_content_impl, url, wait_for_selector, wait_time)
        except Exception as e:
            self.logger.error(f"Failed to fetch page {url}: {e}")
            return None

    def _get_page_content_impl(self, url: str, wait_for_selector: str = None, wait_time: int = 3000) -> Optional[str]:
        self._init_browser_impl()
        page = self.context.new_page()
        self.logger.debug(f"Navigating to: {url}")
        response = page.goto(url, wait_until='domcontentloaded', timeout=60000)
        if response and response.status == 403:
            self.logger.error(f"Access forbidden (403) for URL: {url}")
            page.close()
            return None
        
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
    
    def scrape_search(
        self,
        query: str,
        max_items: int = 100,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        **kwargs
    ) -> List[ScrapedItem]:
        self.logger.info(f"Starting Depop search scrape: '{query}' (max: {max_items})")
        items = []
        offset = 0
        limit = 20
        while len(items) < max_items:
            self.logger.info(f"Scraping offset {offset}...")
            api_items = self.scrape_via_api(
                query=query,
                offset=offset,
                limit=limit,
                category=category,
                min_price=min_price,
                max_price=max_price
            )
            
            if api_items:
                for item_data in api_items:
                    if len(items) >= max_items:
                        break
                    
                    item = self.create_item_from_api(item_data)
                    if item:
                        items.append(item)
                        self.save_item(item)
                        if len(items) % 10 == 0:
                            self.logger.info(f"Scraped {len(items)} items so far...")
                
                offset += limit
            else:
                self.logger.warning("API scraping failed, falling back to web scraping")
                web_items = self.scrape_via_web(query, max_items - len(items))
                items.extend(web_items)
                break
        
        self.logger.info(f"Completed scraping {len(items)} items from Depop")
        return items
    
    def scrape_item(self, item_url: str) -> Optional[ScrapedItem]:
        self.logger.info(f"Scraping item: {item_url}")
        playwright_data = self.get_item_fields_with_playwright(item_url)
        html_content = self.get_page_content(item_url, wait_time=5000)
        if not html_content:
            return None
        
        try:
            item = self.parse_item_page(html_content, item_url, playwright_data)
            if item:
                self.save_item(item)
            return item
            
        except Exception as e:
            self.logger.error(f"Failed to parse item page: {e}")
            self.stats['items_failed'] += 1
            return None
    
    def scrape_via_api(
        self,
        query: str,
        offset: int = 0,
        limit: int = 20,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        try:
            return self.run_browser_task(
                self._scrape_via_api_impl,
                query,
                offset,
                limit,
                category,
                min_price,
                max_price,
            )
        except Exception as e:
            self.logger.error(f"API scraping failed: {e}")
            return []

    def _scrape_via_api_impl(
        self,
        query: str,
        offset: int = 0,
        limit: int = 20,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        from urllib.parse import urlencode

        api_url = f"{self.API_BASE}/search/products/"
        params = {
            'q': query,
            'offset': offset,
            'limit': limit,
        }
        
        if category:
            params['categoryId'] = category
        
        if min_price is not None:
            params['priceMin'] = int(min_price * 100)
        
        if max_price is not None:
            params['priceMax'] = int(max_price * 100)
        
        url = f"{api_url}?{urlencode(params)}"
        self._init_browser_impl()
        page = self.context.new_page()
        
        try:
            response = page.goto(url, wait_until='domcontentloaded', timeout=60000)
            
            if response and response.ok:
                content = page.content()
                page.close()
                soup = BeautifulSoup(content, 'html.parser')
                json_text = soup.get_text()
                try:
                    data = json.loads(json_text)
                    return data.get('products', [])
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse API JSON response")
                    return []

            page.close()
            return []
        except Exception:
            page.close()
            raise
    
    def scrape_via_web(self, query: str, max_items: int) -> List[ScrapedItem]:
        try:
            return self.run_browser_task(self._scrape_via_web_impl, query, max_items)
        except Exception as e:
            self.logger.error(f"Web scraping failed: {e}")
            return []

    def _scrape_via_web_impl(self, query: str, max_items: int) -> List[ScrapedItem]:
        items = []
        search_url = f"{self.SEARCH_URL}/?q={quote(query)}"
        self._init_browser_impl()
        page = self.context.new_page()
        try:
            page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
            self.logger.info("Waiting for products to load (10 seconds)...")
            page.wait_for_timeout(10000)
            page.evaluate("window.scrollTo(0, 800)")
            page.wait_for_timeout(2000)
            product_links = page.locator('a[href*="/products/"]').all()
            self.logger.info(f"Found {len(product_links)} product links")
            for i in range(min(max_items, len(product_links))):
                try:
                    href = product_links[i].get_attribute('href')
                    if href:
                        if href.startswith('/'):
                            product_url = self.BASE_URL + href
                        else:
                            product_url = href
                        
                        item = self.scrape_item(product_url)
                        if item:
                            items.append(item)
                            if len(items) % 5 == 0:
                                self.logger.info(f"Scraped {len(items)} items from web...")
                except Exception as e:
                    self.logger.debug(f"Failed to scrape product {i}: {e}")
                    continue
        finally:
            page.close()

        return items
    
    def parse_search_results(self, html: str) -> List[Dict[str, Any]]:
        items = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            product_cards = soup.find_all(['div', 'article', 'li'], class_=re.compile(r'product|item|card'))
            for card in product_cards:
                try:
                    item_data = self.extract_search_card_data(card)
                    if item_data:
                        items.append(item_data)
                except Exception as e:
                    self.logger.debug(f"Failed to parse search card: {e}")
                    continue
            
            scripts = soup.find_all('script', type='application/json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        products = self.extract_products_from_json(data)
                        items.extend(products)
                        
                except Exception as e:
                    self.logger.debug(f"Failed to parse JSON data: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to parse search results: {e}")
        
        return items
    
    def extract_products_from_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        products = []
        try:
            if 'props' in data:
                page_props = data.get('props', {}).get('pageProps', {})
                for key in ['products', 'items', 'searchResults']:
                    if key in page_props:
                        items = page_props[key]
                        if isinstance(items, list):
                            products.extend(items)
            
            if 'products' in data:
                products.extend(data['products'])
                
        except Exception as e:
            self.logger.debug(f"Failed to extract products from JSON: {e}")
        
        return products
    
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
            
            title_elem = card.find(['h2', 'h3', 'p'], class_=re.compile(r'title|name'))
            title = title_elem.get_text(strip=True) if title_elem else ''
            
            price_elem = card.find(class_=re.compile(r'price'))
            if not price_elem:
                price_elem = card.find(text=re.compile(r'[\$]\d+'))
            price_text = price_elem.get_text(strip=True) if price_elem else '$0'
            price, currency = self.parse_price_with_currency(price_text)
            
            img = card.find('img')
            image_url = None
            if img:
                image_url = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            
            seller_elem = card.find(class_=re.compile(r'seller|username'))
            seller_id = seller_elem.get_text(strip=True) if seller_elem else None
            
            return {
                'item_id': item_id,
                'url': item_url,
                'title': title,
                'price': price,
                'currency': currency,
                'image_url': image_url,
                'seller_id': seller_id,
            }
            
        except Exception as e:
            self.logger.debug(f"Failed to extract card data: {e}")
            return None
    
    def create_item_from_api(self, api_data: Dict[str, Any]) -> Optional[ScrapedItem]:
        try:
            item_id = api_data.get('id') or api_data.get('slug')
            if not item_id:
                return None
            
            item_id = str(item_id)
            
            price_data = api_data.get('price', {})
            if isinstance(price_data, dict):
                price_cents = price_data.get('priceAmount', 0)
                price = price_cents / 100.0
                currency = price_data.get('currencyName', 'USD')
            else:
                price = float(api_data.get('price', 0))
                currency = 'USD'
            
            image_urls = []
            pictures = api_data.get('pictures', [])
            for pic in pictures:
                if isinstance(pic, dict):
                    url = pic.get('url') or pic.get('base')
                    if url:
                        image_urls.append(url)
                elif isinstance(pic, str):
                    image_urls.append(pic)
            
            slug = api_data.get('slug', item_id)
            item_url = f"{self.BASE_URL}/products/{slug}/"
            
            condition = api_data.get('condition')
            if condition:
                condition = self.CONDITION_MAP.get(condition, condition.replace('_', ' ').title())
            
            item = ScrapedItem(
                platform=self.platform,
                item_id=item_id,
                url=item_url,
                title=api_data.get('description', ''),
                price=price,
                currency=currency,
                brand=api_data.get('brand'),
                condition=condition,
                description=api_data.get('description'),
                category=api_data.get('categoryId'),
                size=api_data.get('size'),
                color=api_data.get('colour'),
                image_urls=image_urls,
                seller_id=str(api_data.get('owner', {}).get('id', '')),
                num_likes=api_data.get('likeCount', 0),
                raw_data=api_data
            )
            
            return item
            
        except Exception as e:
            self.logger.error(f"Failed to create item from API data: {e}")
            return None
    
    def parse_item_page(self, html: str, url: str, playwright_data: Optional[dict] = None) -> Optional[ScrapedItem]:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            match = re.search(r'/products/([^/?]+)', url)
            item_id = match.group(1) if match else None
            if not item_id:
                return None
            
            scripts = soup.find_all('script', type='application/ld+json')
            product_data = {}
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and data.get('@type') == 'Product':
                        product_data = data
                        self.logger.debug(f"Found JSON-LD Product data for {item_id}")
                        break
                except Exception as e:
                    self.logger.debug(f"Failed to parse JSON-LD script: {e}")
            
            if product_data:
                title = product_data.get('name', '')
                
                offers = product_data.get('offers', {})
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                
                price = float(offers.get('price', 0))
                currency = offers.get('priceCurrency', 'USD')
                
                description = product_data.get('description', '')
                
                brand = None
                if isinstance(product_data.get('brand'), dict):
                    brand = product_data['brand'].get('name')
                elif isinstance(product_data.get('brand'), str):
                    brand = product_data['brand']
                
                if brand and brand.lower() == 'other':
                    brand = None
                
                image_urls = []
                if product_data.get('image'):
                    imgs = product_data['image']
                    image_urls = imgs if isinstance(imgs, list) else [imgs]
                
                condition = None
                if playwright_data and playwright_data.get('condition'):
                    condition = playwright_data['condition']
                else:
                    condition_url = offers.get('itemCondition', '')
                    if 'NewCondition' in condition_url:
                        condition = 'Brand New'
                    elif 'UsedCondition' in condition_url:
                        condition = 'Used'
                    elif 'RefurbishedCondition' in condition_url:
                        condition = 'Refurbished'
                    
                    if description:
                        desc_lower = description.lower()
                        if 'brand new' in desc_lower or 'new with tags' in desc_lower or 'nwt' in desc_lower:
                            condition = 'Brand New'
                        elif 'like new' in desc_lower or 'excellent condition' in desc_lower:
                            condition = 'Like New'
                
                size = None
                if playwright_data and playwright_data.get('size'):
                    size = playwright_data['size']
                else:
                    if description:
                        size_match = re.search(r'(?:Size[:\s]+|^|\s)([XS|S|M|L|XL|XXL|\d+](?:-\d+)?)\b', description, re.IGNORECASE | re.MULTILINE)
                        if size_match:
                            size = size_match.group(1).upper()
                
                color = None
                color_keywords = [
                    'black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple',
                    'orange', 'brown', 'grey', 'gray', 'beige', 'cream', 'tan', 'navy',
                    'maroon', 'burgundy', 'teal', 'turquoise', 'gold', 'silver', 'bronze'
                ]
                
                if title:
                    title_lower = title.lower()
                    for color_word in color_keywords:
                        if color_word in title_lower:
                            color = color_word.title()
                            break
                
                if not color and description:
                    desc_lower = description.lower()
                    for color_word in color_keywords:
                        if color_word in desc_lower:
                            color = color_word.title()
                            break
                
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
                    category=None,
                    size=size,
                    color=color,
                    image_urls=image_urls,
                    raw_data={'product_data': product_data}
                )
                
                return item
            
            self.logger.debug(f"No JSON-LD found, trying Next.js data for {item_id}")
            next_data = soup.find('script', id='__NEXT_DATA__')
            if next_data:
                try:
                    data = json.loads(next_data.string)
                    api_product = data.get('props', {}).get('pageProps', {}).get('product', {})
                    if api_product:
                        return self.create_item_from_api(api_product)
                except Exception as e:
                    self.logger.debug(f"Failed to parse __NEXT_DATA__: {e}")
            
            self.logger.warning(f"Using fallback HTML parsing for {item_id}")
            title = self.extract_text(soup, ['h1', '[data-testid="product-title"]'])
            price = 0.0
            currency = 'USD'
            
            price_elem = soup.find(class_=re.compile(r'price'))
            if price_elem:
                price, currency = self.parse_price_with_currency(price_elem.get_text(strip=True))
            
            description = self.extract_text(soup, ['[data-testid="product-description"]', '.description'])
            brand = self.extract_text(soup, ['[data-testid="brand"]', '.brand'])
            size = self.extract_text(soup, ['[data-testid="size"]', '.size'])
            color = self.extract_text(soup, ['[data-testid="color"]', '.color'])
            condition = self.extract_text(soup, ['[data-testid="condition"]', '.condition'])
            
            item = ScrapedItem(
                platform=self.platform,
                item_id=item_id,
                url=url,
                title=title or '',
                price=price,
                currency=currency,
                brand=brand,
                condition=condition,
                description=description,
                category=None,
                size=size,
                color=color,
                image_urls=[],
                raw_data={}
            )
            
            return item
            
        except Exception as e:
            self.logger.error(f"Failed to parse item page: {e}")
            return None
    
    def create_item_from_search(self, item_data: Dict[str, Any]) -> Optional[ScrapedItem]:
        try:
            item_id = item_data.get('item_id') or item_data.get('id')
            if not item_id:
                return None
            
            image_urls = []
            if item_data.get('image_url'):
                image_urls = [item_data['image_url']]
            elif item_data.get('image_urls'):
                image_urls = item_data['image_urls']
            
            item = ScrapedItem(
                platform=self.platform,
                item_id=str(item_id),
                url=item_data.get('url', ''),
                title=item_data.get('title', ''),
                price=item_data.get('price', 0.0),
                currency=item_data.get('currency', 'USD'),
                brand=item_data.get('brand'),
                condition=item_data.get('condition'),
                description=item_data.get('description'),
                size=item_data.get('size'),
                image_urls=image_urls,
                seller_id=item_data.get('seller_id'),
                raw_data=item_data
            )
            
            return item
            
        except Exception as e:
            self.logger.error(f"Failed to create item from search data: {e}")
            return None

    def get_item_fields_with_playwright(self, url: str) -> Optional[dict]:
        return self.run_browser_task(self._get_item_fields_with_playwright_impl, url)

    def _get_item_fields_with_playwright_impl(self, url: str) -> Optional[dict]:
        self._init_browser_impl()
        page = self.context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)

            title = page.locator("h1").first.inner_text(timeout=5000).strip()

            body_text = page.locator("body").inner_text(timeout=5000)
            m_price = re.search(r"(\$||)\s?(\d+(?:\.\d{2})?)", body_text)
            price = float(m_price.group(2)) if m_price else 0.0
            currency = "USD" if m_price and m_price.group(1) == "$" else ("GBP" if m_price and m_price.group(1) == "" else ("EUR" if m_price and m_price.group(1) == "" else "USD"))

            m_size = re.search(r"\bSize\s+(\w+)", body_text)
            size = m_size.group(1).strip() if m_size else None

            m_cond = re.search(r"\b(Brand new|Like new|Good condition|Fair condition|Poor condition|Good|Fair|Poor)\b", body_text, re.IGNORECASE)
            condition = m_cond.group(1).strip().title() if m_cond else None

            return {
                "title": title,
                "price": price,
                "currency": currency,
                "size": size,
                "condition": condition,
            }
        finally:
            page.close()
    
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
    
    def parse_price_with_currency(self, price_text: str) -> tuple:
        try:
            currency = 'USD'
            if '' in price_text:
                currency = 'GBP'
            elif '' in price_text:
                currency = 'EUR'
            elif '$' in price_text:
                currency = 'USD'
            
            price_str = re.sub(r'[^\d.]', '', price_text)
            price = float(price_str) if price_str else 0.0
            
            return price, currency
            
        except:
            return 0.0, 'USD'
    
    def close(self):
        if self._browser_thread is not None:
            self.run_browser_task(self._close_browser_impl)
            self.shutdown_browser_thread()
        super().close()

    def _close_browser_impl(self):
        if self.context:
            self.context.close()
            self.context = None
        if self.browser:
            self.browser.close()
            self.browser = None
        if self.playwright:
            self.playwright.stop()
            self.playwright = None
