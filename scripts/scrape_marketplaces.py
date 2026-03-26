#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.scrapers import (
    PoshmarkScraper,
    ThredUpScraper,
    DepopScraper,
    ScrapedItem
)


def scrape_marketplace(
    platform: str,
    query: str,
    max_items: int,
    output_dir: str,
    rate_limit: float = 2.0,
    **filters
) -> List[ScrapedItem]:
    print(f"Starting scrape: {platform.upper()}")
    print(f"Query: {query}")
    print(f"Max items: {max_items}")
    print(f"Rate limit: {rate_limit}s between requests")
    
    if platform == 'poshmark':
        scraper = PoshmarkScraper(output_dir, rate_limit=rate_limit)
    elif platform == 'thredup':
        scraper = ThredUpScraper(output_dir, rate_limit=rate_limit)
    elif platform == 'depop':
        scraper = DepopScraper(output_dir, rate_limit=rate_limit)
    else:
        raise ValueError(f"Unknown platform: {platform}")
    
    try:
        items = scraper.scrape_search(query, max_items=max_items, **filters)
        if items:
            batch_name = f"{platform}_{query.replace(' ', '_')}"
            scraper.save_batch(items, batch_name=batch_name)
        
        return items
        
    finally:
        scraper.close()


def scrape_all_platforms(
    query: str,
    max_items_per_platform: int,
    output_dir: str,
    platforms: Optional[List[str]] = None,
    rate_limit: float = 2.0,
    **filters
):
    if platforms is None:
        platforms = ['poshmark', 'thredup', 'depop']
    
    all_items = []
    
    for platform in platforms:
        try:
            items = scrape_marketplace(
                platform=platform,
                query=query,
                max_items=max_items_per_platform,
                output_dir=output_dir,
                rate_limit=rate_limit,
                **filters
            )
            all_items.extend(items)
            
        except Exception as e:
            print(f"\n Failed to scrape {platform}: {e}\n")
    
    print("SCRAPING COMPLETE")
    print(f"Total items scraped: {len(all_items)}")
    for platform in platforms:
        platform_items = [i for i in all_items if i.platform == platform]
        print(f"  {platform}: {len(platform_items)} items")
    
    print(f"\nData saved to: {output_dir}")


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description='Scrape fashion marketplace data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Scrape 100 items from Poshmark
  python scrape_marketplaces.py --platform poshmark --query "vintage jeans" --max-items 100

  # Scrape 50 items from all platforms
  python scrape_marketplaces.py --query "leather jacket" --max-items 50 --all-platforms

  # Scrape with price filters
  python scrape_marketplaces.py --platform thredup --query "dress" --max-items 200 --min-price 20 --max-price 100

  # Quick test (10 items from each platform)
  python scrape_marketplaces.py --query "t-shirt" --max-items 10 --all-platforms --rate-limit 1.0
        '''
    )
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Search query (e.g., "vintage jeans", "leather jacket")'
    )
    platform_group = parser.add_mutually_exclusive_group()
    platform_group.add_argument(
        '--platform',
        type=str,
        choices=['poshmark', 'thredup', 'depop'],
        help='Platform to scrape'
    )
    platform_group.add_argument(
        '--all-platforms',
        action='store_true',
        help='Scrape all platforms'
    )
    parser.add_argument(
        '--max-items',
        type=int,
        default=100,
        help='Maximum number of items to scrape per platform (default: 100)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/scraped',
        help='Output directory (default: data/scraped)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=2.0,
        help='Seconds between requests (default: 2.0)'
    )
    parser.add_argument(
        '--min-price',
        type=float,
        help='Minimum price filter'
    )
    parser.add_argument(
        '--max-price',
        type=float,
        help='Maximum price filter'
    )
    parser.add_argument(
        '--category',
        type=str,
        help='Category filter (platform-specific)'
    )
    parser.add_argument(
        '--condition',
        type=str,
        help='Condition filter (e.g., "new", "like_new", "good")'
    )
    
    args = parser.parse_args(argv)
    
    if not args.platform and not args.all_platforms:
        parser.error('Must specify either --platform or --all-platforms')
    
    filters = {}
    if args.min_price is not None:
        filters['min_price'] = args.min_price
    if args.max_price is not None:
        filters['max_price'] = args.max_price
    if args.category:
        filters['category'] = args.category
    if args.condition:
        filters['condition'] = args.condition
    
    if args.all_platforms:
        scrape_all_platforms(
            query=args.query,
            max_items_per_platform=args.max_items,
            output_dir=args.output_dir,
            rate_limit=args.rate_limit,
            **filters
        )
    else:
        scrape_marketplace(
            platform=args.platform,
            query=args.query,
            max_items=args.max_items,
            output_dir=args.output_dir,
            rate_limit=args.rate_limit,
            **filters
        )


if __name__ == '__main__':
    main()
