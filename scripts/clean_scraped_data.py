import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


def load_scraped_data(data_dir: Path) -> pd.DataFrame:
    print(f"Loading data from: {data_dir}")
    items = []
    platforms = [p for p in data_dir.iterdir() if p.is_dir() and p.name in ['poshmark', 'depop', 'thredup']]
    if platforms:
        for platform_dir in platforms:
            platform = platform_dir.name
            print(f"Loading {platform}...")
            json_files = list(platform_dir.glob('*.json'))
            for json_file in tqdm(json_files, desc=f"  {platform} items"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        items.append(data)
                except Exception as e:
                    print(f"Failed to load {json_file}: {e}")
            
            batch_dir = platform_dir / 'batches'
            if batch_dir.exists():
                batch_files = list(batch_dir.glob('*.json'))
                for batch_file in tqdm(batch_files, desc=f"  {platform} batches"):
                    try:
                        with open(batch_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'items' in data:
                                items.extend(data['items'])
                    except Exception as e:
                        print(f"Failed to load {batch_file}: {e}")
    else:
        print("Loading from flat directory...")
        json_files = list(data_dir.glob('*.json'))
        for json_file in tqdm(json_files, desc="  Loading items"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    items.append(data)
            except Exception as e:
                print(f"Failed to load {json_file}: {e}")
    
    if not items:
        print("No data found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(items)
    print(f"Loaded {len(df)} total items")
    return df


def compute_item_hash(item: Dict[str, Any]) -> str:
    key_str = f"{item.get('platform', '')}:{item.get('title', '')}:{item.get('price', 0)}"
    return hashlib.md5(key_str.encode()).hexdigest()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("Removing duplicates...")
    initial_count = len(df)
    df['item_hash'] = df.apply(lambda row: compute_item_hash(row.to_dict()), axis=1)
    df_dedup = df.drop_duplicates(subset=['item_hash'], keep='first')
    if 'url' in df_dedup.columns:
        df_dedup = df_dedup.drop_duplicates(subset=['url'], keep='first')
    
    duplicates_removed = initial_count - len(df_dedup)
    print(f"Removed {duplicates_removed} duplicates ({duplicates_removed/initial_count*100:.1f}%)")
    print(f"Remaining items: {len(df_dedup)}")
    return df_dedup


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning prices...")
    if 'price' not in df.columns:
        print("No price column found")
        return df
    
    initial_count = len(df)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df[df['price'] > 0].copy()
    max_reasonable_price = 10000
    outliers = df[df['price'] > max_reasonable_price]
    if len(outliers) > 0:
        print(f"Found {len(outliers)} items with prices > ${max_reasonable_price}")
        df = df[df['price'] <= max_reasonable_price].copy()
    
    if 'currency' not in df.columns:
        df['currency'] = 'USD'
    else:
        df['currency'] = df['currency'].fillna('USD')
    
    removed = initial_count - len(df)
    print(f"Removed {removed} items with invalid prices")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"Median price: ${df['price'].median():.2f}")
    return df


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning text fields...")
    text_fields = ['title', 'description', 'brand', 'category', 'condition']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].astype(str).str.strip()
            df[field] = df[field].replace(['nan', 'None', ''], None)
            non_null = df[field].notna().sum()
            print(f"{field}: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")
    
    return df


def normalize_platform_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Normalizing platform-specific data to unified schema...")
    normalized_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Normalizing records"):
        platform = row.get('platform', '').lower()
        normalized = {
            'title': None,
            'category': None,
            'style': None,
            'brand': None,
            'price': None,
            'size': None,
            'condition': None,
            'image': None,  # Primary image URL
            'url': None,
            'platform': platform,
            'item_id': row.get('item_id'),
            'scraped_at': row.get('scraped_at')
        }
        
        if platform == 'poshmark':
            normalized.update(normalize_poshmark(row))
        elif platform == 'depop':
            normalized.update(normalize_depop(row))
        elif platform == 'thredup':
            normalized.update(normalize_thredup(row))
        else:
            normalized.update({
                'title': row.get('title'),
                'category': row.get('category'),
                'style': row.get('style'),
                'brand': row.get('brand'),
                'price': row.get('price'),
                'size': row.get('size'),
                'condition': row.get('condition'),
                'image': extract_primary_image(row.get('image_urls')),
                'url': row.get('url')
            })
        
        normalized_data.append(normalized)
    
    normalized_df = pd.DataFrame(normalized_data)
    print(f"Normalized {len(normalized_df)} records to unified schema")
    return normalized_df


def normalize_poshmark(row: pd.Series) -> Dict[str, Any]:
    category_raw = row.get('category', '')
    category, style = extract_category_and_style(category_raw)
    raw_data = row.get('raw_data', {})
    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except:
            raw_data = {}
    
    if not style and raw_data:
        style = raw_data.get('style')
    
    return {
        'title': row.get('title'),
        'category': category,
        'style': style,
        'brand': row.get('brand'),
        'price': row.get('price'),
        'size': row.get('size'),
        'condition': row.get('condition'),
        'image': extract_primary_image(row.get('image_urls')),
        'url': row.get('url')
    }


def normalize_depop(row: pd.Series) -> Dict[str, Any]:
    category = row.get('category')
    if not category:
        category = infer_category_from_text(row.get('title', '') + ' ' + row.get('description', ''))
    
    brand = row.get('brand')
    raw_data = row.get('raw_data', {})
    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except:
            raw_data = {}
    
    if not brand or brand == 'Other':
        product_data = raw_data.get('product_data', {})
        brand_data = product_data.get('brand', {})
        if isinstance(brand_data, dict):
            brand_name = brand_data.get('name')
            if brand_name and brand_name != 'Other':
                brand = brand_name
        elif isinstance(brand_data, str):
            brand = brand_data
    
    style = extract_style_from_description(row.get('description', ''))
    
    return {
        'title': row.get('title'),
        'category': category,
        'style': style,
        'brand': brand if brand != 'Other' else None,
        'price': row.get('price'),
        'size': row.get('size'),
        'condition': row.get('condition'),
        'image': extract_primary_image(row.get('image_urls')),
        'url': row.get('url')
    }


def normalize_thredup(row: pd.Series) -> Dict[str, Any]:
    category_raw = row.get('category', '')
    category, style = extract_category_and_style(category_raw)
    return {
        'title': row.get('title'),
        'category': category,
        'style': style,
        'brand': row.get('brand'),
        'price': row.get('price'),
        'size': row.get('size'),
        'condition': row.get('condition'),
        'image': extract_primary_image(row.get('image_urls')),
        'url': row.get('url')
    }


def extract_category_and_style(category_path: str) -> tuple:
    if not category_path or pd.isna(category_path):
        return None, None
    
    parts = str(category_path).split('<')
    if len(parts) >= 2:
        category = parts[1].strip() if len(parts) > 1 else parts[0].strip()
        style = parts[2].strip() if len(parts) > 2 else None
    else:
        category = parts[0].strip()
        style = None
    
    return category, style


def extract_primary_image(image_urls) -> Optional[str]:
    if not image_urls:
        return None
    
    if isinstance(image_urls, list) and len(image_urls) > 0:
        return image_urls[0]
    elif isinstance(image_urls, str):
        try:
            urls = json.loads(image_urls)
            if isinstance(urls, list) and len(urls) > 0:
                return urls[0]
        except:
            return image_urls
    
    return None


def infer_category_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    
    text_lower = text.lower()
    
    category_keywords = {
        'Skirts': ['skirt', 'maxi skirt', 'mini skirt', 'midi skirt'],
        'Dresses': ['dress', 'gown'],
        'Tops': ['top', 'shirt', 'blouse', 't-shirt', 'tee'],
        'Jeans': ['jeans', 'denim'],
        'Pants': ['pants', 'trousers', 'slacks'],
        'Shorts': ['shorts'],
        'Jackets': ['jacket', 'blazer', 'coat'],
        'Sweaters': ['sweater', 'cardigan', 'pullover'],
        'Shoes': ['shoes', 'boots', 'sneakers', 'heels'],
        'Bags': ['bag', 'purse', 'handbag', 'tote'],
    }
    
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return category
    
    return None


def extract_style_from_description(description: str) -> Optional[str]:
    if not description:
        return None
    
    import re
    hashtags = re.findall(r'#(\w+)', description)
    
    if hashtags:
        style_tags = [tag for tag in hashtags if tag.lower() not in 
                      ['new', 'sale', 'fashion', 'style', 'clothing', 'clothes']]
        if style_tags:
            return ', '.join(style_tags[:3])
    
    return None


def standardize_conditions(df: pd.DataFrame) -> pd.DataFrame:
    print("Standardizing conditions...")
    
    if 'condition' not in df.columns:
        print("No condition column found")
        return df
    
    condition_map = {
        'new with tags': 'New With Tags',
        'nwt': 'New With Tags',
        'new_with_tags': 'New With Tags',
        'brand new': 'New With Tags',
        'new without tags': 'New Without Tags',
        'new_without_tags': 'New Without Tags',
        'new': 'New Without Tags',
        'newcondition': 'New Without Tags',
        
        'like new': 'Like New',
        'like_new': 'Like New',
        'excellent': 'Like New',
        'excellent condition': 'Like New',
        
        'gently used': 'Gently Used',
        'gently_used': 'Gently Used',
        'very good': 'Gently Used',
        'good': 'Good',
        'good condition': 'Good',
        'usedcondition': 'Good',
        'used': 'Good',
        'fair': 'Fair',
        'poor': 'Poor',
        'worn': 'Fair',
    }
    
    df['condition_standardized'] = df['condition'].astype(str).str.lower().str.strip().map(condition_map)
    df['condition_standardized'] = df['condition_standardized'].fillna(df['condition'])
    df['condition_original'] = df['condition']
    df['condition'] = df['condition_standardized']
    df = df.drop('condition_standardized', axis=1)
    print(f"Standardized conditions:")
    condition_counts = df['condition'].value_counts()
    for cond, count in condition_counts.items():
        print(f"{cond}: {count}")
    
    return df


def validate_image_urls(df: pd.DataFrame) -> pd.DataFrame:
    print("Validating image URLs...")
    if 'image' not in df.columns:
        print("No image column found")
        return df
    
    def is_valid_url(url):
        if pd.isna(url) or not isinstance(url, str):
            return False
        return url.startswith('http')
    
    valid_images = df['image'].apply(is_valid_url).sum()
    print(f"Items with valid images: {valid_images}/{len(df)} ({valid_images/len(df)*100:.1f}%)")
    
    return df


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    print("Adding derived fields...")
    
    if 'price' in df.columns:
        bins = [0, 20, 50, 100, 200, float('inf')]
        labels = ['<$20', '$20-50', '$50-100', '$100-200', '>$200']
        df['price_bracket'] = pd.cut(df['price'], bins=bins, labels=labels)
    
    if 'title' in df.columns:
        df['title_length'] = df['title'].str.len()
    
    if 'brand' in df.columns:
        df['has_brand'] = df['brand'].notna()
    
    if 'category' in df.columns:
        df['has_category'] = df['category'].notna()
    
    if 'style' in df.columns:
        df['has_style'] = df['style'].notna()
    
    return df


def save_cleaned_data(df: pd.DataFrame, output_dir: Path):
    print(f"Saving cleaned unified data to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    core_columns = [
        'title', 'category', 'style', 'brand', 'price', 'size', 
        'condition', 'image', 'url', 'platform', 'item_id'
    ]
    
    available_core = [col for col in core_columns if col in df.columns]
    
    unified_csv = output_dir / 'unified_cleaned.csv'
    df_unified = df[available_core + [col for col in df.columns if col not in available_core]].copy()
    df_unified.to_csv(unified_csv, index=False, encoding='utf-8')
    print(f"Saved unified CSV: {unified_csv}")
    
    json_file = output_dir / 'unified_cleaned.json'
    df.to_json(json_file, orient='records', indent=2, force_ascii=False)
    print(f"Saved full JSON: {json_file}")
    
    if 'platform' in df.columns:
        for platform in df['platform'].unique():
            if pd.notna(platform):
                platform_df = df[df['platform'] == platform]
                platform_file = output_dir / f'{platform}_cleaned.json'
                platform_df.to_json(platform_file, orient='records', indent=2, force_ascii=False)
                print(f"Saved {platform}: {platform_file} ({len(platform_df)} items)")
    
    stats_file = output_dir / 'unified_stats.json'
    stats = {
        'total_items': len(df),
        'platforms': df['platform'].value_counts().to_dict() if 'platform' in df.columns else {},
        'price_range': {
            'min': float(df['price'].min()) if 'price' in df.columns and df['price'].notna().any() else None,
            'max': float(df['price'].max()) if 'price' in df.columns and df['price'].notna().any() else None,
            'median': float(df['price'].median()) if 'price' in df.columns and df['price'].notna().any() else None,
            'mean': float(df['price'].mean()) if 'price' in df.columns and df['price'].notna().any() else None,
        },
        'field_completeness': {
            col: float((df[col].notna().sum() / len(df)) * 100)
            for col in core_columns if col in df.columns
        },
        'conditions': df['condition'].value_counts().to_dict() if 'condition' in df.columns else {},
        'categories': df['category'].value_counts().head(10).to_dict() if 'category' in df.columns else {},
        'top_brands': df['brand'].value_counts().head(10).to_dict() if 'brand' in df.columns else {},
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics: {stats_file}")
    
    print(f"Unified Data Summary:")
    print(f"Total items: {len(df)}")
    if 'platform' in df.columns:
        print(f"Platforms: {', '.join(df['platform'].unique())}")
    if 'price' in df.columns and df['price'].notna().any():
        print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        print(f"Median price: ${df['price'].median():.2f}")
    print(f"Field completeness:")
    for col in core_columns:
        if col in df.columns:
            completeness = (df[col].notna().sum() / len(df)) * 100
            print(f"{col}: {completeness:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Clean and deduplicate scraped marketplace data')
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/scraped',
        help='Input directory with scraped data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/scraped/cleaned',
        help='Output directory for cleaned data'
    )
    
    parser.add_argument(
        '--remove-no-images',
        action='store_true',
        help='Remove items without valid image URLs'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return
    
    print("Marketplace Data Cleaning & Normalization")
    df = load_scraped_data(input_dir)
    if df.empty:
        print("No data to clean!")
        return
    
    print(f"Initial dataset: {len(df)} items")
    df = normalize_platform_data(df)
    df = remove_duplicates(df)
    df = clean_prices(df)
    df = clean_text_fields(df)
    df = standardize_conditions(df)
    df = validate_image_urls(df)
    if args.remove_no_images:
        print("Filtering items without images...")
        initial = len(df)
        df = df[df['image'].apply(lambda x: isinstance(x, str) and x.startswith('http'))].copy()
        removed = initial - len(df)
        print(f"Removed {removed} items without images")
    
    df = add_derived_fields(df)
    save_cleaned_data(df, output_dir)
    print("Cleaning Complete!")
    print(f"Final dataset: {len(df)} items")
    print(f"Output directory: {output_dir}")
    print(f"Primary output file: {output_dir}/unified_cleaned.csv")


if __name__ == '__main__':
    main()
