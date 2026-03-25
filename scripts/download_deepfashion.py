#!/usr/bin/env python3

import argparse
import json
import sys
import tarfile
import zipfile
import gdown
rom pathlib import Path

from tqdm import tqdm


class DeepFashionDownloader:
    DATASET_INFO = {
        "category": {
            "name": "Category and Attribute Prediction",
            "url": "https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc",
            "files": [
                "img.zip",
                "Anno/list_attr_cloth.txt",
                "Anno/list_attr_img.txt",
                "Anno/list_bbox_inshop.txt",
                "Anno/list_category_cloth.txt",
                "Anno/list_category_img.txt",
                "Eval/list_eval_partition.txt",
            ],
            "expected_images": 289222,
        },
        "inshop": {
            "name": "In-Shop Clothes Retrieval",
            "url": "https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E",
            "files": [
                "img.zip",
                "Anno/list_bbox_inshop.txt",
                "Anno/list_description_inshop.json",
                "Anno/list_item_inshop.txt",
                "Eval/list_eval_partition.txt",
            ],
            "expected_images": 52712,
        },
    }

    def __init__(self, output_dir: str = "data/raw/deepfashion/original"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir = self.output_dir / "img"
        self.anno_dir = self.output_dir / "Anno"
        self.eval_dir = self.output_dir / "Eval"
        for directory in (self.img_dir, self.anno_dir, self.eval_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def subsets(subset: str):
        return ["category", "inshop"] if subset == "all" else [subset]

    def print_download_instructions(self, subset: str = "all"):
        subsets = self.subsets(subset)
        print("DeepFashion Download")
        for subset_name in subsets:
            info = self.DATASET_INFO[subset_name]
            print(info["name"])
            print(f"  {info['url']}")
            for file_path in info["files"]:
                print(f"    {file_path}")
        print()
        print(f"python {sys.argv[0]} --subset {subset} --download --extract --verify")

    def download_from_google_drive(self, subset: str = "all"):
        results = {}
        for subset_name in self._subsets(subset):
            info = self.DATASET_INFO[subset_name]
            target_dir = self.output_dir / subset_name
            target_dir.mkdir(parents=True, exist_ok=True)
            downloaded = gdown.download_folder(
                url=info["url"],
                output=str(target_dir),
                quiet=False,
                use_cookies=False,
                remaining_ok=True,
            )
            results[subset_name] = bool(downloaded)
        return results

    def verify_dataset(self, subset: str = "all"):
        results = {}
        for subset_name in self._subsets(subset):
            info = self.DATASET_INFO[subset_name]
            found = []
            missing = []
            for file_path in info["files"]:
                full_path = self.output_dir / file_path
                if full_path.exists():
                    found.append(file_path)
                else:
                    missing.append(file_path)
            results[subset_name] = {
                "name": info["name"],
                "files_found": found,
                "files_missing": missing,
                "all_found": not missing,
            }
        return results

    def count_images(self, directory: Path) -> int:
        total = 0
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            total += len(list(directory.rglob(f"*{ext}")))
        return total

    def extract_archives(self):
        archives = list(self.output_dir.rglob("*.zip")) + list(self.output_dir.rglob("*.tar*"))
        for archive_path in archives:
            extract_dir = archive_path.parent
            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    for member in tqdm(zip_ref.namelist(), desc=f"Extracting {archive_path.name}"):
                        zip_ref.extract(member, extract_dir)
            else:
                with tarfile.open(archive_path, "r:*") as tar_ref:
                    tar_ref.extractall(extract_dir)

    def generate_dataset_stats(self, subset: str = "all"):
        stats = {
            "subsets": {},
            "total_images": 0,
            "timestamp": str(Path.cwd()),
        }
        for subset_name in self._subsets(subset):
            info = self.DATASET_INFO[subset_name]
            image_count = self.count_images(self.img_dir)
            subset_stats = {
                "name": info["name"],
                "expected_images": info["expected_images"],
                "found_images": image_count,
                "complete": image_count >= info["expected_images"],
                "completion_percentage": (
                    image_count / info["expected_images"] * 100 if info["expected_images"] else 0
                ),
            }
            stats["subsets"][subset_name] = subset_stats
            stats["total_images"] += image_count
        stats_file = self.output_dir.parent / "metadata" / "dataset_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        return stats

    def create_data_splits(self):
        eval_file = self.eval_dir / "list_eval_partition.txt"
        splits = {"train": [], "val": [], "test": []}
        with open(eval_file, "r") as f:
            for line in f.readlines()[2:]:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1] in splits:
                    splits[parts[1]].append(parts[0])
        metadata_dir = self.output_dir.parent / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        for split_name, image_list in splits.items():
            with open(metadata_dir / f"{split_name}_files.txt", "w") as f:
                f.write("\n".join(image_list))


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize DeepFashion dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download/download_deepfashion.py --subset all --instructions
  python scripts/download/download_deepfashion.py --subset all --download --extract --verify
        """,
    )
    parser.add_argument("--subset", choices=["all", "category", "inshop"], default="all")
    parser.add_argument("--output", default="data/raw/deepfashion/original")
    parser.add_argument("--instructions", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--splits", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    downloader = DeepFashionDownloader(output_dir=args.output)
    if not any([args.instructions, args.download, args.verify, args.extract, args.splits, args.stats]):
        args.instructions = True

    if args.instructions:
        downloader.print_download_instructions(args.subset)
    if args.download:
        downloader.download_from_google_drive(args.subset)
    if args.verify:
        print(json.dumps(downloader.verify_dataset(args.subset), indent=2))
    if args.extract:
        downloader.extract_archives()
    if args.splits:
        downloader.create_data_splits()
    if args.stats:
        print(json.dumps(downloader.generate_dataset_stats(args.subset), indent=2))


if __name__ == "__main__":
    main()
