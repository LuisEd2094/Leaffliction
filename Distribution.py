#!/usr/bin/env python3
"""
Leaffliction - Part 1: Analysis of the Data Set
Description:
    Analyzes a plant leaf image dataset and displays the distribution
    of images per category using pie and bar charts.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def count_images_in_subdirs(base_dir: str) -> dict:
    """
    Recursively counts the number of image files in each subdirectory of base_dir.
    Returns a dictionary {category_name: image_count}.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    categories = {}

    # Walk only one level deep (subdirectories)
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            subdir = entry.path
            image_count = sum(
                1 for f in os.listdir(subdir)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            )
            categories[entry.name] = image_count

    if not categories:
        raise ValueError(f"No subdirectories found in {base_dir}")

    return categories


def display_charts(categories: dict, dataset_name: str, save: bool = False):
    """
    Displays or saves pie and bar charts for the given category data.
    If --save is set, charts are saved but not shown.
    """
    labels = list(categories.keys())
    counts = list(categories.values())

    if not any(counts):
        print("No images found in the provided dataset.")
        return

    # --- Pie chart ---
    plt.figure(figsize=(7, 7))
    plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title(f"Distribution of {dataset_name} dataset")

    if save:
        plt.savefig(f"{dataset_name}_pie_chart.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # --- Bar chart ---
    plt.figure(figsize=(8, 5))
    colors = plt.cm.tab20.colors
    bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
    plt.bar(labels, counts, color=bar_colors)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Number of images")
    plt.title(f"Image count per category in {dataset_name}")
    plt.tight_layout()

    if save:
        plt.savefig(f"{dataset_name}_bar_chart.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyze a leaf dataset and display image distribution charts."
    )
    parser.add_argument(
        "directory",
        help="Path to the dataset directory (e.g., ./Apple or ./Dataset)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the charts as PNG images in the current directory."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        categories = count_images_in_subdirs(args.directory)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    dataset_name = os.path.basename(os.path.normpath(args.directory))
    print(f"\n📂 Analyzing dataset: {dataset_name}")
    for name, count in categories.items():
        print(f"  • {name}: {count} images")

    display_charts(categories, dataset_name, save=args.save)

    if args.save:
        print(f"\nCharts saved as '{dataset_name}_pie_chart.png' and '{dataset_name}_bar_chart.png'")


if __name__ == "__main__":
    main()
