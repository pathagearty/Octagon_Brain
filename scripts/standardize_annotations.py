"""Standardize BoxingVI annotation files to a consistent format.

Output format:
- Columns: start_frame, end_frame, class
- Class names: jab, cross, lead_hook, rear_hook, lead_uppercut, rear_uppercut
"""
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime


# Standard class name mapping
CLASS_MAPPING = {
    "jab": "jab",
    "cross": "cross",
    "straight": "cross",
    "lead hook": "lead_hook",
    "lead_hook": "lead_hook",
    "left hook": "lead_hook",
    "rear hook": "rear_hook",
    "rear_hook": "rear_hook",
    "right hook": "rear_hook",
    "lead uppercut": "lead_uppercut",
    "lead_uppercut": "lead_uppercut",
    "left uppercut": "lead_uppercut",
    "rear uppercut": "rear_uppercut",
    "rear_uppercut": "rear_uppercut",
    "right uppercut": "rear_uppercut",
}


def find_column_by_content(df: pd.DataFrame, expected_values: list) -> int:
    """Find column index that contains expected values."""
    for col_idx, col in enumerate(df.columns):
        values = df[col].dropna().astype(str).str.lower().str.strip().unique()
        matches = sum(1 for v in values if any(exp in v for exp in expected_values))
        if matches >= 2:  # At least 2 class matches
            return col_idx
    return -1


def find_numeric_columns(df: pd.DataFrame) -> tuple:
    """Find the start and end frame columns (numeric)."""
    numeric_cols = []
    for col_idx, col in enumerate(df.columns):
        try:
            # Check if mostly numeric
            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
            if numeric_count > len(df) * 0.8:  # 80% numeric
                numeric_cols.append(col_idx)
        except:
            pass

    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]
    return -1, -1


def standardize_class_name(raw_class: str) -> str:
    """Convert raw class name to standard format."""
    cleaned = raw_class.lower().strip().replace("_", " ")

    if cleaned in CLASS_MAPPING:
        return CLASS_MAPPING[cleaned]

    # Fuzzy match
    for key, value in CLASS_MAPPING.items():
        if key in cleaned or cleaned in key:
            return value

    print(f"    WARNING: Unknown class '{raw_class}' - defaulting to 'jab'")
    return "jab"


def standardize_file(input_path: Path, output_path: Path) -> dict:
    """Standardize a single annotation file."""
    stats = {"rows": 0, "classes": set(), "issues": []}

    try:
        # Read raw (no header assumption)
        df_raw = pd.read_excel(input_path, header=None)

        # Also try with headers
        df_header = pd.read_excel(input_path)

        # Determine if first row is header
        first_row = df_raw.iloc[0].astype(str).str.lower().tolist()
        has_header = any(
            val in ['start', 'end', 'class', 'start_frame', 'end_frame', 'start frame']
            for val in first_row
        )

        if has_header:
            df = df_header
            # Strip whitespace from column names
            df.columns = [str(c).strip() for c in df.columns]
        else:
            df = df_raw
            stats["issues"].append("No header row detected")

        # Find class column
        class_keywords = ["jab", "cross", "hook", "uppercut"]

        # First try by column name
        class_col_idx = -1
        for idx, col in enumerate(df.columns):
            col_str = str(col).lower().strip()
            if col_str in ["class", "punch_class", "action", "label", "type"]:
                class_col_idx = idx
                break

        # If not found by name, find by content
        if class_col_idx == -1:
            class_col_idx = find_column_by_content(df, class_keywords)

        if class_col_idx == -1:
            stats["issues"].append("Could not identify class column")
            return stats

        # Find start/end frame columns (first two numeric columns before class)
        start_col_idx, end_col_idx = find_numeric_columns(df)

        if start_col_idx == -1:
            stats["issues"].append("Could not identify frame columns")
            return stats

        # Build standardized dataframe
        standardized_rows = []

        for idx, row in df.iterrows():
            try:
                start_frame = int(float(row.iloc[start_col_idx]))
                end_frame = int(float(row.iloc[end_col_idx]))
                raw_class = str(row.iloc[class_col_idx])

                if pd.isna(row.iloc[class_col_idx]) or raw_class.lower() == 'nan':
                    continue

                std_class = standardize_class_name(raw_class)

                standardized_rows.append({
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "class": std_class
                })

                stats["classes"].add(std_class)

            except (ValueError, TypeError) as e:
                stats["issues"].append(f"Row {idx}: {e}")
                continue

        # Create output dataframe
        output_df = pd.DataFrame(standardized_rows)
        stats["rows"] = len(output_df)

        # Save
        output_df.to_excel(output_path, index=False)

    except Exception as e:
        stats["issues"].append(f"Error: {e}")

    return stats


def main():
    annotation_dir = Path(r"C:\Users\patri\OneDrive\octagon-brain-cv\data\boxingvi\Annotation_files")

    # Create backup directory
    backup_dir = annotation_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(exist_ok=True)

    # Create standardized output directory
    output_dir = annotation_dir / "standardized"
    output_dir.mkdir(exist_ok=True)

    print("BoxingVI Annotation Standardization")
    print("=" * 50)
    print(f"Input: {annotation_dir}")
    print(f"Output: {output_dir}")
    print(f"Backup: {backup_dir}")
    print()

    excel_files = sorted([f for f in annotation_dir.glob("*.xlsx") if not f.name.startswith("~$")])

    all_stats = {}

    for excel_path in excel_files:
        print(f"Processing {excel_path.name}...")

        # Backup original
        try:
            shutil.copy(excel_path, backup_dir / excel_path.name)
        except PermissionError:
            print(f"  SKIPPED: File is open (close Excel)")
            continue

        # Standardize
        output_path = output_dir / excel_path.name
        stats = standardize_file(excel_path, output_path)
        all_stats[excel_path.name] = stats

        print(f"  Rows: {stats['rows']}")
        print(f"  Classes: {stats['classes']}")
        if stats['issues']:
            print(f"  Issues: {stats['issues'][:3]}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    total_rows = sum(s['rows'] for s in all_stats.values())
    all_classes = set()
    for s in all_stats.values():
        all_classes.update(s['classes'])

    print(f"Total files processed: {len(all_stats)}")
    print(f"Total rows: {total_rows}")
    print(f"Classes found: {sorted(all_classes)}")

    print(f"\nStandardized files saved to: {output_dir}")
    print("\nTo use standardized files:")
    print("1. Review the files in the 'standardized' folder")
    print("2. If they look correct, copy them back to replace originals")
    print("3. Or update your Kaggle dataset with the standardized files")


if __name__ == "__main__":
    main()
