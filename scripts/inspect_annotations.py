"""Inspect BoxingVI annotation files to identify structure variations."""
import pandas as pd
from pathlib import Path
import sys


def inspect_excel_files(annotation_dir: Path) -> None:
    """Inspect all Excel files and report their structure."""

    excel_files = sorted(annotation_dir.glob("*.xlsx"))

    if not excel_files:
        print(f"No Excel files found in {annotation_dir}")
        return

    print(f"Found {len(excel_files)} annotation files\n")
    print("=" * 80)

    structures = {}

    for excel_path in excel_files:
        print(f"\n📄 {excel_path.name}")
        print("-" * 40)

        try:
            # Read without assuming headers
            df_raw = pd.read_excel(excel_path, header=None)
            df_with_header = pd.read_excel(excel_path)

            print(f"  Rows: {len(df_with_header)}")
            print(f"  Columns: {list(df_with_header.columns)}")

            # Check if first row looks like headers
            first_row = df_raw.iloc[0].tolist()
            looks_like_header = any(
                isinstance(v, str) and v.lower() in ['class', 'start', 'end', 'frame', 'action', 'label']
                for v in first_row
            )
            print(f"  Has headers: {looks_like_header}")

            # Show first few rows
            print(f"  First 3 rows:")
            for i in range(min(3, len(df_with_header))):
                row = df_with_header.iloc[i].tolist()
                print(f"    {i}: {row}")

            # Check for class column
            class_col = None
            for col in df_with_header.columns:
                if str(col).lower() in ['class', 'punch_class', 'action', 'label']:
                    class_col = col
                    break

            if class_col:
                unique_classes = df_with_header[class_col].dropna().unique()
                print(f"  Class column: '{class_col}'")
                print(f"  Unique classes: {list(unique_classes)}")
            else:
                print(f"  ⚠️  No recognizable class column found!")
                # Try to guess which column has class data
                for col in df_with_header.columns:
                    unique_vals = df_with_header[col].dropna().unique()
                    if len(unique_vals) < 10:  # Likely categorical
                        str_vals = [str(v).lower() for v in unique_vals]
                        if any(c in str_vals for c in ['jab', 'cross', 'hook', 'uppercut']):
                            print(f"  💡 Column '{col}' might be the class column: {list(unique_vals)}")

            # Store structure signature
            sig = (tuple(df_with_header.columns), looks_like_header)
            if sig not in structures:
                structures[sig] = []
            structures[sig].append(excel_path.name)

        except Exception as e:
            print(f"  ❌ Error reading file: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: File Structure Groups")
    print("=" * 80)

    for i, (sig, files) in enumerate(structures.items(), 1):
        cols, has_header = sig
        print(f"\nGroup {i} ({len(files)} files):")
        print(f"  Columns: {list(cols)}")
        print(f"  Has headers: {has_header}")
        print(f"  Files: {files}")


if __name__ == "__main__":
    # Default path - adjust as needed
    if len(sys.argv) > 1:
        annotation_dir = Path(sys.argv[1])
    else:
        # Try common locations
        possible_paths = [
            Path("data/boxingvi/Annotation_files"),
            Path("Annotation_files"),
            Path("."),
        ]
        annotation_dir = None
        for p in possible_paths:
            if p.exists() and list(p.glob("*.xlsx")):
                annotation_dir = p
                break

        if annotation_dir is None:
            print("Usage: python inspect_annotations.py <path_to_annotation_files>")
            print("No annotation directory found in default locations")
            sys.exit(1)

    inspect_excel_files(annotation_dir)
