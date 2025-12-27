from logging import INFO, basicConfig, getLogger
from typing import List, Tuple
from pathlib import Path
import pandas as pd

logger = getLogger(__name__)
basicConfig(
    level=INFO, format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)

root_dir = Path(__file__).parent.parent
RAW_DATA_DIR = root_dir / "data/raw"
INTERIM_DATA_DIR = root_dir / "data/interim"


def get_file_size_mb(file_path: Path) -> float:
    """
    Return file size in megabytes.
    Args:
        file_path: Path to the file
    Returns:
        File size in MB
    """
    return file_path.stat().st_size / (1024 * 1024)


def convert_csv_to_pickle(csv_path: Path) -> Path:
    """
    Convert CSV to pickle.
    Args:
        csv_path: Path to the CSV file
    Returns:
        Path to the created pickle file
    """
    pickle_path = csv_path.with_suffix(".pkl")

    logger.info(f"📁 Convert CSV → Pickle: {csv_path.name}")
    logger.info(f"   Original size: {get_file_size_mb(csv_path):.1f} MB")

    df = pd.read_csv(csv_path)
    df.to_pickle(pickle_path)

    logger.info(f"   New size: {get_file_size_mb(pickle_path):.1f} MB")
    logger.info(
        f"   Reduction: {((get_file_size_mb(csv_path) - get_file_size_mb(pickle_path)) / get_file_size_mb(csv_path) * 100):.1f}%"
    )

    return pickle_path


def convert_pickle_to_csv(pickle_path: Path) -> Path:
    """
    Convert pickle to CSV.
    Args:
        pickle_path: Path to the pickle file
    Returns:
        Path to the created CSV file
    """
    csv_path = pickle_path.with_suffix(".csv")

    logger.info(f"📁 Convert Pickle → CSV: {pickle_path.name}")
    logger.info(f"   Original size: {get_file_size_mb(pickle_path):.1f} MB")

    df = pd.read_pickle(pickle_path)
    df.to_csv(csv_path, index=False)

    logger.info(f"   New size: {get_file_size_mb(csv_path):.1f} MB")

    return csv_path


def process_directory(
    directory: Path, csv_to_pickle: bool = True
) -> List[Tuple[Path, Path]]:
    """
    Process all files in a directory.

    Args:
        directory: Directory to process
        csv_to_pickle: If True, convert CSV→Pickle. If False, Pickle→CSV

    Returns:
        List of tuples (original_file, converted_file)
    """
    if not directory.exists():
        logger.warning(f"⚠️  Directory does not exist: {directory}")
        return []

    conversions = []

    if csv_to_pickle:
        csv_files = list(directory.glob("*.csv"))

        if not csv_files:
            logger.info(f"📂 No CSV files found in: {directory}")
            return []

        logger.info(f"{'='*80}")
        logger.info(f"🔄 CONVERTING CSV → PICKLE in {directory}")
        logger.info(f"   Found {len(csv_files)} CSV files")
        logger.info(f"{'='*80}")

        for csv_file in csv_files:
            pickle_file = csv_file.with_suffix(".pkl")

            if pickle_file.exists():
                logger.info(f"⏭️  Pickle file already exists: {pickle_file.name}")
                conversions.append((csv_file, pickle_file))
            else:
                new_pickle = convert_csv_to_pickle(csv_file)
                conversions.append((csv_file, new_pickle))

    else:
        pickle_files = list(directory.glob("*.pkl"))

        if not pickle_files:
            logger.info(f"📂 No Pickle files found in: {directory}")
            return []

        logger.info(f"{'='*80}")
        logger.info(f"🔄 CONVERTING PICKLE → CSV in {directory}")
        logger.info(f"   Found {len(pickle_files)} Pickle files")
        logger.info(f"{'='*80}")

        for pickle_file in pickle_files:
            csv_file = pickle_file.with_suffix(".csv")

            if csv_file.exists():
                logger.info(f"⏭️  CSV file already exists: {csv_file.name}")
                conversions.append((pickle_file, csv_file))
            else:
                new_csv = convert_pickle_to_csv(pickle_file)
                conversions.append((pickle_file, new_csv))

    return conversions


def auto_convert_data_files():
    """
    Automatically convert data files between CSV and Pickle formats
    based on the existing files in the raw and interim data directories.
    """
    logger.info("🚀 AUTOMATIC DATA CONVERSION")
    logger.info("=" * 80)

    directories_to_process = [RAW_DATA_DIR, INTERIM_DATA_DIR]

    total_conversions = 0

    for directory in directories_to_process:
        if not directory.exists():
            logger.info(f"📁 Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            continue

        csv_files = list(directory.glob("*.csv"))
        pickle_files = list(directory.glob("*.pkl"))

        if csv_files and not pickle_files:
            logger.info(
                f"📂 {directory.name}: Found only CSV files, converting to Pickle..."
            )
            conversions = process_directory(directory, csv_to_pickle=True)
            total_conversions += len(conversions)

        elif pickle_files and not csv_files:
            logger.info(
                f"📂 {directory.name}: Found only Pickle files, converting to CSV..."
            )
            conversions = process_directory(directory, csv_to_pickle=False)
            total_conversions += len(conversions)

        elif csv_files and pickle_files:
            logger.info(
                f"📂 {directory.name}: Found both CSV and Pickle files, no conversion needed"
            )

        else:
            logger.info(f"📂 {directory.name}: No data files found")

    logger.info("=" * 80)
    logger.info(f"✅ CONVERSION COMPLETED: {total_conversions} files processed")
    logger.info("=" * 80)


def main():
    auto_convert_data_files()


if __name__ == "__main__":
    main()
