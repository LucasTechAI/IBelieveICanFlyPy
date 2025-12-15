from pathlib import Path
from sys import path

project_root = Path(__file__).parent.parent
path.insert(0, str(project_root))

from src.etl.data_cleaning import DataCleaner

def main():
    data_cleaner = DataCleaner()
    cleaned_flights_df = data_cleaner.clean_flights_data()
    data_cleaner.save_cleaned_data(cleaned_flights_df)

if __name__ == "__main__":
    main()