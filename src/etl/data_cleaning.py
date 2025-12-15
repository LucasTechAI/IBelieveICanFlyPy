from pandas import DataFrame, read_csv, to_datetime
from logging import getLogger, basicConfig, INFO
from dotenv import load_dotenv
from pathlib import Path
from sys import path


current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
path.insert(0, str(root_dir))

from src.utils.config import Config  # Agora vai funcionar

from src.utils.config import Config

load_dotenv()

logger = getLogger(__name__)
basicConfig(
    level=INFO, format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)


class DataCleaner:
    def __init__(self) -> None:
        """
        Initializes the DataCleaner by loading datasets.
        """
        self.config = Config()
        self.airlines = read_csv(self.config.paths["airlines"])
        self.airports = read_csv(self.config.paths["airports"])
        self.flights = read_csv(self.config.paths["flights"])

    def __drop_nan_80_percent_columns(self, df: DataFrame) -> DataFrame:
        """
        Drops columns from the DataFrame that have more than 80% NaN values.
        Args:
            df (DataFrame): The input DataFrame.
        Returns:
            DataFrame: The DataFrame with specified columns dropped.
        """
        logger.info("Dropping columns with more than 80% NaN values")
        flights_df_cleaned = df.copy()
        flights_df_cleaned.drop(
            columns=self.config.COLUMNS_NAN_80_PERCENT, inplace=True
        )
        return flights_df_cleaned

    def __remove_leakage_columns(self, df: DataFrame) -> DataFrame:
        """
        Removes columns that could lead to data leakage.
        Args:
            df (DataFrame): The input DataFrame.
        Returns:
            DataFrame: The DataFrame with leakage columns removed.
        """
        logger.info("Removing columns that could lead to data leakage")
        df_cleaned = df.copy()
        df_cleaned.drop(columns=self.config.LEAKAGE_COLUMNS, inplace=True)
        return df_cleaned

    def __remove_canceled_and_diverted_flights(self, df: DataFrame) -> DataFrame:
        """
        Removes rows corresponding to canceled and diverted flights.
        Args:
            df (DataFrame): The input DataFrame.
        Returns:
            DataFrame: The DataFrame with canceled and diverted flights removed.
        """
        logger.info("Removing rows corresponding to canceled and diverted flights")
        df_cleaned = df.copy()
        not_canceled_or_diverted = (df_cleaned["CANCELLED"] == 0) & (
            df_cleaned["DIVERTED"] == 0
        )
        df_cleaned = df_cleaned[not_canceled_or_diverted]
        df_cleaned.drop(columns=["CANCELLED", "DIVERTED"], inplace=True)
        return df_cleaned

    def __drop_outliers(
        self, df: DataFrame, column: str, standard_deviations: float = 3
    ) -> DataFrame:
        """
        Drops outliers from a specified column based on standard deviations from the mean.
        Args:
            df (DataFrame): The input DataFrame.
            column (str): The column to check for outliers.
            standard_deviations (float): The number of standard deviations to use as the threshold.
        Returns:
            DataFrame: The DataFrame with outliers removed.
        """
        logger.info(f"Dropping outliers from column '{column}'")
        df_cleaned = df.copy()
        mean = df_cleaned[column].mean()
        std_dev = df_cleaned[column].std()
        lower_bound = mean - standard_deviations * std_dev
        upper_bound = mean + standard_deviations * std_dev
        df_cleaned = df_cleaned[
            (df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)
        ]
        df_cleaned.reset_index(drop=True, inplace=True)
        return df_cleaned

    def __convert_date_to_day_of_year(self, df: DataFrame) -> DataFrame:
        """
        Converts date columns (YEAR, MONTH, DAY) to a single DAY_OF_YEAR column.
        Args:
            df (DataFrame): The input DataFrame.
        Returns:
            DataFrame: The DataFrame with a new DAY_OF_YEAR column and original date columns removed.
        """
        logger.info("Converting date columns to DAY_OF_YEAR")
        df_cleaned = df.copy()
        df_cleaned["DAY_OF_YEAR"] = to_datetime(
            df_cleaned[self.config.DATE_COLUMNS]
        ).dt.dayofyear

        df_cleaned.drop(columns=self.config.DATE_COLUMNS, inplace=True)

        return df_cleaned

    def __set_target_column(
        self,
        df: DataFrame,
        lower_percentile: float = None,
        upper_percentile: float = None,
    ) -> DataFrame:
        """
        Sets the target column 'ARRIVAL_DELAY' by removing outliers based on specified percentiles
        Args:
            df (DataFrame): The input DataFrame.
            lower_percentile (float): The lower percentile threshold.
            upper_percentile (float): The upper percentile threshold.
        Returns:
            DataFrame: The DataFrame with outliers in 'ARRIVAL_DELAY' removed.
        """
        if lower_percentile is None:
            lower_percentile = self.config.LOWER_PERCENTILE
        if upper_percentile is None:
            upper_percentile = self.config.UPPER_PERCENTILE

        logger.info("Setting target column 'ARRIVAL_DELAY' by removing outliers")
        df_cleaned = df.copy()
        arrival_delay = df_cleaned["ARRIVAL_DELAY"]
        lower_bound = arrival_delay.quantile(lower_percentile)
        upper_bound = arrival_delay.quantile(upper_percentile)
        df_cleaned = df_cleaned[
            (arrival_delay >= lower_bound) & (arrival_delay <= upper_bound)
        ]

        df_cleaned.reset_index(drop=True, inplace=True)
        df_cleaned.dropna(subset=["ARRIVAL_DELAY"], inplace=True)
        return df_cleaned

    @staticmethod
    def get_airport_coord(
        airports_match: dict, iata_code: str, coord_type: str
    ) -> float:
        """
        Retrieves the latitude or longitude of an airport given its IATA code.
        Args:
            airports_match (dict): A dictionary mapping IATA codes to their coordinates.
            iata_code (str): The IATA code of the airport.
            coord_type (str): The type of coordinate to retrieve ('LATITUDE' or 'LONGITUDE').
        Returns:
            float: The requested coordinate value.
        """
        try:
            return airports_match[iata_code][coord_type]
        except KeyError:
            logger.warning(f"IATA code '{iata_code}' not found in airports data")
            return None

    def __convert_airport_codes_to_lat_long(self, df: DataFrame) -> DataFrame:
        """
        Converts airport codes to their corresponding latitude and longitude.
        Args:
            df (DataFrame): The input DataFrame.
        Returns:
            DataFrame: The DataFrame with airport codes replaced by latitude and longitude.
        """
        logger.info("Converting airport codes to latitude and longitude")
        df_cleaned = df.copy()

        airports_coords = self.airports[["IATA_CODE", "LATITUDE", "LONGITUDE"]]
        airports_coords.index = airports_coords["IATA_CODE"]
        airports_coords.drop(columns=["IATA_CODE"], inplace=True)
        airports_coords_dict = airports_coords.to_dict(orient="index")

        df_cleaned["ORIGIN_AIRPORT_LATITUDE"] = df_cleaned["ORIGIN_AIRPORT"].map(
            lambda x: self.get_airport_coord(airports_coords_dict, x, "LATITUDE")
        )

        df_cleaned["ORIGIN_AIRPORT_LONGITUDE"] = df_cleaned["ORIGIN_AIRPORT"].map(
            lambda x: self.get_airport_coord(airports_coords_dict, x, "LONGITUDE")
        )

        df_cleaned["DESTINATION_AIRPORT_LATITUDE"] = df_cleaned[
            "DESTINATION_AIRPORT"
        ].map(lambda x: self.get_airport_coord(airports_coords_dict, x, "LATITUDE"))

        df_cleaned["DESTINATION_AIRPORT_LONGITUDE"] = df_cleaned[
            "DESTINATION_AIRPORT"
        ].map(lambda x: self.get_airport_coord(airports_coords_dict, x, "LONGITUDE"))

        df_cleaned.drop(columns=["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], inplace=True)
        return df_cleaned

    def __drop_useless_columns(self, df: DataFrame) -> DataFrame:
        """
        Drops useless columns from the DataFrame.
        Args:
            df (DataFrame): The input DataFrame.
        Returns:
            DataFrame: The DataFrame with useless columns dropped.
        """
        logger.info("Dropping useless columns")
        df_cleaned = df.copy()
        df_cleaned.drop(columns=self.config.USELESS_COLUMNS, inplace=True)
        return df_cleaned

    def clean_flights_data(self) -> DataFrame:
        """
        Cleans the flight data by applying various preprocessing steps.
        Returns:
            DataFrame: The cleaned flight DataFrame.
        """
        logger.info("Starting flight data cleaning process")
        flights_df_cleaned = self.flights.copy()

        flights_df_cleaned = self.__drop_nan_80_percent_columns(flights_df_cleaned)
        flights_df_cleaned = self.__remove_leakage_columns(flights_df_cleaned)
        flights_df_cleaned = self.__remove_canceled_and_diverted_flights(
            flights_df_cleaned
        )
        flights_df_cleaned = self.__convert_date_to_day_of_year(flights_df_cleaned)
        flights_df_cleaned = self.__convert_airport_codes_to_lat_long(
            flights_df_cleaned
        )
        flights_df_cleaned = self.__set_target_column(flights_df_cleaned)
        flights_df_cleaned = self.__drop_useless_columns(flights_df_cleaned)
        flights_df_cleaned = self.__drop_outliers(
            flights_df_cleaned, "ARRIVAL_DELAY", standard_deviations=1
        )
        flights_df_cleaned.dropna(inplace=True)
        flights_df_cleaned.reset_index(drop=True, inplace=True)
        return flights_df_cleaned

    def save_cleaned_data(self, df: DataFrame, path: str = None) -> None:
        """
        Saves the cleaned DataFrame to a CSV file.
        Args:
            df (DataFrame): The cleaned DataFrame.
            path (str): The file path to save the CSV.
        """
        if path is None:
            path = self.config.INTERIM_DATA_DIR / "cleaned_flights.csv"
        df.to_csv(path, index=False)
        logger.info(f"Cleaned data saved to {path}")
