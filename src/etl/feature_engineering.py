from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from pandas import DataFrame, read_csv, cut, concat
from logging import getLogger, basicConfig, INFO
from typing import List, Optional
from numpy import pi, sin, cos
from scipy.stats import zscore

logger = getLogger(__name__)
basicConfig(
    level=INFO, format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)


class FeatureEngineer:
    def __init__(self, df_path: Optional[str] = None) -> None:
        """
        Initializes the feature engineering helper and optionally loads a CSV.

        Args:
            df_path (Optional[str]): Path to a CSV file to load into self.df. If None, no data is loaded.

        Attributes:
            df (Optional[DataFrame]): The DataFrame to be processed.
            scalers (dict): Stored scaler objects for normalized columns.
            encoders (dict): Stored encoder objects for categorical features.
        """
        if df_path:
            self.df = read_csv(df_path)
        else:
            self.df = None

        self.scalers = {}
        self.encoders = {}

    def load_data(self, df_path: str) -> "FeatureEngineer":
        """
        Loads data from a CSV file into the dataframe.
        Args:
            df_path: Path to the CSV file.
        Returns:
            self (for chaining)
        """
        self.df = read_csv(df_path)
        return self

    def set_dataframe(self, df: DataFrame) -> "FeatureEngineer":
        """
        Sets the dataframe directly.
        Args:
            df: DataFrame to set.
        Returns:
            self (for chaining)
        """
        self.df = df.copy()
        return self

    def encoding_cyclic_feature(
        self, col: str, period: int, prefix: Optional[str] = None
    ) -> "FeatureEngineer":
        """
        Applies cyclic encoding to a feature.
        Args:
            col: Name of the column to encode.
            period: The period of the cycle (e.g., 24 for hours in a day).
            prefix: Optional prefix for new columns. If None, uses col name.
        Returns:
            self (for chaining)
        """
        if self.df is None:
            raise ValueError(
                "DataFrame does not exist. Load data before applying transformations."
            )

        prefix = prefix.upper() if prefix else col.upper()

        self.df[f"{prefix}_SIN"] = sin(2 * pi * self.df[col] / period)
        self.df[f"{prefix}_COS"] = cos(2 * pi * self.df[col] / period)

        logger.info(f"-> Cyclic encoding applied: {col} → {prefix}_SIN, {prefix}_COS")
        return self

    def process_time_hhmm(
        self, col: str, keep_original: bool = False
    ) -> "FeatureEngineer":
        """
        Processing time in HHMM format into cyclic features.
        Args:
            col: Name of the column with time in HHMM format.
            keep_original: If True, keeps the original HHMM column.
        Returns:
            self (for chaining)
        """
        if self.df is None:
            raise ValueError(
                "DataFrame does not exist. Load data before applying transformations."
            )

        # Convert HHMM to minutes since midnight
        hours = self.df[col] // 100
        minutes = self.df[col] % 100
        total_minutes = hours * 60 + minutes

        col_minutes = f"{col}_MINUTES".upper()
        self.df[col_minutes] = total_minutes

        # Apply cyclic encoding (1440 minutes in a day)
        self.encoding_cyclic_feature(col_minutes, period=1440, prefix=col)

        if not keep_original:
            self.df.drop(columns=[col], inplace=True)

        logger.info(f"-> Time processed: {col} → {col_minutes}, {col}_SIN, {col}_COS")
        return self

    @staticmethod
    def __categorize_time(minutes: int) -> str:
        if 0 <= minutes < 360:  # 00:00 - 06:00
            return "midnight"
        elif 360 <= minutes < 720:  # 06:00 - 12:00
            return "morning"
        elif 720 <= minutes < 1080:  # 12:00 - 18:00
            return "afternoon"
        else:  # 18:00 - 24:00
            return "night"

    def create_time_periods(self, col_minutes: str) -> "FeatureEngineer":
        """
        Create time of day periods from minutes since midnight.
        Args:
            col_minutes: Column with minutes since midnight
        Returns:
            self
        """
        self.df[f"{col_minutes}_PERIOD"] = self.df[col_minutes].apply(
            self.__categorize_time
        )
        logger.info(f"-> Time periods created: {col_minutes}_PERIOD")
        return self

    def create_rush_hour(self, col_minutes: str) -> "FeatureEngineer":
        """
        Create a binary feature indicating rush hour periods.
        Args:
            col_minutes: Column with minutes since midnight
        Returns:
            self
        """
        # Rush hours: 06:00-09:00 (360-540), 17:00-20:00 (1020-1200)
        rush_morning = (self.df[col_minutes] >= 360) & (self.df[col_minutes] < 540)
        rush_evening = (self.df[col_minutes] >= 1020) & (self.df[col_minutes] < 1200)

        self.df[f"{col_minutes}_RUSH_HOUR"] = (rush_morning | rush_evening).astype(int)
        logger.info(f"-> Rush hour created: {col_minutes}_RUSH_HOUR")
        return self

    def normalize_column(self, col: str, method: str = "standard") -> "FeatureEngineer":
        """
        Normalize a numerical column.
        Args:
            col: Name of the column to normalize
            method: Normalization method ("standard" or "minmax")
        Returns:
            self
        """
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        self.df[f"{col}_NORMALIZED"] = scaler.fit_transform(self.df[[col]])
        self.scalers[col] = scaler

        logger.info(
            f"-> Column normalized: {col} → {col}_NORMALIZED using {method} scaler"
        )
        return self

    def create_distance_bins(
        self, col: str = "DISTANCE", bins: List[float] = None, labels: List[str] = None
    ) -> "FeatureEngineer":
        """
        Create distance categories based on specified bins.
        Args:
            col: Column with distance values
            bins: List of bin edges
            labels: List of labels for the bins
        Returns:
            self
        """
        DEFAULT_BINS = [0, 500, 1000, 2000, float("inf")]
        DEFAULT_LABELS = ["short", "medium", "long", "very_long"]

        bins = bins if bins is not None else DEFAULT_BINS
        labels = labels if labels is not None else DEFAULT_LABELS

        self.df[f"{col}_CATEGORY"] = cut(self.df[col], bins=bins, labels=labels)
        logger.info(f"-> Distance categories created: {col}_CATEGORY")
        return self

    def create_weekend_feature(self, col: str = "DAY_OF_WEEK") -> "FeatureEngineer":
        """
        Create a binary feature indicating weekends.
        Args:
            col: Column with day of the week (1=Monday, 7=Sunday)
        Returns:
            self
        """
        weekends = [6, 7]  # Saturday=6, Sunday=7
        self.df["IS_WEEKEND"] = self.df[col].isin(weekends).astype(int)
        logger.info(f"-> Weekend feature created: IS_WEEKEND")
        return self

    @staticmethod
    def __get_season(day: int) -> str:
        """
        Determines the season based on the day of the year (Hemisfério Norte).
        Args:
            day: Day of the year (1-365)
        Returns:
            str: Season name
        """
        if day < 80 or day >= 355:  # ~21 dez - ~20 mar
            return "winter"
        elif 80 <= day < 172:  # ~21 mar - ~20 jun
            return "spring"
        elif 172 <= day < 266:  # ~21 jun - ~22 set
            return "summer"
        elif 266 <= day < 355:
            return "autumn"

    def create_season(self, col: str = "DAY_OF_YEAR") -> "FeatureEngineer":
        """
        Cria feature de estação do ano (Hemisfério Norte).

        Args:
            col: Coluna do dia do ano (1-365)

        Returns:
            self
        """

        self.df["SEASON"] = self.df[col].apply(self.__get_season)
        logger.info(f"-> Season feature created: SEASON")
        return self

    def one_hot_encode(self, col: str) -> "FeatureEngineer":
        """
        Applies one-hot encoding to a categorical column.
        Args:
            col: Name of the column to encode
        Returns:
            self
        """
        if self.df is None:
            raise ValueError(
                "DataFrame does not exist. Load data before applying transformations."
            )
        
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(self.df[[col]])

        feature_names = encoder.get_feature_names_out([col])
        
        logger.info(f"Categories found: {encoder.categories_[0]}")
        encoded_df = DataFrame(
            encoded, 
            columns=feature_names,
            index=self.df.index
        )
        
        self.encoders[col] = encoder
        
        self.df = concat([self.df, encoded_df], axis=1)
        self.df.drop(columns=[col], inplace=True)

        logger.info(
            f"-> One-hot encoding applied: {col} → {len(encoded_df.columns)} columns: {encoded_df.columns.tolist()}"
        )
        return self

    def frequency_encode(self, col: str) -> "FeatureEngineer":
        """
        Applies frequency encoding to a categorical column.
        Args:
            col: Name of the column to encode
        Returns:
            self
        """
        freq = self.df[col].value_counts(normalize=True)
        self.df[f"{col}_FREQ"] = self.df[col].map(freq)

        logger.info(f"-> Frequency encoding applied: {col} → {col}_FREQ")
        return self

    def get_dataframe(self) -> DataFrame:
        """
        Returns the processed dataframe.
        Returns:
            DataFrame: The processed dataframe
        """
        return self.df

    def save_processed_data(self, output_path: str) -> "FeatureEngineer":
        """
        Saves the processed dataframe.
        Args:
            output_path: Path to save the CSV file
        Returns:
            self
        """
        self.df.to_csv(output_path, index=False)
        logger.info(f"-> Data saved to: {output_path}")
        return self

    def rename_columns_uppercase(self) -> "FeatureEngineer":
        """
        Renames all columns to uppercase.
        Returns:
            self
        """
        self.df.columns = [col.upper() for col in self.df.columns]
        logger.info("-> Columns renamed to uppercase.")
        return self

    def drop_columns(self, cols: List[str]) -> "FeatureEngineer":
        """
        Removes specified columns from the dataframe.
        Returns:
            self
        """
        self.df.drop(columns=cols, inplace=True)
        logger.info(f"-> Columns removed: {cols}")
        return self

    def set_all_columns_uppercase(self) -> "FeatureEngineer":
        """
        Sets all column names to uppercase.
        Returns:
            self
        """
        self.df.columns = [col.upper() for col in self.df.columns]
        logger.info("-> All columns set to uppercase.")
        return self

    def drop_NaN_rows(self) -> "FeatureEngineer":
        """
        Drops all rows with any NaN values.
        Returns:
            self
        """
        initial_count = len(self.df)
        self.df.dropna(inplace=True)
        final_count = len(self.df)
        logger.info(f"-> Dropped {initial_count - final_count} rows with NaN values.")
        return self

    def set_bools_to_ints(self, bool_cols: List[str] = "all") -> "FeatureEngineer":
        """
        Converts boolean columns to integers (0 and 1).
        Args:
            bool_cols: List of boolean column names to convert.
        Returns:
            self
        """
        if bool_cols == "all":
            bool_cols = self.df.select_dtypes(include=["bool"]).columns.tolist()
        for col in bool_cols:
            self.df[col] = self.df[col].astype(int)
        logger.info(f"-> Converted boolean columns to integers: {bool_cols}")
        return self

    def drop_outliers_zscore(
        self, col: str, threshold: float = 3.0
    ) -> "FeatureEngineer":
        """
        Removes outliers from a numerical column based on Z-score.
        Args:
            col: Name of the column to check for outliers
            threshold: Z-score threshold to identify outliers
        Returns:
            self
        """

        z_scores = zscore(self.df[col])
        mask = abs(z_scores) < threshold
        initial_count = len(self.df)
        self.df = self.df[mask]
        final_count = len(self.df)
        logger.info(
            f"-> Dropped {initial_count - final_count} outlier rows from column: {col}"
        )
        return self

