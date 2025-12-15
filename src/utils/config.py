from logging import getLogger, basicConfig, INFO
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import List, Dict
from pathlib import Path
from os import getenv

load_dotenv()

logger = getLogger(__name__)
basicConfig(
    level=INFO, format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)


@dataclass
class Config:
    """
    Centralized project configurations using environment variables.
    """

    ROOT_DIR: Path = Path(getenv("PROJECT_ROOT", Path.cwd()))
    DATA_DIR: Path = ROOT_DIR / getenv("DATA_DIR", "data")
    RAW_DATA_DIR: Path = DATA_DIR / getenv("RAW_DATA_DIR", "raw")
    INTERIM_DATA_DIR: Path = DATA_DIR / getenv("INTERIM_DATA_DIR", "interim")
    PROCESSED_DATA_DIR: Path = DATA_DIR / getenv("PROCESSED_DATA_DIR", "processed")

    AIRLINES_FILE: str = getenv("AIRLINES_FILE", "airlines.csv")
    AIRPORTS_FILE: str = getenv("AIRPORTS_FILE", "airports.csv")
    FLIGHTS_FILE: str = getenv("FLIGHTS_FILE", "flights.csv")

    LOWER_PERCENTILE: float = float(getenv("LOWER_PERCENTILE", "0.0"))
    UPPER_PERCENTILE: float = float(getenv("UPPER_PERCENTILE", "1.0"))

    RANDOM_SEED: int = int(getenv("RANDOM_SEED", "42"))

    COLUMNS_NAN_80_PERCENT: List[str] = None
    LEAKAGE_COLUMNS: List[str] = None
    USELESS_COLUMNS: List[str] = None
    DATE_COLUMNS: List[str] = None
    CATEGORICAL_COLUMNS: List[str] = None
    NUMERICAL_COLUMNS: List[str] = None

    def __post_init__(self):
        """Innit default column lists after initialization."""
        self.COLUMNS_NAN_80_PERCENT = [
            "CANCELLATION_REASON",
            "AIR_SYSTEM_DELAY",
            "SECURITY_DELAY",
            "AIRLINE_DELAY",
            "LATE_AIRCRAFT_DELAY",
            "WEATHER_DELAY",
        ]

        self.LEAKAGE_COLUMNS = [
            "DEPARTURE_TIME",
            "ARRIVAL_TIME",
            "WHEELS_ON",
            "AIR_TIME",
            "ELAPSED_TIME",
            "TAXI_IN",
        ]

        self.USELESS_COLUMNS = ["TAIL_NUMBER"]
        self.DATE_COLUMNS = ["YEAR", "MONTH", "DAY"]
        self.CATEGORICAL_COLUMNS = ["AIRLINE"]
        self.NUMERICAL_COLUMNS = [
            "DAY_OF_WEEK",
            "FLIGHT_NUMBER",
            "DISTANCE",
            "SCHEDULED_DEPARTURE",
            "SCHEDULED_TIME",
            "SCHEDULED_ARRIVAL",
            "DAY_OF_YEAR",
            "ORIGIN_AIRPORT_LATITUDE",
            "ORIGIN_AIRPORT_LONGITUDE",
            "DESTINATION_AIRPORT_LATITUDE",
            "DESTINATION_AIRPORT_LONGITUDE",
        ]

        self._create_directories()

        logger.info(f"Configuration loaded:")
        logger.info(f"Root directory: {self.ROOT_DIR}")
        logger.info(f"Data directory: {self.DATA_DIR}")
        logger.info(f"Random seed: {self.RANDOM_SEED}")

    def _create_directories(self):
        """Creates necessary directory structure."""
        for directory in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.INTERIM_DATA_DIR,
            self.PROCESSED_DATA_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def paths(self) -> Dict[str, Path]:
        """Return paths to raw data files."""
        return {
            "airlines": self.RAW_DATA_DIR / self.AIRLINES_FILE,
            "airports": self.RAW_DATA_DIR / self.AIRPORTS_FILE,
            "flights": self.RAW_DATA_DIR / self.FLIGHTS_FILE,
        }
