from logging import INFO, basicConfig, getLogger
from pandas import read_csv
from pathlib import Path
from sys import path

root_dir = Path(__file__).parent.parent
path.insert(0, str(root_dir))

from src.models.naive_bayes_pipeline import NaiveBayesTrainer, NaiveBayesPredictor


logger = getLogger(__name__)
basicConfig(
    level=INFO, format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)

CLEANED_DATA_PATH = root_dir / "data/interim/cleaned_flights.csv"


def main():
    trainer = NaiveBayesTrainer(
        model_name="naive_bayes_flight_delay", var_smoothing=1e-05
    )
    trainer.train_and_save()

    logger.info("\n" + "=" * 80)
    logger.info("🔮 TESTANDO PREDITOR")
    logger.info("=" * 80)

    predictor = NaiveBayesPredictor(model_name="naive_bayes_flight_delay")

    test_data = read_csv(CLEANED_DATA_PATH).head(10)
    predictions = predictor.predict(test_data)

    logger.info(f"📊 First 10 predictions:")
    logger.info(predictions)


if __name__ == "__main__":
    main()
