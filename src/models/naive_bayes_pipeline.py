from sklearn.model_selection import cross_val_score, KFold, train_test_split
from logging import getLogger, basicConfig, INFO
from pandas import read_csv, DataFrame, Series
from sklearn.naive_bayes import GaussianNB
from typing import Tuple, Dict, Optional
from multiprocessing import cpu_count
from numpy import mean, std
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from joblib import dump, load

from pathlib import Path
from gc import collect
from time import time
from sys import path
import json


# Setup Logger
logger = getLogger(__name__)
basicConfig(
    level=INFO, format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)

# Paths Configuration
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
path.append(str(root_dir))
from src.etl.feature_engineering import FeatureEngineer

# Constants
MODELS_DIR = root_dir / "data/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_DATA_PATH = root_dir / "data/interim/cleaned_flights.csv"
N_CORES = max(1, cpu_count() - 2)

logger.info(f"🖥️  CPUs available: {cpu_count()} | Using: {N_CORES} cores")

# Features selecionadas (DEVE ser definido aqui para consistência)
KEEP_FEATURES = [
    "WHEELS_OFF",
    "SCHEDULED_DEPARTURE_SIN",
    "FLIGHT_NUMBER_FREQ",
    "SCHEDULED_ARRIVAL_COS",
    "SCHEDULED_ARRIVAL_SIN",
    "ORIGIN_AIRPORT_LATITUDE",
    "TAXI_OUT",
    "DESTINATION_AIRPORT_LATITUDE",
    "DESTINATION_AIRPORT_LONGITUDE",
    "ORIGIN_AIRPORT_LONGITUDE",
    "DEPARTURE_DELAY",
    "DISTANCE_NORMALIZED",
    "DAY_OF_YEAR_COS",
    "DAY_OF_YEAR_SIN",
    "SCHEDULED_TIME_NORMALIZED",
]


class NaiveBayesTrainer:
    """
    Class to train and save a Naive Bayes model for flight delay prediction.
    """

    def __init__(
        self,
        model_name: str = "naive_bayes_model",
        test_size: float = 0.3,
        random_state: int = 42,
        var_smoothing: float = 1e-05,
    ):
        """
        Initializes the Naive Bayes trainer.

        Args:
            model_name: Name of the model to save
            test_size: Proportion of data for testing
            random_state: Seed for reproducibility
            var_smoothing: Smoothing parameter (default: 1e-05)
        """
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        self.var_smoothing = var_smoothing
        self.model_path = MODELS_DIR / f"{model_name}.pkl"
        self.metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        self.fe = FeatureEngineer()

        self.model: Optional[GaussianNB] = None
        self.metrics: Optional[Dict] = None
        self.feature_columns: Optional[list] = None

    def preprocess_flights_data(self, flights_df: DataFrame) -> DataFrame:
        """
        Applies feature engineering to flight data.

        Args:
            flights_df: DataFrame with cleaned flight data

        Returns:
            DataFrame with engineered features
        """
        logger.info("🔧 Applying feature engineering...")

        self.fe.set_dataframe(flights_df.copy())

        COLUMNS_DROP = [
            "DAY_OF_WEEK",
            "DAY_OF_YEAR",
            "SCHEDULED_DEPARTURE_MINUTES",
            "SCHEDULED_ARRIVAL_MINUTES",
            "SCHEDULED_DEPARTURE",
            "SCHEDULED_ARRIVAL",
            "FLIGHT_NUMBER",
            "DISTANCE",
            "SCHEDULED_TIME",
        ]

        (
            self.fe.encoding_cyclic_feature("DAY_OF_WEEK", period=7)
            .encoding_cyclic_feature("DAY_OF_YEAR", period=365)
            .process_time_hhmm("SCHEDULED_DEPARTURE", keep_original=True)
            .process_time_hhmm("SCHEDULED_ARRIVAL", keep_original=True)
            .create_weekend_feature()
            .create_season()
            .normalize_column("DISTANCE")
            .normalize_column("SCHEDULED_TIME")
            .create_time_periods("SCHEDULED_DEPARTURE")
            .create_time_periods("SCHEDULED_ARRIVAL")
            .create_distance_bins()
            .one_hot_encode("AIRLINE")
            .one_hot_encode("SEASON")
            .one_hot_encode("DISTANCE_CATEGORY")
            .one_hot_encode("SCHEDULED_DEPARTURE_PERIOD")
            .one_hot_encode("SCHEDULED_ARRIVAL_PERIOD")
            .frequency_encode("FLIGHT_NUMBER")
            .rename_columns_uppercase()
            .drop_columns(COLUMNS_DROP)
            .drop_NaN_rows()
            .set_bools_to_ints(bool_cols="all")
        )

        del flights_df
        collect()

        return self.fe.get_dataframe()

    def split_and_preprocess_data(
        self, cleaned_flights_df: DataFrame
    ) -> Tuple[DataFrame, DataFrame, Series, Series]:
        """
        Splits and processes data into training and testing sets.

        Args:
            cleaned_flights_df: DataFrame with cleaned data

        Returns:
            Processed X_train, X_test, y_train, y_test
        """
        logger.info(f"{'='*80}")
        logger.info(f"📊 DATA SPLIT & PREPROCESSING")
        logger.info(f"{'='*80}")
        logger.info(f"Original shape: {cleaned_flights_df.shape}")

        X = cleaned_flights_df.drop(columns=["ARRIVAL_DELAY"])
        y = cleaned_flights_df["ARRIVAL_DELAY"]

        logger.info(f"Features: {X.shape[1]} | Samples: {len(X):,}")
        logger.info(f"Class distribution:")
        logger.info(
            f"  Class 0 (No Delay): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)"
        )
        logger.info(
            f"  Class 1 (Delay):    {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        del X, y
        collect()

        logger.info(f"✂️  Split: Train={len(X_train):,} | Test={len(X_test):,}")

        logger.info(f"{'='*80}")
        logger.info(f"🔧 FEATURE ENGINEERING - Training Set")
        logger.info(f"{'='*80}")
        X_train_processed = self.preprocess_flights_data(X_train)
        y_train = y_train.loc[X_train_processed.index]

        COLUMNS_TO_REMOVE = [
            col for col in X_train_processed.columns if col not in KEEP_FEATURES
        ]
        X_train_processed = X_train_processed.drop(
            columns=COLUMNS_TO_REMOVE, errors="ignore"
        )

        self.feature_columns = X_train_processed.columns.tolist()
        logger.info(f"✅ {len(self.feature_columns)} features selected for training")

        logger.info(f"{'='*80}")
        logger.info(f"🔧 FEATURE ENGINEERING - Test Set")
        logger.info(f"{'='*80}")
        X_test_processed = self.preprocess_flights_data(X_test)
        y_test = y_test.loc[X_test_processed.index]

        X_test_processed = X_test_processed[self.feature_columns]

        logger.info(f"{'='*80}")
        logger.info(f"✅ FINAL SHAPE OF DATA")
        logger.info(f"{'='*80}")
        logger.info(f"X_train: {X_train_processed.shape} | y_train: {y_train.shape}")
        logger.info(f"X_test:  {X_test_processed.shape} | y_test:  {y_test.shape}")
        logger.info(f"{'='*80}")

        del X_train, X_test
        collect()

        return X_train_processed, X_test_processed, y_train, y_test

    def train(
        self,
        X_train: DataFrame,
        y_train: Series,
        X_test: DataFrame,
        y_test: Series,
        cv_splits: int = 5,
        threshold: float = 0.5,
    ) -> Tuple[GaussianNB, Dict]:
        """
        Trains the Naive Bayes model with a fixed hyperparameter.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target
            cv_splits: NNumber of folds for cross-validation
            threshold: Classification threshold

        Returns:
            Trained model and metrics dictionary
        """
        logger.info(f"{'='*80}")
        logger.info(f"🚀 TRAINING NAIVE BAYES")
        logger.info(f"{'='*80}")
        logger.info(f"Hyperparameter: var_smoothing = {self.var_smoothing}")
        logger.info(f"CV Folds: {cv_splits}")
        logger.info(f"Features: {X_train.shape[1]} | Train samples: {len(X_train):,}")

        nb_model = GaussianNB(var_smoothing=self.var_smoothing)

        logger.info("⏳ Running cross-validation...")
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            nb_model, X_train, y_train, cv=kf, scoring="f1", n_jobs=N_CORES
        )

        cv_mean = mean(cv_scores)
        cv_std = std(cv_scores)

        logger.info(f"📊 Cross-Validation F1-Scores: {cv_scores}")
        logger.info(f"🎯 CV F1-Score: {cv_mean:.4f} (±{cv_std:.4f})")

        start_time = time()
        logger.info("⏳ Training final model on full training set...")
        nb_model.fit(X_train, y_train)
        training_time = time() - start_time

        logger.info(f"✅ Training completed in {training_time:.2f}s")

        y_pred_train = nb_model.predict(X_train)
        y_pred_train_proba = nb_model.predict_proba(X_train)[:, 1]
        y_pred_test_proba = nb_model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_pred_test_proba >= threshold).astype(int)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_test_proba)
        roc_auc = roc_auc_score(y_test, y_pred_test_proba)

        logger.info(f"🔍 Prediction Analysis:")
        logger.info(f"  Custom threshold: {threshold:.4f}")
        logger.info(f"  ROC-AUC:          {roc_auc:.4f}")

        metrics = {
            "train": {
                "Accuracy": accuracy_score(y_train, y_pred_train),
                "Precision": precision_score(y_train, y_pred_train, zero_division=0),
                "Recall": recall_score(y_train, y_pred_train, zero_division=0),
                "F1-Score": f1_score(y_train, y_pred_train, zero_division=0),
                "ROC-AUC": roc_auc_score(y_train, y_pred_train_proba),
            },
            "test": {
                "Accuracy": accuracy_score(y_test, y_pred_test),
                "Precision": precision_score(y_test, y_pred_test, zero_division=0),
                "Recall": recall_score(y_test, y_pred_test, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred_test, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_pred_test_proba),
            },
            "cv_scores": cv_scores.tolist(),
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "var_smoothing": self.var_smoothing,
            "training_time": training_time,
            "threshold": threshold,
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
                "auc": roc_auc,
            },
        }

        self._log_metrics(metrics, y_test, y_pred_test)

        self.model = nb_model
        self.metrics = metrics

        return nb_model, metrics

    def _log_metrics(self, metrics: Dict, y_test: Series, y_pred_test: Series) -> None:
        """
        Logs the training and testing metrics in a formatted table.
        Args:
            metrics: Dictionary with training and testing metrics
            y_test: True labels for the test set
            y_pred_test: Predicted labels for the test set
        """
        logger.info(f"{'='*80}")
        logger.info(f"📈 MODEL PERFORMANCE")
        logger.info(f"{'='*80}")
        logger.info(
            f"{'Metric':<15} {'Train':>15} {'Test':>15} {'Diff':>15} {'Overfit?':>12}"
        )
        logger.info(f"{'-'*80}")

        for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            train_val = metrics["train"][metric]
            test_val = metrics["test"][metric]
            diff = abs(train_val - test_val)
            overfit = "⚠️ Yes" if diff > 0.1 else "✅ No"
            logger.info(
                f"{metric:<15} {train_val:>15.4f} {test_val:>15.4f} {diff:>15.4f} {overfit:>12}"
            )

        logger.info(f"{'-'*80}")
        logger.info(
            f"{'CV F1-Score':<15} {metrics['cv_mean']:>15.4f} (±{metrics['cv_std']:.4f})"
        )
        logger.info(f"{'='*80}")

        cm = confusion_matrix(y_test, y_pred_test)
        logger.info(f"📊 Confusion Matrix (Test):")
        logger.info(f"                Predicted")
        logger.info(f"                No Delay  |  Delay")
        logger.info(f"Actual No Delay    {cm[0,0]:>6}  |  {cm[0,1]:>6}")
        logger.info(f"Actual Delay       {cm[1,0]:>6}  |  {cm[1,1]:>6}")

        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        logger.info(f"📊 Additional Metrics:")
        logger.info(f"  True Positives:  {tp:>6} (Correctly predicted delays)")
        logger.info(f"  True Negatives:  {tn:>6} (Correctly predicted no delays)")
        logger.info(f"  False Positives: {fp:>6} (False alarms)")
        logger.info(f"  False Negatives: {fn:>6} (Missed delays)")
        logger.info(f"  Sensitivity:     {sensitivity:>6.4f} (True Positive Rate)")
        logger.info(f"  Specificity:     {specificity:>6.4f} (True Negative Rate)")
        logger.info(f"{'='*80}")

    def save_model(self) -> None:
        """
        Saves the trained model and metadata to disk.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        logger.info(f"💾 Saving model...")

        dump(self.model, self.model_path)
        logger.info(f"✅ Model saved at: {self.model_path}")

        metadata = {
            "model_name": self.model_name,
            "feature_columns": self.feature_columns,
            "metrics": {
                "train": self.metrics["train"],
                "test": self.metrics["test"],
                "cv_mean": self.metrics["cv_mean"],
                "cv_std": self.metrics["cv_std"],
                "threshold": self.metrics["threshold"],
            },
            "var_smoothing": self.metrics["var_smoothing"],
            "training_time": self.metrics["training_time"],
            "trained_at": time(),
        }

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"✅ Metadata saved at: {self.metadata_path}")

    def train_and_save(self, data_path: Optional[str] = None) -> None:
        """
        Complete pipeline: loads data, trains, and saves model.

        Args:
            data_path: Path to cleaned data (uses default if None)
        """
        logger.info("=" * 80)
        logger.info(f"🚀 NAIVE BAYES TRAINING PIPELINE: {self.model_name}")
        logger.info("=" * 80)

        data_path = data_path or CLEANED_DATA_PATH
        logger.info(f"📂 Loading data from: {data_path}")
        cleaned_flights_df = read_csv(data_path)

        # Define target (>= 15 min = delay)
        cleaned_flights_df["ARRIVAL_DELAY"] = (
            cleaned_flights_df["ARRIVAL_DELAY"] >= 15
        ).astype(int)

        logger.info(f"✅ Loaded {len(cleaned_flights_df):,} flights")
        logger.info(f"📊 Target: ARRIVAL_DELAY >= 15 min = 1 (Delay)")
        logger.info(f"   Class 0: {(cleaned_flights_df['ARRIVAL_DELAY']==0).sum():,}")
        logger.info(f"   Class 1: {(cleaned_flights_df['ARRIVAL_DELAY']==1).sum():,}")

        X_train, X_test, y_train, y_test = self.split_and_preprocess_data(
            cleaned_flights_df
        )

        self.train(X_train, y_train, X_test, y_test)

        self.save_model()

        logger.info("=" * 80)
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info("=" * 80)


class NaiveBayesPredictor:
    """
    Class to load trained model and make predictions.
    """

    def __init__(self, model_name: str = "naive_bayes_model") -> None:
        """
        Initializes the predictor.

        Args:
            model_name: Name of the model to load
        """
        self.model_name = model_name
        self.model_path = MODELS_DIR / f"{model_name}.pkl"
        self.metadata_path = MODELS_DIR / f"{model_name}_metadata.json"

        self.model: Optional[GaussianNB] = None
        self.metadata: Optional[Dict] = None
        self.feature_columns: Optional[list] = None
        self.fe = FeatureEngineer()

    def load_model(self):
        """Loads model and metadata from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Run training first using NaiveBayesTrainer."
            )

        logger.info(f"📂 Loading model from: {self.model_path}")
        self.model = load(self.model_path)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.feature_columns = self.metadata.get("feature_columns", [])
            logger.info(f"✅ Model loaded with {len(self.feature_columns)} features")
        else:
            logger.warning(f"⚠️ Metadata not found: {self.metadata_path}")

    def preprocess_for_prediction(self, flights_df: DataFrame) -> DataFrame:
        """
        Preprocess data for prediction (same pipeline as training).

        Args:
            flights_df: DataFrame with raw data

        Returns:
            Processed DataFrame
        """
        self.fe.set_dataframe(flights_df.copy())

        COLUMNS_DROP = [
            "DAY_OF_WEEK",
            "DAY_OF_YEAR",
            "SCHEDULED_DEPARTURE_MINUTES",
            "SCHEDULED_ARRIVAL_MINUTES",
            "SCHEDULED_DEPARTURE",
            "SCHEDULED_ARRIVAL",
            "FLIGHT_NUMBER",
            "DISTANCE",
            "SCHEDULED_TIME",
        ]

        (
            self.fe.encoding_cyclic_feature("DAY_OF_WEEK", period=7)
            .encoding_cyclic_feature("DAY_OF_YEAR", period=365)
            .process_time_hhmm("SCHEDULED_DEPARTURE", keep_original=True)
            .process_time_hhmm("SCHEDULED_ARRIVAL", keep_original=True)
            .create_weekend_feature()
            .create_season()
            .normalize_column("DISTANCE")
            .normalize_column("SCHEDULED_TIME")
            .create_time_periods("SCHEDULED_DEPARTURE")
            .create_time_periods("SCHEDULED_ARRIVAL")
            .create_distance_bins()
            .one_hot_encode("AIRLINE")
            .one_hot_encode("SEASON")
            .one_hot_encode("DISTANCE_CATEGORY")
            .one_hot_encode("SCHEDULED_DEPARTURE_PERIOD")
            .one_hot_encode("SCHEDULED_ARRIVAL_PERIOD")
            .frequency_encode("FLIGHT_NUMBER")
            .rename_columns_uppercase()
            .drop_columns(COLUMNS_DROP)
            .drop_NaN_rows()
            .set_bools_to_ints(bool_cols="all")
        )

        processed_df = self.fe.get_dataframe()

        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(processed_df.columns)
            if missing_cols:
                logger.info(
                    f"➕ Adding {len(missing_cols)} missing columns with value 0"
                )
                for col in missing_cols:
                    processed_df[col] = 0

            extra_cols = set(processed_df.columns) - set(self.feature_columns)
            if extra_cols:
                logger.info(
                    f"➖ Removing {len(extra_cols)} extra columns not used in training"
                )
                processed_df = processed_df.drop(columns=list(extra_cols))

            processed_df = processed_df[self.feature_columns]

        return processed_df

    def predict(self, flights_df: DataFrame, return_proba: bool = False) -> Series:
        """
        Make predictions for new data.

        Args:
            flights_df: DataFrame with raw data
            return_proba: If True, return probabilities instead of classes

        Returns:
            Series with predictions (0/1) or probabilities
        """
        if self.model is None:
            self.load_model()

        logger.info(f"🔮 Making predictions for {len(flights_df)} flights...")

        X_processed = self.preprocess_for_prediction(flights_df)

        if return_proba:
            predictions = self.model.predict_proba(X_processed)[:, 1]
            logger.info(f"✅ Probabilities calculated")
        else:
            predictions = self.model.predict(X_processed)
            delays = (predictions == 1).sum()
            logger.info(
                f"✅ Predictions: {delays} delays out of {len(predictions)} flights ({delays/len(predictions)*100:.1f}%)"
            )

        return Series(predictions, index=X_processed.index)

    def predict_single_flight(self, flight_data: dict) -> Tuple[int, float]:
        """
        Predict delay for a single flight.

        Args:
            flight_data: Dictionary with flight data

        Returns:
            (prediction, probability)
        """
        df = DataFrame([flight_data])
        proba = self.predict(df, return_proba=True).iloc[0]
        prediction = 1 if proba >= 0.5 else 0

        logger.info(f"✈️  Flight: {flight_data.get('FLIGHT_NUMBER', 'N/A')}")
        logger.info(f"   Prediction: {'DELAY' if prediction == 1 else 'ON TIME'}")
        logger.info(f"   Delay probability: {proba:.2%}")

        return prediction, proba
