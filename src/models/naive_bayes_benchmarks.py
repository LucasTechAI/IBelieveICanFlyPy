from typing import Tuple
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from logging import getLogger, basicConfig, INFO
from pandas import read_csv, DataFrame, Series
from sklearn.naive_bayes import GaussianNB
from multiprocessing import cpu_count
from lightgbm import LGBMClassifier
from numpy import mean, median
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from pathlib import Path
from gc import collect
from time import time
from json import dump
from sys import path


logger = getLogger(__name__)
basicConfig(
    level=INFO, format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)



current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
path.append(str(root_dir))
from src.etl.feature_engineering import FeatureEngineer

FEATURES_IMPORTANCE_PATH = root_dir / "data/models/benchmarks/lightgbm_feature_importance_nb.csv"
CLEANED_DATA_PATH = root_dir / "data/interim/cleaned_flights.csv"

N_CORES = max(1, cpu_count() - 2)
logger.info(f"🖥️  CPUs available: {cpu_count()} | Using: {N_CORES} cores")

fe = FeatureEngineer()


def preprocess_flights_data(cleaned_flights_df: DataFrame) -> DataFrame:
    """
    Apply feature engineering to the flights data.
    Args:
        cleaned_flights_df (DataFrame): Cleaned flights data.
    Returns:
        DataFrame: Processed flights data with feature engineering applied.
    """
    logger
    fe.set_dataframe(cleaned_flights_df.copy())
    COLUMNS_DROP = [
        "DAY_OF_WEEK",
        "DAY_OF_YEAR",
        "SCHEDULED_DEPARTURE_MINUTES",
        "SCHEDULED_ARRIVAL_MINUTES",
        "SCHEDULED_DEPARTURE",
        "SCHEDULED_ARRIVAL",
        "FLIGHT_NUMBER",
        "DISTANCE",
        "SCHEDULED_TIME"
    ]

    (
        fe.encoding_cyclic_feature("DAY_OF_WEEK", period=7)
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

    del cleaned_flights_df
    collect()

    return fe.get_dataframe()


def test_1_full_dataset(X_train: DataFrame, y_train: Series, X_test: DataFrame, y_test: Series) -> Tuple[GaussianNB, dict, dict]:
    """
    Test1: Train Naive Bayes on the full dataset
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        X_test (DataFrame): Testing features
        y_test (Series): Testing target
    Returns:

        Tuple[GaussianNB, dict, dict]: Trained model, training metrics, testing metrics
    """
    
    logger.info(f"{'='*80}")
    logger.info(f"📊 TESTE 1: NAIVE BAYES - ALL DATABASE")
    logger.info(f"{'='*80}")
    
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Samples train: {len(X_train):,}")
    logger.info(f"Samples test: {len(X_test):,}")
    
    return train_naive_bayes(X_train, y_train, X_test, y_test, 
                            test_name="Test 1 - All Database")



def test_2_top_features_lightgbm(X_train, y_train, X_test, y_test, top_n=15) -> Tuple[GaussianNB, dict, dict]:
    """
    Test 2: Select top N features based on LightGBM feature importance
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        X_test (DataFrame): Testing features
        y_test (Series): Testing target
        top_n (int): Number of top features to select
    Returns:
        Tuple[GaussianNB, dict, dict]: Trained model, training metrics, testing metrics
    """
    logger.info(f"{'='*80}")
    logger.info(f"📊 TEST 2: TOP {top_n} FEATURES - LIGHTGBM FEATURE IMPORTANCE")
    logger.info(f"{'='*80}")
    
    logger.info(f"Extracting feature importances using LightGBM...")
    
    lgbm = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=15,
        random_state=42,
        verbose=1,
        n_jobs=N_CORES,
        force_col_wise=True
    )
    
    start_time = time()
    lgbm.fit(X_train, y_train)
    lgbm_time = time() - start_time

    y_pred = lgbm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall_score_val = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    precision_score_val = precision_score(y_test, y_pred)
    f1_score_val = f1_score(y_test, y_pred)

    logger.info(f"💡 LightGBM Performance (using all features):"
          f"  Accuracy : {accuracy:.4f}"
          f"  Precision: {precision_score_val:.4f}"
          f"  Recall   : {recall_score_val:.4f}"
          f"  ROC-AUC  : {roc:.4f}"
          f"  F1-Score : {f1_score_val:.4f}")
    
    feature_importance = DataFrame({
        'feature': X_train.columns,
        'importance': lgbm.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"✅ LightGBM trained in {lgbm_time:.2f}s")
    logger.info(f"🏆 Top {top_n} Features by Importance:")
    logger.info(feature_importance.head(top_n).to_string(index=False))

    feature_importance.to_csv(FEATURES_IMPORTANCE_PATH, index=False)
    
    top_features = feature_importance.head(top_n)['feature'].tolist()
    
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    
    logger.info(f"📊 Reduced Dataset:")
    logger.info(f"  Features: {len(top_features)}")
    logger.info(f"  Samples train: {len(X_train_top):,}")
    
    del lgbm
    collect()
    
    return train_naive_bayes(X_train_top, y_train, X_test_top, y_test,
                            test_name=f"Test 2 - Top {top_n} Features (LightGBM)")


def test_3_high_correlation(X_train: DataFrame, y_train: Series, X_test: DataFrame, y_test: Series, threshold:int=0.6) -> Tuple[GaussianNB, dict, dict]:
    """
    Test 3: Select features with correlation > threshold
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        X_test (DataFrame): Testing features
        y_test (Series): Testing target
        threshold (int): Correlation threshold
    Returns:
        Tuple[GaussianNB, dict, dict]: Trained model, training metrics, testing metrics
    """
    logger.info(f"{'='*80}")
    logger.info(f"📊 TEST 3: FEATURES HIGH CORRELATION > |{threshold}|")
    logger.info(f"{'='*80}")
    
    logger.info(f"🔍 Calculating correlations with target...")
    
    correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    high_corr_features = correlations[correlations > threshold].index.tolist()
    
    logger.info(f"📊 Features with correlation > |{threshold}|: {len(high_corr_features)}")
    
    if len(high_corr_features) == 0:
        logger.warning(f"⚠️  No features with correlation > |{threshold}|")
        logger.info(f"📉 Top 15 correlations found:")
        logger.info(correlations.head(15))
        logger.info(f"💡 Using top 15 features instead...")
        high_corr_features = correlations.head(15).index.tolist()
    else:
        logger.info(f"🏆 Selected features:")
        for feat in high_corr_features:
            logger.info(f"  {feat}: {correlations[feat]:.4f}")
    
    X_train_corr = X_train[high_corr_features]
    X_test_corr = X_test[high_corr_features]
    
    return train_naive_bayes(X_train_corr, y_train, X_test_corr, y_test,
                            test_name=f"Test 3 - Correlation > |{threshold}|")


def test_4_smote_full(X_train: DataFrame, y_train: Series, X_test: DataFrame, y_test: Series) -> Tuple[GaussianNB, dict, dict]:
    """
    Test 4: SMOTE with sampling_strategy='auto' (full balancing 1:1)
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        X_test (DataFrame): Testing features
        y_test (Series): Testing target
    Returns:
        Tuple[GaussianNB, dict, dict]: Trained model, training metrics, testing metrics
    """
    logger.info(f"{'='*80}")
    logger.info(f"📊 TEST 4: SMOTE - FULL BALANCING (1:1)")
    logger.info(f"{'='*80}")
    
    logger.info(f"⚖️  Applying SMOTE...")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    
    original_counts = Series(y_train).value_counts()
    logger.info(f"📊 Original distribution:")
    logger.info(f"  Class 0: {original_counts[0]:,}")
    logger.info(f"  Class 1: {original_counts[1]:,}")
    logger.info(f"  Ratio: {original_counts[0]/original_counts[1]:.2f}:1")
    
    start_time = time()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    smote_time = time() - start_time
    
    balanced_counts = Series(y_train_balanced).value_counts()
    logger.info(f"📊 Balanced distribution:")
    logger.info(f"  Class 0: {balanced_counts[0]:,}")
    logger.info(f"  Class 1: {balanced_counts[1]:,}")
    logger.info(f"  Ratio: {balanced_counts[0]/balanced_counts[1]:.2f}:1")
    logger.info(f"  Time: {smote_time:.2f}s")
    
    return train_naive_bayes(X_train_balanced, y_train_balanced, X_test, y_test,
                            test_name="Test 4 - SMOTE (1:1)")


def test_5_smote_30_percent(X_train: DataFrame, y_train: Series, X_test: DataFrame, y_test: Series) -> Tuple[GaussianNB, dict, dict]:
    """
    Test 5: SMOTE with sampling_strategy=0.3 (30% of majority class)
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        X_test (DataFrame): Testing features
        y_test (Series): Testing target
    Returns:
        Tuple[GaussianNB, dict, dict]: Trained model, training metrics, testing metrics
    """
    logger.info(f"{'='*80}")
    logger.info(f"📊 TEST 5: SMOTE - 30% OF MAJORITY CLASS")
    logger.info(f"{'='*80}")
    
    logger.info(f"⚖️  Applying SMOTE (30%)...")
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    
    original_counts = Series(y_train).value_counts()
    logger.info(f"📊 Original distribution:")
    logger.info(f"  Class 0: {original_counts[0]:,}")
    logger.info(f"  Class 1: {original_counts[1]:,}")
    logger.info(f"  Ratio: {original_counts[0]/original_counts[1]:.2f}:1")
    
    start_time = time()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    smote_time = time() - start_time
    
    balanced_counts = Series(y_train_balanced).value_counts()
    logger.info(f"📊 Balanced distribution:")
    logger.info(f"  Class 0: {balanced_counts[0]:,}")
    logger.info(f"  Class 1: {balanced_counts[1]:,}")
    logger.info(f"  Ratio: {balanced_counts[0]/balanced_counts[1]:.2f}:1")
    logger.info(f"  Time: {smote_time:.2f}s")
    
    return train_naive_bayes(X_train_balanced, y_train_balanced, X_test, y_test,
                            test_name="Test 5 - SMOTE (30%)")


def test_6_random_oversample(X_train: DataFrame, y_train: Series, X_test: DataFrame, y_test: Series) -> Tuple[GaussianNB, dict, dict]:
    """
    Test 6: Random Oversampling to equalize classes
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        X_test (DataFrame): Testing features
        y_test (Series): Testing target
    Returns:
        Tuple[GaussianNB, dict, dict]: Trained model, training metrics, testing metrics
    """
    logger.info(f"{'='*80}")
    logger.info(f"📊 TEST 6: RANDOM OVERSAMPLING - EQUALIZE CLASSES")
    logger.info(f"{'='*80}")
    
    logger.info(f"⚖️  Applying Random Oversampling...")
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    
    original_counts = Series(y_train).value_counts()
    logger.info(f"📊 Original distribution:")
    logger.info(f"  Class 0: {original_counts[0]:,}")
    logger.info(f"  Class 1: {original_counts[1]:,}")
    logger.info(f"  Ratio: {original_counts[0]/original_counts[1]:.2f}:1")
    
    start_time = time()
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
    ros_time = time() - start_time
    
    balanced_counts = Series(y_train_balanced).value_counts()
    logger.info(f"📊 Balanced distribution:")
    logger.info(f"  Class 0: {balanced_counts[0]:,}")
    logger.info(f"  Class 1: {balanced_counts[1]:,}")
    logger.info(f"  Ratio: {balanced_counts[0]/balanced_counts[1]:.2f}:1")
    logger.info(f"  Time: {ros_time:.2f}s")
    logger.info(f"💡 MMethod: Random duplication of minority class samples")
    
    return train_naive_bayes(X_train_balanced, y_train_balanced, X_test, y_test,
                            test_name="Test 6 - Random Oversampling")


def train_naive_bayes(X_train: DataFrame, y_train: Series, X_test: DataFrame, y_test: Series, test_name: str="Naive Bayes", 
                     n_iter: int=9, cv_splits: int=5, threshold: float=0.5) -> Tuple[GaussianNB, dict, dict]:
    """
    Train Naive Bayes with RandomizedSearchCV
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training target
        X_test (DataFrame): Testing features
        y_test (Series): Testing target
        test_name (str): Name of the test
        n_iter (int): Number of random iterations for hyperparameter search
        cv_splits (int): Number of cross-validation splits
        threshold (float): Classification threshold
    """
    logger.info(f"{'='*80}")
    logger.info(f"🚀 TRAINING: {test_name}")
    logger.info(f"{'='*80}")
    logger.info(f"CV Folds: {cv_splits} | Random iterations: {n_iter}")
    
    naive_bayes_params = {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    }

    nb_model = GaussianNB()
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=nb_model,
        param_distributions=naive_bayes_params,
        n_iter=min(n_iter, len(naive_bayes_params["var_smoothing"])),
        scoring="f1",
        cv=kf,
        verbose=1,
        random_state=42,
        n_jobs=2,
        return_train_score=True,
    )

    start_time = time()

    logger.info("⏳ Training model...")
    random_search.fit(X_train, y_train)
    training_time = time() - start_time

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    logger.info(f"✅ Training completed - {training_time:.2f}s")
    logger.info(f"🏆 Best Hyperparameters:")
    for param, value in sorted(best_params.items()):
        logger.info(f"  {param:25} : {value}")
    logger.info(f"🎯 Best CV F1-Score: {best_score:.4f}")

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    y_pred_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]


    logger.info(f"🔍 Threshold Analysis: {threshold}" )
    
    y_pred_test = (y_pred_test_proba >= threshold).astype(int)
    fbr, tpr, thresholds = roc_curve(y_test, y_pred_test_proba)
    roc_auc = roc_auc_score(y_test, y_pred_test_proba)

    logger.info(f"Median threshold (test): {median(thresholds):.4f} | AUC: {roc_auc:.4f}")
    logger.info(f"Mean threshold (test): {mean(thresholds):.4f}")
    logger.info(f"Custom threshold (test): {threshold:.4f}")


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
        "cv_score": best_score,
        "best_params": best_params,
        "training_time": training_time,
        "columns_used": X_train.columns.tolist(),
    }

    logger.info(f"{'='*80}")
    logger.info(f"📈 MODEL PERFORMANCE")
    logger.info(f"{'='*80}")   
    logger.info(f"{'Metric':<15} {'Train':>15} {'Test':>15} {'Diff':>15} {'Overfit?':>12}")
    logger.info(f"{'-'*80}")
    
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        train_val = metrics["train"][metric]
        test_val = metrics["test"][metric]
        diff = abs(train_val - test_val)
        overfit = "⚠️ Yes" if diff > 0.1 else "✅ No"
        logger.info(f"{metric:<15} {train_val:>15.4f} {test_val:>15.4f} {diff:>15.4f} {overfit:>12}")
    
    logger.info(f"{'-'*80}")
    logger.info(f"{'CV F1-Score':<15} {metrics['cv_score']:>15.4f}")
    logger.info(f"{'='*80}")

    logger.info(f"📊 Confusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_pred_test)
    logger.info(f"                Predito")
    logger.info(f"                Sem Atraso  |  Com Atraso")
    logger.info(f"Real Sem Atraso    {cm[0,0]:>6}  |  {cm[0,1]:>6}")
    logger.info(f"Real Com Atraso    {cm[1,0]:>6}  |  {cm[1,1]:>6}")
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    logger.info(f"📊 Additional Metrics:")
    logger.info(f"  True Positives:  {tp:>6} (Atrasos corretamente previstos)")
    logger.info(f"  True Negatives:  {tn:>6} (Sem atraso corretamente previsto)")
    logger.info(f"  False Positives: {fp:>6} (Falso alarme de atraso)")
    logger.info(f"  False Negatives: {fn:>6} (Atrasos não detectados)")
    logger.info(f"  Sensitivity:     {sensitivity:>6.4f} (Taxa de Verdadeiros Positivos)")
    logger.info(f"  Specificity:     {specificity:>6.4f} (Taxa de Verdadeiros Negativos)")

    logger.info(f"{'='*80}")

    roc_data = {
        "thresholds": thresholds.tolist(),
        "fpr": fbr.tolist(),
        "tpr": tpr.tolist(),
        "auc": roc_auc
    }

    del random_search
    collect()

    return best_model, metrics, roc_data



def split_data(cleaned_flights_df: DataFrame, test_size: float = 0.2, random_state:int=42):
    """
    Split the cleaned flights data into training and testing sets.
    Args:
        cleaned_flights_df (DataFrame): Cleaned flights data.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    Returns:
        Tuple[DataFrame, DataFrame, Series, Series]: X_train, X_test, y_train, y_test
    """
    logger.info(f"{'='*80}")
    logger.info(f"📊 DATA SPLIT")
    logger.info(f"{'='*80}")
    logger.info(f"Original shape: {cleaned_flights_df.shape}")

    X = cleaned_flights_df.drop(columns=["ARRIVAL_DELAY"])
    y = cleaned_flights_df["ARRIVAL_DELAY"]

    logger.info(f"Features: {X.shape[1]} | Samples: {len(X):,}")
    logger.info(f"Class distribution:")
    logger.info(f"  Class 0: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
    logger.info(f"  Class 1: {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    del X, y
    collect()

    logger.info(f"✂️  Split: Train={len(X_train):,} | Test={len(X_test):,}")

    logger.info(f"{'='*80}")
    logger.info(f"🔧 FEATURE ENGINEERING - Training Set")
    logger.info(f"{'='*80}")
    X_train_processed = preprocess_flights_data(X_train)
    logger.info(f"Features Columns: {X_train_processed.columns.tolist()}")
    y_train = y_train.loc[X_train_processed.index]

    logger.info(f"{'='*80}")
    logger.info(f"🔧 FEATURE ENGINEERING - Test Set")
    logger.info(f"{'='*80}")
    X_test_processed = preprocess_flights_data(X_test)
    y_test = y_test.loc[X_test_processed.index]

    logger.info(f"{'='*80}")
    logger.info(f"✅ FINAL SHAPE OF DATA")
    logger.info(f"{'='*80}")
    logger.info(f"X_train: {X_train_processed.shape} | y_train: {y_train.shape}")
    logger.info(f"X_test:  {X_test_processed.shape} | y_test:  {y_test.shape}")
    logger.info(f"{'='*80}")

    del X_train, X_test
    collect()

    return X_train_processed, X_test_processed, y_train, y_test

def main():
    benchmark_results = {}

    logger.info("=" * 80)
    logger.info("🚀 BENCHMARK: NAIVE BAYES - 6 TESTS")
    logger.info("=" * 80)
    logger.info("📂 Load Database")
    
    cleaned_flights_df = read_csv(CLEANED_DATA_PATH)
    cleaned_flights_df["ARRIVAL_DELAY"] = (
        cleaned_flights_df["ARRIVAL_DELAY"] >= 15
    ).astype(int)
    
    logger.info(f"✅ Loaded {len(cleaned_flights_df):,} flights")
    logger.info(f"📊 Target: ARRIVAL_DELAY >= 15 min = 1 (Delay)")
    logger.info(f"   Class 0: {(cleaned_flights_df['ARRIVAL_DELAY']==0).sum():,}")
    logger.info(f"   Class 1: {(cleaned_flights_df['ARRIVAL_DELAY']==1).sum():,}")

    X_train, X_test, y_train, y_test = split_data(
        cleaned_flights_df, test_size=0.3, random_state=42
    )

    
    model_1, metrics_1, roc_1 = test_1_full_dataset(
        X_train.copy(), y_train.copy(), 
        X_test.copy(), y_test.copy()
    )
    benchmark_results["teste_1_base_completa"] = {
        "description": "All dataset - Baseline",
        "metrics": metrics_1,
        "roc_curve": roc_1
    }
    del model_1; collect()

    
    model_2, metrics_2, roc_2 = test_2_top_features_lightgbm(
        X_train.copy(), y_train.copy(), 
        X_test.copy(), y_test.copy(),
        top_n=15
    )
    benchmark_results["teste_2_top_15_lightgbm"] = {
        "description": "Top 15 features - LightGBM importance",
        "metrics": metrics_2,
        "roc_curve": roc_2
    }
    del model_2; collect()

    
    model_3, metrics_3, roc_3 = test_3_high_correlation(
        X_train.copy(), y_train.copy(), 
        X_test.copy(), y_test.copy(),
        threshold=0.6
    )
    benchmark_results["teste_3_correlacao"] = {
        "description": "Features com correlação > |0.6|",
        "metrics": metrics_3,
        "roc_curve": roc_3
    }
    del model_3; collect()

    
    model_4, metrics_4, roc_4 = test_4_smote_full(
        X_train.copy(), y_train.copy(), 
        X_test.copy(), y_test.copy()
    )
    benchmark_results["teste_4_smote_full"] = {
        "description": "SMOTE - Balanceamento completo 1:1",
        "metrics": metrics_4,
        "roc_curve": roc_4
    }
    del model_4; collect()

    
    model_5, metrics_5, roc_5 = test_5_smote_30_percent(
        X_train.copy(), y_train.copy(), 
        X_test.copy(), y_test.copy()
    )
    benchmark_results["teste_5_smote_30"] = {
        "description": "SMOTE - 30% da classe majoritária",
        "metrics": metrics_5,
        "roc_curve": roc_5
    }
    del model_5; collect()

    
    model_6, metrics_6, roc_6 = test_6_random_oversample(
        X_train.copy(), y_train.copy(), 
        X_test.copy(), y_test.copy()
    )
    benchmark_results["teste_6_random_oversample"] = {
        "description": "Random Oversampling - Igualar classes",
        "metrics": metrics_6,
        "roc_curve": roc_6
    }
    del model_6; collect()

    logger.info("=" * 80)
    logger.info("📊 FINAL COMPARISON - 6 NAIVE BAYES TESTS")
    logger.info("=" * 80)
    
    comparison_df = DataFrame([
        {
            "Teste": results["description"],
            "F1-Score": results["metrics"]["test"]["F1-Score"],
            "Recall": results["metrics"]["test"]["Recall"],
            "Precision": results["metrics"]["test"]["Precision"],
            "ROC-AUC": results["metrics"]["test"]["ROC-AUC"],
            "Accuracy": results["metrics"]["test"]["Accuracy"],
            "Tempo (s)": results["metrics"]["training_time"],
        }
        for _, results in benchmark_results.items()
    ])
    
    comparison_df = comparison_df.sort_values("F1-Score", ascending=False)
    comparison_df.insert(0, 'Rank', range(1, len(comparison_df) + 1))
    
    logger.info("🏆 Ranking by F1-Score (Test):")
    logger.info("=" * 40)
    logger.info(comparison_df.to_string(index=False))
    logger.info("=" * 40)
    
    best_test = comparison_df.iloc[0]
    logger.info(f"🥇 MELHOR TESTE: {best_test['Teste']}")
    logger.info(f"   F1-Score: {best_test['F1-Score']:.4f}")
    logger.info(f"   Recall:   {best_test['Recall']:.4f}")
    logger.info(f"   Precision: {best_test['Precision']:.4f}")
    logger.info(f"   ROC-AUC:  {best_test['ROC-AUC']:.4f}")
    
    logger.info(f"📊 TRADE-OFF ANALYSIS:")
    logger.info(f"   Best F1-Score:  {comparison_df.iloc[0]['Teste']} ({comparison_df.iloc[0]['F1-Score']:.4f})")
    logger.info(f"   Best Recall:    {comparison_df.loc[comparison_df['Recall'].idxmax()]['Teste']} ({comparison_df['Recall'].max():.4f})")
    logger.info(f"   Best Precision: {comparison_df.loc[comparison_df['Precision'].idxmax()]['Teste']} ({comparison_df['Precision'].max():.4f})")
    logger.info(f"   Best ROC-AUC:   {comparison_df.loc[comparison_df['ROC-AUC'].idxmax()]['Teste']} ({comparison_df['ROC-AUC'].max():.4f})")
    logger.info(f"   Fastest:        {comparison_df.loc[comparison_df['Tempo (s)'].idxmin()]['Teste']} ({comparison_df['Tempo (s)'].min():.2f}s)")
    
    output_path = root_dir / "data/models/benchmarks/naive_bayes.json"
    with open(output_path, "w") as f:
        dump(benchmark_results, f, indent=4, default=str)

    comparison_csv_path = root_dir / "data/models/benchmarks/naive_bayes.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)

    logger.info(f"💾 Results:")
    logger.info(f"   JSON: {output_path}")
    logger.info(f"   CSV:  {comparison_csv_path}")
    logger.info("=" * 80)
    logger.info("✅ ALL DONE!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
    