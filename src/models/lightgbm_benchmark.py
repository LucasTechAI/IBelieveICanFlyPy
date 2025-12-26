from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from logging import getLogger, basicConfig, INFO
from multiprocessing import cpu_count
from pandas import DataFrame, Series
from lightgbm import LGBMClassifier
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    make_scorer
)
from pandas import read_csv
from pathlib import Path
from gc import collect
from time import time
from json import dump
from sys import path
import numpy as np

logger = getLogger(__name__)
basicConfig(
    level=INFO, format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)

N_CORES = max(1, cpu_count() - 2)

current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
path.append(str(root_dir))
from src.etl.feature_engineering import FeatureEngineer


def reduce_memory_usage(df: DataFrame, verbose: bool = True) -> DataFrame:
    """
    Reduce memory usage of a DataFrame by downcasting numeric types
    Args:
        df (DataFrame): Input DataFrame
        verbose (bool): Whether to log memory reduction info
    Returns:
        DataFrame: DataFrame with reduced memory usage
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        logger.info(f'💾 Memory reduced from {start_mem:.2f}MB to {end_mem:.2f}MB '
                   f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def lgbm_random_grid_search_optimized(
    X_train: DataFrame, 
    y_train: Series, 
    X_test: DataFrame, 
    y_test: Series,
    cv_splits: int = 5,
) -> Tuple[LGBMClassifier, Dict, Dict]:
    """
    Perform LightGBM Grid Search with memory optimization
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training labels
        X_test (DataFrame): Test features
        y_test (Series): Test labels
        cv_splits (int): Number of CV splits
        fast_mode (bool): Whether to use a reduced grid for faster execution
    Returns:
        Tuple[LGBMClassifier, Dict, Dict]: Best model, metrics, and grid results
    """
    logger.info(f"{'='*80}")
    logger.info(f"🚀 LIGHTGBM GRID SEARCH")
    logger.info(f"{'='*80}")
    
    X_train = reduce_memory_usage(X_train, verbose=True)
    X_test = reduce_memory_usage(X_test, verbose=True)
    y_train = y_train.astype(np.int8)
    y_test = y_test.astype(np.int8)
    
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Samples train: {len(X_train):,}")
    logger.info(f"Samples test: {len(X_test):,}")
    logger.info(f"CV Folds: {cv_splits}")
    logger.info(f"Cores: {N_CORES}")
    
    param_grid = {
        'n_estimators': [800,  1500, 2000],
        'learning_rate': [0.05, 0.1],
        'max_depth': [10, 15],
        'num_leaves': [31, 63],
        'min_child_samples': [30],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'reg_lambda': [0, 0.5]
    }
    
    lgbm = LGBMClassifier(
        random_state=42,
        n_jobs=1,
        force_col_wise=True,
        verbose=-1,
        class_weight='balanced',
        max_bin=255, 
        min_data_in_bin=3
    )
    
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, zero_division=0)
    
    n_iter = 5
    grid_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=kf,
        scoring=f1_scorer,
        n_jobs=N_CORES,
        verbose=2,
        return_train_score=False,
        error_score='raise',
        pre_dispatch='2*n_jobs',
        random_state=42 
    )
    
    logger.info(f"⏳ Starting Grid Search...")
    start_time = time()
    
    try:
        grid_search.fit(X_train, y_train)
        training_time = time() - start_time
        logger.info(f"✅ Grid Search completed in {training_time/60:.2f} minutes")
    except Exception as e:
        logger.error(f"❌ Error in Grid Search: {e}")
        raise
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"{'='*80}")
    logger.info(f"🏆 BEST HYPERPARAMETERS")
    logger.info(f"{'='*80}")
    for param, value in sorted(best_params.items()):
        logger.info(f"  {param:25} : {value}")
    logger.info(f"🎯 Best CV F1-Score: {best_score:.4f}")
    logger.info("🔮 Generating predictions...")
    
    y_pred_train = best_model.predict(X_train)
    y_pred_train_proba = best_model.predict_proba(X_train)[:, 1]
    
    y_pred_test = best_model.predict(X_test)
    y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "train": {
            "Accuracy": float(accuracy_score(y_train, y_pred_train)),
            "Precision": float(precision_score(y_train, y_pred_train, zero_division=0)),
            "Recall": float(recall_score(y_train, y_pred_train, zero_division=0)),
            "F1-Score": float(f1_score(y_train, y_pred_train, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_train, y_pred_train_proba)),
        },
        "test": {
            "Accuracy": float(accuracy_score(y_test, y_pred_test)),
            "Precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
            "F1-Score": float(f1_score(y_test, y_pred_test, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, y_pred_test_proba)),
        },
        "cv_score": float(best_score),
        "best_params": best_params,
        "training_time": float(training_time),
        "n_combinations": len(grid_search.cv_results_['params']),
    }
    
    del y_pred_train, y_pred_train_proba
    collect()
    
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
    logger.info(f"  True Positives:  {tp:>6}")
    logger.info(f"  True Negatives:  {tn:>6}")
    logger.info(f"  False Positives: {fp:>6}")
    logger.info(f"  False Negatives: {fn:>6}")
    logger.info(f"  Sensitivity:     {sensitivity:>6.4f}")
    logger.info(f"  Specificity:     {specificity:>6.4f}")
    
    feature_importance = DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    logger.info(f"{'='*80}")
    logger.info(f"🔝 TOP 15 FEATURES BY IMPORTANCE")
    logger.info(f"{'='*80}")
    logger.info(feature_importance.to_string(index=False))
    
    grid_results = {
        "best_params": best_params,
        "best_score": float(best_score),
        "feature_importance": feature_importance.to_dict('records')
    }
    
    del grid_search
    collect()
    
    logger.info(f"{'='*80}")
    
    return best_model, metrics, grid_results


def preprocess_flights_data(cleaned_flights_df: DataFrame) -> DataFrame:
    """
    Apply feature engineering to the flights data with memory optimization
    Args:
        cleaned_flights_df (DataFrame): Cleaned flights DataFrame
    Returns:
        DataFrame: Processed DataFrame ready for modeling
    """
    fe = FeatureEngineer()
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

    result_df = fe.get_dataframe()
    
    result_df = reduce_memory_usage(result_df, verbose=False)
    
    del cleaned_flights_df, fe
    collect()

    return result_df


def split_data(cleaned_flights_df: DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[DataFrame, DataFrame, Series, Series]:
    """
    Split data with memory optimization
    Args:
        cleaned_flights_df (DataFrame): Cleaned flights DataFrame
        test_size (float): Proportion of test set
        random_state (int): Random state for reproducibility
    Returns:
        Tuple[DataFrame, DataFrame, Series, Series]: X_train, X_test,
    """
    logger.info(f"{'='*80}")
    logger.info(f"📊 DATA SPLIT")
    logger.info(f"{'='*80}")
    logger.info(f"Original shape: {cleaned_flights_df.shape}")

    X = cleaned_flights_df.drop(columns=["ARRIVAL_DELAY"])
    y = cleaned_flights_df["ARRIVAL_DELAY"].astype(np.int8)

    logger.info(f"Features: {X.shape[1]} | Samples: {len(X):,}")
    logger.info(f"Class distribution:")
    logger.info(f"  Class 0: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
    logger.info(f"  Class 1: {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    del X, y, cleaned_flights_df
    collect()

    logger.info(f"✂️  Split: Train={len(X_train):,} | Test={len(X_test):,}")

    logger.info(f"{'='*80}")
    logger.info(f"🔧 FEATURE ENGINEERING - Training Set")
    logger.info(f"{'='*80}")
    X_train_processed = preprocess_flights_data(X_train)
    y_train = y_train.loc[X_train_processed.index].astype(np.int8)

    logger.info(f"{'='*80}")
    logger.info(f"🔧 FEATURE ENGINEERING - Test Set")
    logger.info(f"{'='*80}")
    X_test_processed = preprocess_flights_data(X_test)
    y_test = y_test.loc[X_test_processed.index].astype(np.int8)

    logger.info(f"{'='*80}")
    logger.info(f"✅ FINAL SHAPE OF DATA")
    logger.info(f"{'='*80}")
    logger.info(f"X_train: {X_train_processed.shape} | y_train: {y_train.shape}")
    logger.info(f"X_test:  {X_test_processed.shape} | y_test:  {y_test.shape}")
    logger.info(f"{'='*80}")

    del X_train, X_test
    collect()

    return X_train_processed, X_test_processed, y_train, y_test


def main() -> Tuple[LGBMClassifier, Dict, Dict]:
    """
    Full pipeline to perform LightGBM Grid Search with memory optimization
    Returns:
        Tuple[LGBMClassifier, Dict, Dict]: Best model, metrics, and grid results
    """
    
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent.parent
    CLEANED_DATA_PATH = root_dir / "data/interim/cleaned_flights.csv"
    
    logger.info("=" * 80)
    logger.info("🚀 LIGHTGBM RANDOM GRID SEARCH - MEMORY OPTIMIZED")
    logger.info("=" * 80)
    
    logger.info("📂 Loading data...")
    cleaned_flights_df = read_csv(CLEANED_DATA_PATH, low_memory=False)
    
    cleaned_flights_df = reduce_memory_usage(cleaned_flights_df)
    
    cleaned_flights_df["ARRIVAL_DELAY"] = (
        cleaned_flights_df["ARRIVAL_DELAY"] >= 15
    ).astype(np.int8)
    
    logger.info(f"✅ Loaded {len(cleaned_flights_df):,} flights")
    
    X_train, X_test, y_train, y_test = split_data(
        cleaned_flights_df, test_size=0.3, random_state=42
    )
    
    best_model, metrics, grid_results = lgbm_random_grid_search_optimized(
        X_train, y_train, X_test, y_test,
        cv_splits=5,
    )
    
    output_path = root_dir / "data/models/benchmarks/lightgbm_random_grid_search.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_to_save = {
        "metrics": metrics,
        "grid_results": grid_results
    }
    
    with open(output_path, "w") as f:
        dump(results_to_save, f, indent=4, default=str)
    
    logger.info(f"💾 Results saved to: {output_path}")
    logger.info("=" * 80)
    logger.info("✅ COMPLETED!")
    logger.info("=" * 80)
    
    return best_model, metrics, grid_results


if __name__ == "__main__":
    main()