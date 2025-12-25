import sys
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
import json
import gc

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

# LightGBM para feature importance
from lightgbm import LGBMClassifier

# Técnicas de balanceamento
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Adicionar path para importar FeatureEngineer
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from src.etl.feature_engineering import FeatureEngineer

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================
import multiprocessing

N_CORES = max(1, multiprocessing.cpu_count() - 2)
print(f"🖥️  CPUs disponíveis: {multiprocessing.cpu_count()} | Usando: {N_CORES} cores\n")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
fe = FeatureEngineer()


def preprocess_flights_data(cleaned_flights_df):
    """Feature engineering pipeline com drop otimizado"""
    fe.set_dataframe(cleaned_flights_df.copy())
    # "TAXI_OUT", "WHEELS_OFF", "DEPARTURE_DELAY"
    COLUMNS_DROP = [
        "DAY_OF_WEEK",
        "DAY_OF_YEAR",
        "SCHEDULED_DEPARTURE_MINUTES",
        "SCHEDULED_ARRIVAL_MINUTES",
        "SCHEDULED_DEPARTURE",
        "SCHEDULED_ARRIVAL",
        "FLIGHT_NUMBER",
    ]

    # Pipeline otimizado
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
    gc.collect()
    return fe.get_dataframe()


# ============================================================================
# TESTE 1: BASE COMPLETA
# ============================================================================
def test_1_full_dataset(X_train, y_train, X_test, y_test):
    """
    Teste 1: Naive Bayes com todas as features
    """
    print(f"\n{'='*80}")
    print(f"📊 TESTE 1: NAIVE BAYES - BASE COMPLETA")
    print(f"{'='*80}")
    
    # Remove colunas não processadas
    X_train_clean = X_train.drop(columns=["DISTANCE", "SCHEDULED_TIME"], errors="ignore")
    X_test_clean = X_test.drop(columns=["DISTANCE", "SCHEDULED_TIME"], errors="ignore")
    
    print(f"Features utilizadas: {X_train_clean.shape[1]}")
    print(f"Samples treino: {len(X_train_clean):,}")
    print(f"Samples teste: {len(X_test_clean):,}")
    
    return train_naive_bayes(X_train_clean, y_train, X_test_clean, y_test, 
                            test_name="Teste 1 - Base Completa")


# ============================================================================
# TESTE 2: TOP 15 FEATURES (LIGHTGBM)
# ============================================================================
def test_2_top_features_lightgbm(X_train, y_train, X_test, y_test, top_n=15):
    """
    Teste 2: Extrair top N features usando LightGBM e treinar Naive Bayes
    """
    print(f"\n{'='*80}")
    print(f"📊 TESTE 2: TOP {top_n} FEATURES - LIGHTGBM FEATURE IMPORTANCE")
    print(f"{'='*80}")
    
    # Remove colunas não processadas
    X_train_clean = X_train.drop(columns=["DISTANCE", "SCHEDULED_TIME"], errors="ignore")
    X_test_clean = X_test.drop(columns=["DISTANCE", "SCHEDULED_TIME"], errors="ignore")
    
    print(f"🔍 Extraindo feature importance com LightGBM...")
    
    # Treinar LightGBM simples para obter importâncias
    lgbm = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=15,
        random_state=42,
        verbose=1,
        n_jobs=N_CORES
    )
    
    start_time = time.time()
    lgbm.fit(X_train_clean, y_train)
    lgbm_time = time.time() - start_time

    # avaliar o modelo
    y_pred = lgbm.predict(X_test_clean)

    accuracy = accuracy_score(y_test, y_pred)
    recall_score_val = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    precision_score_val = precision_score(y_test, y_pred)
    f1_score_val = f1_score(y_test, y_pred)

    print(f"\n💡 LightGBM Performance (usando todas as features):"
          f"\n  Accuracy : {accuracy:.4f}"
          f"\n  Precision: {precision_score_val:.4f}"
          f"\n  Recall   : {recall_score_val:.4f}"
          f"\n  ROC-AUC  : {roc:.4f}"
          f"\n  F1-Score : {f1_score_val:.4f}")
    
    # Obter importâncias
    feature_importance = pd.DataFrame({
        'feature': X_train_clean.columns,
        'importance': lgbm.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n✅ LightGBM treinado em {lgbm_time:.2f}s")
    print(f"\n🏆 Top {top_n} Features por Importância:")
    print(feature_importance.head(top_n).to_string(index=False))

    # se não tiver salva salvar em csv
    feature_importance.to_csv("lightgbm_feature_importance.csv", index=False)
    
    # Selecionar top features
    top_features = feature_importance.head(top_n)['feature'].tolist()
    
    X_train_top = X_train_clean[top_features]
    X_test_top = X_test_clean[top_features]
    
    print(f"\n📊 Dataset reduzido:")
    print(f"  Features: {len(top_features)}")
    print(f"  Samples treino: {len(X_train_top):,}")
    
    del lgbm
    gc.collect()
    
    return train_naive_bayes(X_train_top, y_train, X_test_top, y_test,
                            test_name=f"Teste 2 - Top {top_n} Features (LightGBM)")


# ============================================================================
# TESTE 3: FEATURES COM CORRELAÇÃO > |0.6|
# ============================================================================
def test_3_high_correlation(X_train, y_train, X_test, y_test, threshold=0.6):
    """
    Teste 3: Selecionar features com correlação absoluta > threshold com target
    """
    print(f"\n{'='*80}")
    print(f"📊 TESTE 3: FEATURES COM CORRELAÇÃO > |{threshold}|")
    print(f"{'='*80}")
    
    # Remove colunas não processadas
    X_train_clean = X_train.drop(columns=["DISTANCE", "SCHEDULED_TIME"], errors="ignore")
    X_test_clean = X_test.drop(columns=["DISTANCE", "SCHEDULED_TIME"], errors="ignore")
    
    print(f"🔍 Calculando correlações com target...")
    
    # Calcular correlação com target
    correlations = X_train_clean.corrwith(y_train).abs().sort_values(ascending=False)
    
    # Filtrar features com correlação > threshold
    high_corr_features = correlations[correlations > threshold].index.tolist()
    
    print(f"\n📊 Features com correlação > |{threshold}|: {len(high_corr_features)}")
    
    if len(high_corr_features) == 0:
        print(f"⚠️  Nenhuma feature com correlação > |{threshold}|")
        print(f"📉 Top 10 correlações encontradas:")
        print(correlations.head(10))
        print(f"\n💡 Usando top 15 features ao invés...")
        high_corr_features = correlations.head(15).index.tolist()
    else:
        print(f"\n🏆 Features selecionadas:")
        for feat in high_corr_features:
            print(f"  {feat}: {correlations[feat]:.4f}")
    
    X_train_corr = X_train_clean[high_corr_features]
    X_test_corr = X_test_clean[high_corr_features]
    
    return train_naive_bayes(X_train_corr, y_train, X_test_corr, y_test,
                            test_name=f"Teste 3 - Correlação > |{threshold}|")


# ============================================================================
# TESTE 4: SMOTE (BALANCEAMENTO COMPLETO)
# ============================================================================
def test_4_smote_full(X_train, y_train, X_test, y_test):
    """
    Teste 4: SMOTE com sampling_strategy='auto' (balanceamento completo 1:1)
    """
    print(f"\n{'='*80}")
    print(f"📊 TESTE 4: SMOTE - BALANCEAMENTO COMPLETO (1:1)")
    print(f"{'='*80}")
    
    # Remove colunas não processadas
    X_train_clean = X_train.drop(columns=["DISTANCE", "SCHEDULED_TIME"], errors="ignore")
    X_test_clean = X_test.drop(columns=["DISTANCE", "SCHEDULED_TIME"], errors="ignore")
    
    # Aplicar SMOTE
    print(f"⚖️  Aplicando SMOTE...")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    
    original_counts = pd.Series(y_train).value_counts()
    print(f"\n📊 Distribuição original:")
    print(f"  Classe 0: {original_counts[0]:,}")
    print(f"  Classe 1: {original_counts[1]:,}")
    print(f"  Ratio: {original_counts[0]/original_counts[1]:.2f}:1")
    
    start_time = time.time()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_clean, y_train)
    smote_time = time.time() - start_time
    
    balanced_counts = pd.Series(y_train_balanced).value_counts()
    print(f"\n📊 Distribuição balanceada:")
    print(f"  Classe 0: {balanced_counts[0]:,}")
    print(f"  Classe 1: {balanced_counts[1]:,}")
    print(f"  Ratio: {balanced_counts[0]/balanced_counts[1]:.2f}:1")
    print(f"  Tempo: {smote_time:.2f}s")
    
    return train_naive_bayes(X_train_balanced, y_train_balanced, X_test_clean, y_test,
                            test_name="Teste 4 - SMOTE (1:1)")


# ============================================================================
# TESTE 5: SMOTE 30%
# ============================================================================
def test_5_smote_30_percent(X_train, y_train, X_test, y_test):
    """
    Teste 5: SMOTE com sampling_strategy=0.3 (classe minoritária = 30% da majoritária)
    """
    print(f"\n{'='*80}")
    print(f"📊 TESTE 5: SMOTE - 30% DA CLASSE MAJORITÁRIA")
    print(f"{'='*80}")
    
    # Remove colunas não processadas
    X_train_clean = X_train.drop(columns=["DISTANCE", "SCHEDULED_TIME"], errors="ignore")
    X_test_clean = X_test.drop(columns=["DISTANCE", "SCHEDULED_TIME"], errors="ignore")
    
    # Aplicar SMOTE com 30%
    print(f"⚖️  Aplicando SMOTE (30%)...")
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    
    original_counts = pd.Series(y_train).value_counts()
    print(f"\n📊 Distribuição original:")
    print(f"  Classe 0: {original_counts[0]:,}")
    print(f"  Classe 1: {original_counts[1]:,}")
    print(f"  Ratio: {original_counts[0]/original_counts[1]:.2f}:1")
    
    start_time = time.time()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_clean, y_train)
    smote_time = time.time() - start_time
    
    balanced_counts = pd.Series(y_train_balanced).value_counts()
    print(f"\n📊 Distribuição balanceada:")
    print(f"  Classe 0: {balanced_counts[0]:,}")
    print(f"  Classe 1: {balanced_counts[1]:,}")
    print(f"  Ratio: {balanced_counts[0]/balanced_counts[1]:.2f}:1")
    print(f"  Tempo: {smote_time:.2f}s")
    
    return train_naive_bayes(X_train_balanced, y_train_balanced, X_test_clean, y_test,
                            test_name="Teste 5 - SMOTE (30%)")


# ============================================================================
# TESTE 6: RANDOM OVERSAMPLING (IGUALAR À CLASSE MAJORITÁRIA)
# ============================================================================
def test_6_random_oversample(X_train, y_train, X_test, y_test):
    """
    Teste 6: Random Oversampling - duplicar aleatoriamente classe minoritária
    até igualar à classe majoritária
    """
    print(f"\n{'='*80}")
    print(f"📊 TESTE 6: RANDOM OVERSAMPLING - IGUALAR CLASSES")
    print(f"{'='*80}")
    
    # Remove colunas não processadas
    X_train_clean = X_train.drop(columns=["DISTANCE", "SCHEDULED_TIME"], errors="ignore")
    X_test_clean = X_test.drop(columns=["DISTANCE", "SCHEDULED_TIME"], errors="ignore")
    
    # Aplicar Random Oversampling
    print(f"⚖️  Aplicando Random Oversampling...")
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    
    original_counts = pd.Series(y_train).value_counts()
    print(f"\n📊 Distribuição original:")
    print(f"  Classe 0: {original_counts[0]:,}")
    print(f"  Classe 1: {original_counts[1]:,}")
    print(f"  Ratio: {original_counts[0]/original_counts[1]:.2f}:1")
    
    start_time = time.time()
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train_clean, y_train)
    ros_time = time.time() - start_time
    
    balanced_counts = pd.Series(y_train_balanced).value_counts()
    print(f"\n📊 Distribuição balanceada:")
    print(f"  Classe 0: {balanced_counts[0]:,}")
    print(f"  Classe 1: {balanced_counts[1]:,}")
    print(f"  Ratio: {balanced_counts[0]/balanced_counts[1]:.2f}:1")
    print(f"  Tempo: {ros_time:.2f}s")
    print(f"\n💡 Método: Duplicação aleatória de amostras da classe minoritária")
    
    return train_naive_bayes(X_train_balanced, y_train_balanced, X_test_clean, y_test,
                            test_name="Teste 6 - Random Oversampling")

def find_best_threshold(y_true, y_proba):
    thresholds = np.array([0.05, 0.3, 0.5, 0.7, 0.95])
    f1_scores = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

# ============================================================================
# FUNÇÃO GENÉRICA DE TREINAMENTO NAIVE BAYES
# ============================================================================
def train_naive_bayes(X_train, y_train, X_test, y_test, test_name="Naive Bayes", 
                     n_iter=9, cv_splits=5, threshold=0.5):
    """
    Treina Naive Bayes com RandomizedSearchCV
    """
    print(f"\n{'='*80}")
    print(f"🚀 TREINANDO: {test_name}")
    print(f"{'='*80}")
    print(f"CV Folds: {cv_splits} | Random iterations: {n_iter}")
    
    # Hiperparâmetros
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
        verbose=0,
        random_state=42,
        n_jobs=2,
        return_train_score=True,
    )

    start_time = time.time()
    print("⏳ Treinando modelo...\n")
    random_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print(f"✅ Treinamento concluído - {training_time:.2f}s")
    print(f"\n🏆 Melhores Hiperparâmetros:")
    for param, value in sorted(best_params.items()):
        print(f"  {param:25} : {value}")
    print(f"\n🎯 Melhor CV F1-Score: {best_score:.4f}")

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    y_pred_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]

    best_threshold, _ = find_best_threshold(
        y_train, y_pred_train_proba
    )

    print(f"\n🔍 Análise de Thresholds: {threshold}" )
    print(f"Best threshold (treino): {best_threshold:.4f}")
    
    y_pred_test = (y_pred_test_proba >= threshold).astype(int)
    # ROC Curve
    _, _, thresholds = roc_curve(y_test, y_pred_test_proba)
    roc_auc = roc_auc_score(y_test, y_pred_test_proba)

    print(f"Median threshold (teste): {np.median(thresholds):.4f} | AUC: {roc_auc:.4f}")
    print(f"Mean threshold (teste): {np.mean(thresholds):.4f}")
    print(f"Custom threshold (teste): {threshold:.4f}")


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
        "best_threshold": best_threshold,
    }

    # Display Metrics
    print(f"\n{'='*80}")
    print(f"📈 PERFORMANCE DO MODELO")
    print(f"{'='*80}")   
    print(f"{'Métrica':<15} {'Treino':>15} {'Teste':>15} {'Diff':>15} {'Overfit?':>12}")
    print(f"{'-'*80}")
    
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        train_val = metrics["train"][metric]
        test_val = metrics["test"][metric]
        diff = abs(train_val - test_val)
        overfit = "⚠️ Sim" if diff > 0.1 else "✅ Não"
        print(f"{metric:<15} {train_val:>15.4f} {test_val:>15.4f} {diff:>15.4f} {overfit:>12}")
    
    print(f"{'-'*80}")
    print(f"{'CV F1-Score':<15} {metrics['cv_score']:>15.4f}")
    print(f"{'='*80}\n")

    # Confusion Matrix
    print(f"📊 Matriz de Confusão (Teste):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\n                Predito")
    print(f"                Sem Atraso  |  Com Atraso")
    print(f"Real Sem Atraso    {cm[0,0]:>6}  |  {cm[0,1]:>6}")
    print(f"Real Com Atraso    {cm[1,0]:>6}  |  {cm[1,1]:>6}")
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n📊 Métricas Adicionais:")
    print(f"  True Positives:  {tp:>6} (Atrasos corretamente previstos)")
    print(f"  True Negatives:  {tn:>6} (Sem atraso corretamente previsto)")
    print(f"  False Positives: {fp:>6} (Falso alarme de atraso)")
    print(f"  False Negatives: {fn:>6} (Atrasos não detectados)")
    print(f"  Sensitivity:     {sensitivity:>6.4f} (Taxa de Verdadeiros Positivos)")
    print(f"  Specificity:     {specificity:>6.4f} (Taxa de Verdadeiros Negativos)")

    print(f"\n{'='*80}\n")

    # ROC data
    roc_data = {
        "thresholds": thresholds.tolist(),
        "auc": roc_auc
    }

    del random_search
    gc.collect()

    return best_model, metrics, roc_data


# ============================================================================
# SPLIT DATA
# ============================================================================
def split_data(cleaned_flights_df, test_size=0.2, random_state=42):
    """Split com feature engineering aplicado"""
    print(f"{'='*80}")
    print(f"📊 DIVISÃO DOS DADOS")
    print(f"{'='*80}")
    print(f"Shape original: {cleaned_flights_df.shape}")

    X = cleaned_flights_df.drop(columns=["ARRIVAL_DELAY"])
    y = cleaned_flights_df["ARRIVAL_DELAY"]

    print(f"Features: {X.shape[1]} | Amostras: {len(X):,}")
    print(f"Distribuição das classes:")
    print(f"  Classe 0: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Classe 1: {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    del X, y
    gc.collect()

    print(f"\n✂️  Split: Treino={len(X_train):,} | Teste={len(X_test):,}")

    print(f"\n{'='*80}")
    print(f"🔧 FEATURE ENGINEERING - Conjunto de Treino")
    print(f"{'='*80}")
    X_train_processed = preprocess_flights_data(X_train)
    print(f"Features Columns: {X_train_processed.columns.tolist()}")
    y_train = y_train.loc[X_train_processed.index]

    print(f"\n{'='*80}")
    print(f"🔧 FEATURE ENGINEERING - Conjunto de Teste")
    print(f"{'='*80}")
    X_test_processed = preprocess_flights_data(X_test)
    y_test = y_test.loc[X_test_processed.index]

    print(f"\n{'='*80}")
    print(f"✅ SHAPE FINAL DOS DADOS")
    print(f"{'='*80}")
    print(f"X_train: {X_train_processed.shape} | y_train: {y_train.shape}")
    print(f"X_test:  {X_test_processed.shape} | y_test:  {y_test.shape}")
    print(f"{'='*80}\n")

    del X_train, X_test
    gc.collect()

    return X_train_processed, X_test_processed, y_train, y_test


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    benchmark_results = {}

    # Carregar dados
    print("=" * 80)
    print("🚀 BENCHMARK: NAIVE BAYES - 6 TESTES")
    print("=" * 80)
    print("📂 Carregando dados...")
    
    cleaned_flights_df = pd.read_csv(
        "/home/lucas/IBelieveICanFlyPy/data/interim/cleaned_flights.csv"
    )
    
    # Criar target binário
    cleaned_flights_df["ARRIVAL_DELAY"] = (
        cleaned_flights_df["ARRIVAL_DELAY"] >= 15
    ).astype(int)
    
    print(f"✅ Carregados {len(cleaned_flights_df):,} voos")
    print(f"📊 Target: ARRIVAL_DELAY >= 15 min = 1 (Atraso)")
    print(f"   Classe 0: {(cleaned_flights_df['ARRIVAL_DELAY']==0).sum():,}")
    print(f"   Classe 1: {(cleaned_flights_df['ARRIVAL_DELAY']==1).sum():,}\n")

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        cleaned_flights_df, test_size=0.3, random_state=42
    )

    # ========================================================================
    # EXECUTAR OS 6 TESTES
    # ========================================================================
    
    # Teste 1: Base completa
    print("\n" + "🔷" * 40)
    model_1, metrics_1, roc_1 = test_1_full_dataset(
        X_train.copy(), y_train.copy(), 
        X_test.copy(), y_test.copy()
    )
    benchmark_results["teste_1_base_completa"] = {
        "description": "Base completa - todas as features",
        "metrics": metrics_1,
        "roc_curve": roc_1
    }
    del model_1; gc.collect()

    # Teste 2: Top 15 features (LightGBM)
    print("\n" + "🔷" * 40)
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
    del model_2; gc.collect()

    # Teste 3: Correlação > |0.6|
    print("\n" + "🔷" * 40)
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
    del model_3; gc.collect()

    # Teste 4: SMOTE (1:1)
    print("\n" + "🔷" * 40)
    model_4, metrics_4, roc_4 = test_4_smote_full(
        X_train.copy(), y_train.copy(), 
        X_test.copy(), y_test.copy()
    )
    benchmark_results["teste_4_smote_full"] = {
        "description": "SMOTE - Balanceamento completo 1:1",
        "metrics": metrics_4,
        "roc_curve": roc_4
    }
    del model_4; gc.collect()

    # Teste 5: SMOTE 30%
    print("\n" + "🔷" * 40)
    model_5, metrics_5, roc_5 = test_5_smote_30_percent(
        X_train.copy(), y_train.copy(), 
        X_test.copy(), y_test.copy()
    )
    benchmark_results["teste_5_smote_30"] = {
        "description": "SMOTE - 30% da classe majoritária",
        "metrics": metrics_5,
        "roc_curve": roc_5
    }
    del model_5; gc.collect()

    # Teste 6: Random Oversampling
    print("\n" + "🔷" * 40)
    model_6, metrics_6, roc_6 = test_6_random_oversample(
        X_train.copy(), y_train.copy(), 
        X_test.copy(), y_test.copy()
    )
    benchmark_results["teste_6_random_oversample"] = {
        "description": "Random Oversampling - Igualar classes",
        "metrics": metrics_6,
        "roc_curve": roc_6
    }
    del model_6; gc.collect()

    # ========================================================================
    # COMPARAÇÃO FINAL DOS 6 TESTES
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 COMPARAÇÃO FINAL - 6 TESTES NAIVE BAYES")
    print("=" * 80)
    
    comparison_df = pd.DataFrame([
        {
            "Teste": results["description"],
            "F1-Score": results["metrics"]["test"]["F1-Score"],
            "Recall": results["metrics"]["test"]["Recall"],
            "Precision": results["metrics"]["test"]["Precision"],
            "ROC-AUC": results["metrics"]["test"]["ROC-AUC"],
            "Accuracy": results["metrics"]["test"]["Accuracy"],
            "Tempo (s)": results["metrics"]["training_time"],
        }
        for test_name, results in benchmark_results.items()
    ])
    
    # Ordenar por F1-Score
    comparison_df = comparison_df.sort_values("F1-Score", ascending=False)
    comparison_df.insert(0, 'Rank', range(1, len(comparison_df) + 1))
    
    print("\n🏆 Ranking por F1-Score (Teste):")
    print("=" * 120)
    print(comparison_df.to_string(index=False))
    print("=" * 120)
    
    # Identificar o melhor teste
    best_test = comparison_df.iloc[0]
    print(f"\n🥇 MELHOR TESTE: {best_test['Teste']}")
    print(f"   F1-Score: {best_test['F1-Score']:.4f}")
    print(f"   Recall:   {best_test['Recall']:.4f}")
    print(f"   Precision: {best_test['Precision']:.4f}")
    print(f"   ROC-AUC:  {best_test['ROC-AUC']:.4f}")
    
    # Análise de trade-offs
    print(f"\n📊 ANÁLISE DE TRADE-OFFS:")
    print(f"\n   Melhor F1-Score:  {comparison_df.iloc[0]['Teste']} ({comparison_df.iloc[0]['F1-Score']:.4f})")
    print(f"   Melhor Recall:    {comparison_df.loc[comparison_df['Recall'].idxmax()]['Teste']} ({comparison_df['Recall'].max():.4f})")
    print(f"   Melhor Precision: {comparison_df.loc[comparison_df['Precision'].idxmax()]['Teste']} ({comparison_df['Precision'].max():.4f})")
    print(f"   Melhor ROC-AUC:   {comparison_df.loc[comparison_df['ROC-AUC'].idxmax()]['Teste']} ({comparison_df['ROC-AUC'].max():.4f})")
    print(f"   Mais Rápido:      {comparison_df.loc[comparison_df['Tempo (s)'].idxmin()]['Teste']} ({comparison_df['Tempo (s)'].min():.2f}s)")
    
    # Salvar resultados
    output_path = "/home/lucas/IBelieveICanFlyPy/data/models/benchmark_naive_bayes_6_testes_taxi_out_and_wheels_off_and_departure_delay.json"
    with open(output_path, "w") as f:
        json.dump(benchmark_results, f, indent=4, default=str)

    # Salvar tabela de comparação
    comparison_csv_path = "/home/lucas/IBelieveICanFlyPy/data/models/benchmark_naive_bayes_comparison_taxi_out_and_wheels_off_and_departure_delay.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)

    print(f"\n💾 Resultados salvos:")
    print(f"   JSON: {output_path}")
    print(f"   CSV:  {comparison_csv_path}")
    print("=" * 80)
    print("✅ TODOS OS 6 TESTES CONCLUÍDOS!")
    print("=" * 80)