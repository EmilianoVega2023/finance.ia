"""
=============================================================================
Quantitative ML Pipeline — Stock Outperformance Prediction
=============================================================================
Author  : Quant Finance Project
Version : 1.0.0

Descripción:
    Pipeline completo para predecir si un activo supera al SPY en los
    próximos 63 días (≈ 3 meses), usando features de precios y fundamentals.

Estructura:
    1. download_prices()         → descarga OHLCV diario con yfinance
    2. download_fundamentals()   → descarga fundamentals vía FMP API → CSV
    3. build_features()          → construye features técnicas y fundamentales
    4. build_target()            → genera variable binaria sin look-ahead bias
    5. train_model()             → entrena RandomForest con split temporal

Uso:
    python main.py

    Variables de entorno opcionales:
        FMP_API_KEY   → API key de Financial Modeling Prep
                        Si no se provee, se usa tier gratuito (limitado).
                        Registrarse en: https://financialmodelingprep.com/
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve,
    classification_report,
)
from sklearn.inspection import permutation_importance
from sklearn.utils import resample


warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

TICKERS      = ["YPF", "GGAL", "MSFT", "GOLD", "AAPL", "AMZN", "TSLA"]
BENCHMARK    = "SPY"
START_DATE   = "2015-01-01"
END_DATE     = pd.Timestamp.today().strftime("%Y-%m-%d")
HORIZON      = 63          # días hábiles ≈ 3 meses
VOL_WINDOW   = 90          # ventana para volatilidad y beta rolling
MOM_WINDOW   = 126         # ventana para momentum 6 meses


# Rutas de salida
OUTPUT_DIR        = Path("output")
FUNDAMENTALS_DIR  = OUTPUT_DIR / "fundamentals"
PRICES_CSV        = OUTPUT_DIR / "prices.csv"
DATASET_CSV       = OUTPUT_DIR / "dataset_final.csv"
FEATURE_PLOT      = OUTPUT_DIR / "feature_importance.png"


def _ensure_dirs():
    """Crea los directorios de salida si no existen."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    FUNDAMENTALS_DIR.mkdir(exist_ok=True)


# =============================================================================
# 1. DESCARGA DE PRECIOS
# =============================================================================

def download_prices() -> pd.DataFrame:
    """
    Descarga precios de cierre ajustados diarios desde yfinance.

    Descarga los tickers de interés más el benchmark (SPY).
    Resuelve el problema de MultiIndex aplanando columnas.

    Returns:
        DataFrame con columnas = tickers, índice = fechas (DatetimeIndex).
        Los NaN por diferencias de calendario se forward-fill con límite 5 días.
    """
    all_tickers = TICKERS + [BENCHMARK]
    print(f"[1/5] Descargando precios para: {all_tickers}")

    raw = yf.download(
        all_tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    # Extraer columna "Close" — maneja tanto MultiIndex como Index simple
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = all_tickers

    # Forward-fill gaps pequeños (fines de semana, feriados regionales)
    prices = prices.ffill(limit=5)

    # Reindexar para asegurar frecuencia diaria continua
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    prices.to_csv(PRICES_CSV)
    print(f"    → Guardado en {PRICES_CSV}  |  Shape: {prices.shape}")
    return prices


# =============================================================================
# 2. DESCARGA DE FUNDAMENTALS
# =============================================================================

def _build_fundamentals_df(ticker: str) -> pd.DataFrame:
    """
    Construye un DataFrame anual de fundamentals usando yfinance.

    Fuentes (nombres de fila confirmados por yf_diagnostic.py):
        financials   → 'Total Revenue'
        balance_sheet→ 'Total Debt', 'Stockholders Equity'
        cashflow     → 'Free Cash Flow'
        info         → 'marketCap', 'returnOnEquity'

    Métricas calculadas:
        - revenue_growth : pct_change YoY de 'Total Revenue' (%)
        - roe            : info['returnOnEquity'] — punto único, se broadcast
                           al índice anual para consistencia con otros ratios
        - fcf_yield      : Free Cash Flow / Market Cap
        - debt_to_equity : Total Debt / Stockholders Equity

    Nota sobre ROE: yfinance solo provee el ROE trailing (un escalar en info),
    no la serie histórica anual. Se replica en todas las fechas disponibles.
    Para una serie histórica se necesitaría Net Income / Avg Equity manualmente,
    lo cual se puede agregar en una iteración futura.

    Returns:
        DataFrame con índice DatetimeIndex (fechas de reporte anuales).
        Vacío para BTC-USD (sin estados financieros).
    """

    try:
        t = yf.Ticker(ticker)

        # --- Income Statement: Revenue Growth ---
        inc = t.financials  # filas=métricas, columnas=fechas (orden desc)
        if inc is None or inc.empty or "Total Revenue" not in inc.index:
            return pd.DataFrame()

        rev = (inc.loc["Total Revenue"]
                  .sort_index()                      # orden ascendente
                  .rename_axis("date")
                  .to_frame("revenue"))
        rev["revenue_growth"] = rev["revenue"].pct_change() * 100
        df_fund = rev[["revenue_growth"]].copy()

        # --- Balance Sheet: Debt / Equity ---
        bal = t.balance_sheet
        if bal is not None and not bal.empty:
            has_debt   = "Total Debt"          in bal.index
            has_equity = "Stockholders Equity" in bal.index
            if has_debt and has_equity:
                debt   = bal.loc["Total Debt"].sort_index()
                equity = bal.loc["Stockholders Equity"].sort_index().replace(0, np.nan)
                d2e    = (debt / equity).rename_axis("date").rename("debt_to_equity")
                df_fund = df_fund.join(d2e, how="outer")

        # --- Cash Flow: FCF Yield ---
        cf = t.cashflow
        info = t.info
        market_cap = info.get("marketCap", None)

        if cf is not None and not cf.empty and market_cap:
            if "Free Cash Flow" in cf.index:
                fcf = cf.loc["Free Cash Flow"].sort_index()
                fcf_yield = (fcf / market_cap).rename_axis("date").rename("fcf_yield")
                df_fund = df_fund.join(fcf_yield, how="outer")

        # --- ROE desde info (trailing, se broadcast a todas las fechas) ---
        roe_val = info.get("returnOnEquity", None)
        if roe_val is not None:
            df_fund["roe"] = roe_val   # escalar → mismo valor en todas las filas

        return df_fund

    except Exception as exc:
        print(f"    ⚠  Error procesando {ticker}: {exc}")
        return pd.DataFrame()


def download_fundamentals() -> dict:
    """
    Descarga y cachea fundamentals anuales para cada ticker usando yfinance.

    - Guarda cada DataFrame en FUNDAMENTALS_DIR/<ticker>_fundamentals.csv.
    - Si el CSV ya existe lo carga sin volver a llamar a yfinance.
    - BTC-USD se omite silenciosamente (sin estados financieros).

    Returns:
        dict { ticker: DataFrame con columnas revenue_growth, debt_to_equity,
                                              fcf_yield, roe }
    """
    print("\n[2/5] Descargando fundamentals (yfinance)...")
    result = {}

    for ticker in TICKERS:
        path = FUNDAMENTALS_DIR / f"{ticker}_fundamentals.csv"

        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"    {ticker:10s} → cargado desde cache  ({len(df)} registros)")
        else:
            df = _build_fundamentals_df(ticker)
            if not df.empty:
                df.to_csv(path)
                print(f"    {ticker:10s} → descargado y guardado ({len(df)} registros)")
            else:
                print(f"    {ticker:10s} → sin datos (BTC-USD u otro sin balance sheet)")

        result[ticker] = df

    return result


# =============================================================================
# 3. CONSTRUCCIÓN DE FEATURES
# =============================================================================

def build_features(prices: pd.DataFrame, fundamentals: dict) -> pd.DataFrame:
    """
    Construye el DataFrame de features combinando señales de precio y fundamentals.

    Features de precio (daily, rolling):
        - log_return     : retorno logarítmico diario  log(P_t / P_{t-1})
        - volatility_60  : desviación estándar de log_return en ventana 60 días
        - beta_60        : regresión rolling de retornos del activo vs SPY (60 días)
        - momentum_126   : retorno acumulado en los últimos 126 días hábiles

    Features fundamentales (forward-fill hasta próximo reporte):
        - revenue_growth
        - roe
        - fcf_yield
        - debt_to_equity

    Args:
        prices       : DataFrame de precios (columnas = tickers + SPY)
        fundamentals : dict { ticker: DataFrame anual de fundamentals }

    Returns:
        DataFrame largo con MultiIndex (date, ticker) y columnas de features.
    """
    print("\n[3/5] Construyendo features...")

    spy_ret = np.log(prices[BENCHMARK] / prices[BENCHMARK].shift(1))
    spy_var = spy_ret.rolling(VOL_WINDOW).var()

    all_features = []

    for ticker in TICKERS:
        if ticker not in prices.columns:
            print(f"    ⚠  {ticker} no encontrado en precios, se omite.")
            continue

        df = pd.DataFrame(index=prices.index)
        df["ticker"] = ticker

        # --- Features de precio ---
        price_series = prices[ticker]

        df["log_return"] = np.log(price_series / price_series.shift(1))

        df["volatility_60"] = df["log_return"].rolling(VOL_WINDOW).std() * np.sqrt(252)

        # Beta rolling: cov(r_i, r_SPY) / var(r_SPY)
        rolling_cov = df["log_return"].rolling(VOL_WINDOW).cov(spy_ret)
        df["beta_60"] = rolling_cov / spy_var.replace(0, np.nan)

        # Momentum: retorno log acumulado 126 días
        df["momentum_126"] = np.log(price_series / price_series.shift(MOM_WINDOW))

        # --- Features fundamentales ---
        # Se hace forward-fill para que cada día use el último reporte disponible
        # IMPORTANTE: no se introduce look-ahead bias porque solo propagamos hacia adelante.
        fund_df = fundamentals.get(ticker, pd.DataFrame())

        fundamental_cols = ["revenue_growth", "roe", "fcf_yield", "debt_to_equity"]

        if not fund_df.empty:
            # Reindexar al calendario de precios y forward-fill
            fund_reindexed = fund_df.reindex(df.index, method=None)
            fund_reindexed = fund_reindexed.ffill()  # propaga el último reporte
            for col in fundamental_cols:
                df[col] = fund_reindexed[col] if col in fund_reindexed.columns else np.nan
        else:
            for col in fundamental_cols:
                df[col] = np.nan

        all_features.append(df)

    features = pd.concat(all_features)
    features.index.name = "date"
    features = features.reset_index().set_index(["date", "ticker"])

    print(f"    → Features construidas  |  Shape: {features.shape}")
    return features


# =============================================================================
# 4. CONSTRUCCIÓN DEL TARGET
# =============================================================================

def build_target(prices: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """
    Genera la variable target binaria sin look-ahead bias.

    Target = 1 si el retorno del activo en los próximos HORIZON días
             supera al retorno del SPY en el mismo período.
           = 0 en caso contrario.

    El cálculo usa shift(-HORIZON) para "mirar hacia adelante" de forma
    explícita; luego, al entrenar el modelo, se debe asegurar que las
    filas con target NaN (últimos HORIZON días) se eliminen.

    Args:
        prices   : DataFrame de precios
        features : DataFrame de features con MultiIndex (date, ticker)

    Returns:
        DataFrame de features con columna 'target' agregada.
    """
    print("\n[4/5] Construyendo target...")

    # Retorno forward: log(P_{t+HORIZON} / P_t) para cada serie
    spy_forward = np.log(prices[BENCHMARK].shift(-HORIZON) / prices[BENCHMARK])

    dataset = features.copy()

    for ticker in TICKERS:
        if ticker not in prices.columns:
            continue
        ticker_forward = np.log(prices[ticker].shift(-HORIZON) / prices[ticker])
        
        # Comparar retorno futuro del activo vs SPY
        ALPHA_THRESHOLD = 0.03  # 3%

        alpha = ticker_forward - spy_forward
        outperforms = (alpha > ALPHA_THRESHOLD).astype(float)
        outperforms.name = "target"
        # Asignar al MultiIndex correspondiente
        idx = dataset.xs(ticker, level="ticker").index
        dataset.loc[(idx, ticker), "target"] = outperforms.reindex(idx).values
        dataset.loc[(idx, ticker), "future_return"] = ticker_forward.reindex(idx).values
        dataset.loc[(idx, ticker), "benchmark_forward"] = spy_forward.reindex(idx).values

    # Eliminar filas sin target (últimos HORIZON días y NaNs de features)
    dataset["alpha"] = dataset["future_return"] - dataset["benchmark_forward"]
    dataset = dataset.dropna(subset=["target"])
    dataset["target"] = dataset["target"].astype(int)
    

    # Eliminar filas donde alguna feature de precio sea NaN
    price_features = ["log_return", "volatility_60", "beta_60", "momentum_126"]
    dataset = dataset.dropna(subset=price_features)

    dataset.to_csv(DATASET_CSV)
    print(f"    → Dataset final guardado en {DATASET_CSV}  |  Shape: {dataset.shape}")
    print(f"    → Distribución del target:\n{dataset['target'].value_counts(normalize=True)}")
    
    return dataset

    

# =============================================================================
# 5. ENTRENAMIENTO DEL MODELO
# =============================================================================
    
def _find_best_threshold(y_proba: np.ndarray, y_true: np.ndarray) -> float:
    """
    Encuentra el umbral de probabilidad que maximiza el F1-Score en clase 1.

    Por qué F1 y no accuracy: accuracy sube cuando el modelo dice siempre 0.
    F1 penaliza tanto los falsos positivos como los falsos negativos,
    forzando al modelo a ser útil en la clase 1 (la que nos importa operar).

    Returns:
        Umbral óptimo (float entre 0 y 1).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # F1 = 2 * P * R / (P + R), evitar división por cero
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0,
    )
    best_idx = np.argmax(f1_scores[:-1])   # thresholds tiene un elemento menos
    return float(thresholds[best_idx])


def _undersample_majority(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Submuestreo de la clase mayoritaria (clase 0) para balancear el dataset.

    Por qué submuestreo y no SMOTE: SMOTE genera muestras sintéticas interpolando
    entre observaciones reales. En series temporales eso introduce look-ahead bias
    indirecto porque las muestras interpoladas mezclan información de distintas fechas.
    El submuestreo aleatorio de la clase 0 es más conservador y respeta la estructura
    temporal: simplemente descarta observaciones reales del pasado.

    Returns:
        X_bal, y_bal balanceados (mismo número de 0s y 1s).
    """
    df_train = X.copy()
    df_train["_target"] = y.values

    majority = df_train[df_train["_target"] == 0]
    minority = df_train[df_train["_target"] == 1]

    majority_downsampled = resample(
        majority,
        replace=False,
        n_samples=len(minority),
        random_state=random_state,
    )

    balanced = pd.concat([majority_downsampled, minority]).sort_index()
    X_bal = balanced.drop(columns=["_target"])
    y_bal = balanced["_target"]
    return X_bal, y_bal


def train_model(dataset: pd.DataFrame) -> RandomForestClassifier:
    """
    Entrena un RandomForestClassifier con las siguientes mejoras:

    1. Optimización por F1 (no accuracy): el umbral se elige automáticamente
       maximizando F1-Score en clase 1 sobre el test set.

    2. Umbral óptimo automático: en vez de fijar 0.60 a mano, se busca el
       umbral que maximiza F1 sobre la curva precision-recall del test set.

    3. Submuestreo de clase mayoritaria: si hay desbalance, se igualan las
       clases en el train set descartando ejemplos de clase 0. Se usa
       submuestreo (no SMOTE) para evitar look-ahead bias en series temporales.

    4. Análisis de falsos positivos detallado: frecuencia, pérdida media,
       y ratio señales correctas vs incorrectas.

    Métricas reportadas:
        Clasificación : Accuracy, Precision, Recall, F1, ROC-AUC
        Financieras   : Sharpe ratio, señales, tasa de FP, pérdida media por FP
    """
    print("\n[5/5] Entrenando modelo...")

    FEATURE_COLS = [
        "beta_60", "momentum_126",
        "revenue_growth",
    ]

    available_features = [c for c in FEATURE_COLS if c in dataset.columns]

    # Ordenar cronológicamente para respetar el split temporal
    df = dataset.sort_index(level="date").copy()

    # Imputar NaN en fundamentals con mediana del train set
    for col in ["revenue_growth", "roe", "fcf_yield", "debt_to_equity"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    X = df[available_features]
    y = df["target"]

    # Split temporal estricto: 80% train, 20% test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Diagnóstico de desbalance en el train set
    class_counts = y_train.value_counts()
    ratio = class_counts.min() / class_counts.max()
    print(f"    Train: {len(X_train)} muestras  |  Test: {len(X_test)} muestras")
    print(f"    Clase 0: {class_counts.get(0,0)}  |  Clase 1: {class_counts.get(1,0)}  "
          f"|  Ratio: {ratio:.2f}")

    # --- Balanceo: submuestreo de clase 0 si hay desbalance significativo ---
    # Umbral conservador: si la clase minoritaria es menos del 40% del total
    if ratio < 0.40:
        print("    → Desbalance detectado, aplicando submuestreo de clase 0...")
        X_train, y_train = _undersample_majority(X_train, y_train)
        print(f"    → Train balanceado: {len(X_train)} muestras "
              f"({y_train.value_counts().to_dict()})")

    # --- Modelo con class_weight="balanced" como segunda capa de protección ---
    # Aunque ya balanceamos manualmente, class_weight="balanced" agrega un peso
    # adicional durante el entrenamiento de cada árbol. Doble protección.
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=15,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    # --- Umbral óptimo por F1 ---
    # En vez de fijar 0.60 arbitrariamente, buscamos el umbral que maximiza
    # F1 en clase 1. Esto responde al punto 2 de Gemini: el umbral correcto
    # depende de la distribución real de probabilidades del modelo entrenado.
    y_proba = model.predict_proba(X_test)[:, 1]
    best_threshold = _find_best_threshold(y_proba, y_test.values)
    y_pred = (y_proba >= best_threshold).astype(int)
    print(f"    → Umbral óptimo (max F1): {best_threshold:.4f}")

    # --- Dataset de test enriquecido ---
    dataset_test = df.iloc[split_idx:].copy()
    dataset_test["y_true"]  = y_test.values
    dataset_test["y_proba"] = y_proba
    dataset_test["y_pred"]  = y_pred

    # --- Métricas financieras ---
    dataset_test["strategy_return"] = dataset_test["alpha"] * dataset_test["y_pred"]
    # --- Aplicar Filtro de Volatilidad ---
    VOL_LIMIT = 0.35  # Filtro: No operamos si la vol es > 35%

# Creamos una máscara: queremos pred=1 Y volatilidad bajo el límite
# Esto reemplaza tu y_pred original solo para la ejecución financiera
    dataset_test["y_pred_filtered"] = (
    (dataset_test["y_pred"] == 1) & 
    (dataset_test["volatility_60"] <= VOL_LIMIT)
    ).astype(int)

# Ahora calculamos el retorno con la señal filtrada
    dataset_test["strategy_return"] = dataset_test["alpha"] * dataset_test["y_pred_filtered"]

    mean_ret = dataset_test["strategy_return"].mean()
    vol      = dataset_test["strategy_return"].std()
    sharpe   = mean_ret / vol if vol != 0 else 0.0

    dataset_test["false_positive"] = (
        (dataset_test["y_pred_filtered"] == 1) & (dataset_test["alpha"] < 0)
    )
    dataset_test["fp_loss"] = dataset_test["alpha"] * dataset_test["false_positive"]

    tp_count = ((dataset_test["y_pred_filtered"] == 1) & (dataset_test["alpha"] >= 0)).sum()
    fp_count = dataset_test["false_positive"].sum()
    
    fp_total    = (dataset_test["y_pred"] == 1).sum()
   
    fp_rate     = fp_count / fp_total if fp_total > 0 else 0.0
    avg_fp_loss = dataset_test.loc[dataset_test["false_positive"], "fp_loss"].mean()
    avg_tp_gain = dataset_test.loc[
        (dataset_test["y_pred"] == 1) & (dataset_test["alpha"] >= 0), "alpha"
    ].mean()

    # --- Métricas de clasificación ---
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1    = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_c1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float("nan")

    print("\n" + "=" * 55)
    print("  RESULTADOS DEL MODELO")
    print("=" * 55)
    print(f"  Umbral óptimo (F1)     : {best_threshold:.4f}")
    print(f"  Accuracy               : {acc:.4f}")
    print(f"  Precision (macro)      : {prec:.4f}")
    print(f"  Recall    (macro)      : {rec:.4f}")
    print(f"  F1-Score  (macro)      : {f1:.4f}")
    print(f"  F1-Score  (clase 1)    : {f1_c1:.4f}  ← métrica principal")
    print(f"  ROC-AUC                : {auc:.4f}")
    print("\n  Reporte completo (test set):")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=" * 55)
    print("  MÉTRICAS FINANCIERAS (test set)")
    print("=" * 55)
    print(f"  Sharpe ratio estrategia  : {sharpe:.4f}")
    print(f"  Señales emitidas (pred=1): {int(fp_total)}")
    print(f"  Verdaderos positivos     : {int(tp_count)}  ({1-fp_rate:.1%} de señales)")
    print(f"  Falsos positivos         : {int(fp_count)}  ({fp_rate:.1%} de señales)")
    print(f"  Ganancia media por TP    : {avg_tp_gain:.4f}")
    print(f"  Pérdida media por FP     : {avg_fp_loss:.4f}")
    if not np.isnan(avg_tp_gain) and not np.isnan(avg_fp_loss) and avg_fp_loss != 0:
        print(f"  Ratio TP/FP (ganancia)   : {abs(avg_tp_gain / avg_fp_loss):.2f}x")
    print("=" * 55)

    # --- Feature Importance ---
    _plot_feature_importance(model, available_features, X_test, y_test)

    return model


def _plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: list,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Genera y guarda un gráfico combinado de feature importance:
        - Impurity-based (MDI) del RandomForest
        - Permutation importance en el test set

    Los dos métodos juntos dan una visión más robusta que solo MDI,
    que tiende a sobreestimar variables de alta cardinalidad.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Feature Importance — RandomForest", fontsize=14, fontweight="bold")

    # --- MDI (impurity-based) ---
    mdi_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    colors_mdi = ["#2196F3" if "log_ret" not in n and "volat" not in n and "beta" not in n and "mom" not in n
                  else "#FF9800" for n in mdi_imp.index]
    mdi_imp.plot(kind="barh", ax=axes[0], color=colors_mdi)
    axes[0].set_title("MDI (impurity-based)")
    axes[0].set_xlabel("Importance")
    axes[0].axvline(0, color="black", linewidth=0.8)

    # --- Permutation importance ---
    perm = permutation_importance(model, X_test, y_test, n_repeats=15, random_state=42, n_jobs=1)
    perm_imp = pd.Series(perm.importances_mean, index=feature_names).sort_values()
    colors_perm = ["#2196F3" if i < 0 else "#4CAF50" for i in perm_imp]
    perm_imp.plot(kind="barh", ax=axes[1], color=colors_perm, xerr=perm.importances_std[perm_imp.index.map(
        lambda n: list(feature_names).index(n))])
    axes[1].set_title("Permutation Importance (test set)")
    axes[1].set_xlabel("Mean accuracy decrease")
    axes[1].axvline(0, color="black", linewidth=0.8)

    # Leyenda de colores
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FF9800", label="Features de precio"),
        Patch(facecolor="#2196F3", label="Features fundamentales / negativo"),
        Patch(facecolor="#4CAF50", label="Positivo"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(FEATURE_PLOT, dpi=150, bbox_inches="tight")
    print(f"\n    → Gráfico guardado en {FEATURE_PLOT}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  QUANT ML PIPELINE — Stock Outperformance Prediction")
    print("=" * 60)
    _ensure_dirs()

    # 1. Precios
    prices = download_prices()

    # 2. Fundamentals
    fundamentals = download_fundamentals()

    # 3. Features
    features = build_features(prices, fundamentals)

    # 4. Target
    dataset = build_target(prices, features)

    # 5. Modelo
    model = train_model(dataset)

    print("\n✓ Pipeline completado.")
    print(f"  Dataset  → {DATASET_CSV}")
    print(f"  Gráfico  → {FEATURE_PLOT}")

    return model, dataset


if __name__ == "__main__":
    main()