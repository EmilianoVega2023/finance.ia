"""
Diagnóstico de fundamentals disponibles en yfinance.
Ejecutar con: python yf_diagnostic.py

Muestra qué datos reales devuelve yfinance para cada ticker
antes de reescribir download_fundamentals().
"""

import yfinance as yf
import pandas as pd

TICKERS = ["YPF", "GGAL", "MSFT", "GOLD"]

for ticker in TICKERS:
    print(f"\n{'='*60}")
    print(f"  {ticker}")
    print(f"{'='*60}")

    t = yf.Ticker(ticker)

    # --- Income Statement ---
    try:
        inc = t.financials  # anual, filas = métricas, columnas = fechas
        if inc is not None and not inc.empty:
            print(f"\n  Income Statement — filas disponibles:")
            for row in inc.index:
                print(f"    {row}")
        else:
            print("  Income Statement → vacío")
    except Exception as e:
        print(f"  Income Statement → error: {e}")

    # --- Balance Sheet ---
    try:
        bal = t.balance_sheet
        if bal is not None and not bal.empty:
            print(f"\n  Balance Sheet — filas disponibles:")
            for row in bal.index:
                print(f"    {row}")
        else:
            print("  Balance Sheet → vacío")
    except Exception as e:
        print(f"  Balance Sheet → error: {e}")

    # --- Cash Flow ---
    try:
        cf = t.cashflow
        if cf is not None and not cf.empty:
            print(f"\n  Cash Flow — filas disponibles:")
            for row in cf.index:
                print(f"    {row}")
        else:
            print("  Cash Flow → vacío")
    except Exception as e:
        print(f"  Cash Flow → error: {e}")

    # --- Info (market cap, etc.) ---
    try:
        info = t.info
        keys_we_need = ["marketCap", "trailingEps", "bookValue", "returnOnEquity"]
        print(f"\n  Info — campos relevantes:")
        for k in keys_we_need:
            print(f"    {k:<30} {info.get(k, 'N/A')}")
    except Exception as e:
        print(f"  Info → error: {e}")
