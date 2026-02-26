"""
Diagnóstico de endpoints disponibles en tu cuenta FMP.
Ejecutar con: python fmp_diagnostic.py
"""

import requests

API_KEY = "saHv6pVcfJ7jpsqSle8U50fR61AUIHc7"  # reemplazá con tu key si es distinta
TICKER  = "MSFT"

# Lista de URLs candidatas a probar
candidates = [
    # Stable endpoints
    ("stable /income-statement",        f"https://financialmodelingprep.com/stable/income-statement",         {"symbol": TICKER, "period": "annual", "limit": 3}),
    ("stable /balance-sheet-statement", f"https://financialmodelingprep.com/stable/balance-sheet-statement",  {"symbol": TICKER, "period": "annual", "limit": 3}),
    ("stable /key-metrics",             f"https://financialmodelingprep.com/stable/key-metrics",              {"symbol": TICKER, "period": "annual", "limit": 3}),
    ("stable /financial-ratios",        f"https://financialmodelingprep.com/stable/ratios",                   {"symbol": TICKER, "period": "annual", "limit": 3}),

    # v4 endpoints
    ("v4 /income-statement",            f"https://financialmodelingprep.com/api/v4/income-statement",         {"symbol": TICKER, "period": "annual", "limit": 3}),
    ("v4 /key-metrics",                 f"https://financialmodelingprep.com/api/v4/key-metrics",              {"symbol": TICKER, "period": "annual", "limit": 3}),

    # v3 con symbol como param (no en path)
    ("v3 income symbol-param",          f"https://financialmodelingprep.com/api/v3/income-statement",         {"symbol": TICKER, "period": "annual", "limit": 3}),

    # Profile (siempre disponible en casi todos los planes)
    ("stable /profile",                 f"https://financialmodelingprep.com/stable/profile",                  {"symbol": TICKER}),
    ("v3 /profile",                     f"https://financialmodelingprep.com/api/v3/profile/{TICKER}",         {}),
]

print(f"{'Endpoint':<40} {'Status':>8}  {'Resultado'}")
print("-" * 80)

for name, url, params in candidates:
    params["apikey"] = API_KEY
    try:
        r = requests.get(url, params=params, timeout=10)
        status = r.status_code
        if status == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                preview = f"OK — {len(data)} registros, keys: {list(data[0].keys())[:4]}"
            elif isinstance(data, dict) and "Error Message" in data:
                preview = f"ERROR FMP: {data['Error Message'][:60]}"
            else:
                preview = f"Respuesta vacía o inesperada: {str(data)[:60]}"
        else:
            preview = r.text[:80]
    except Exception as e:
        status = "ERR"
        preview = str(e)[:80]

    print(f"{name:<40} {str(status):>8}  {preview}")
