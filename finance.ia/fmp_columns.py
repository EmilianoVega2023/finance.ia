"""
Muestra todas las keys disponibles en cada endpoint que funciona.
Ejecutar con: python fmp_columns.py
"""

import requests
import json

API_KEY = "saHv6pVcfJ7jpsqSle8U50fR61AUIHc7"
TICKER  = "MSFT"
BASE    = "https://financialmodelingprep.com/stable"

endpoints = [
    "/income-statement",
    "/balance-sheet-statement",
    "/key-metrics",
]

for ep in endpoints:
    r = requests.get(
        BASE + ep,
        params={"symbol": TICKER, "period": "annual", "limit": 1, "apikey": API_KEY},
        timeout=10
    )
    data = r.json()
    if isinstance(data, list) and data:
        print(f"\n{'='*60}")
        print(f"  {ep}")
        print(f"{'='*60}")
        for k, v in data[0].items():
            print(f"  {k:<45} {v}")
