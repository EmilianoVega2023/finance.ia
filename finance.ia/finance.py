import yfinance as yf
import numpy as np
import pandas as pd

# 1) Descargar
data = yf.download(["MSFT","SPY"], start="2015-01-01")

# 2) Quedarse con cierres
prices = data["Close"]

# 3) Retornos log
returns = np.log(prices / prices.shift(1)).dropna()

# 4) Extraer vectores si querés
r_msft = returns["MSFT"].values
r_spy  = returns["SPY"].values
np.dot(r_msft, r_spy)
cov_matrix = returns.cov().values

w = np.array([0.6, 0.4])

portfolio_variance = w.T @ cov_matrix @ w
print("Varianza del portafolio:", portfolio_variance)
returns.shape

