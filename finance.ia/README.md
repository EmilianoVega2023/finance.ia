solucione el problema con la api
cambie fmp por yfinance
ahora queda sacar bitcoin para que el modelo no tenga ruido
=======================================================
  PRIMEROS RESULTADOS DEL MODELO
=======================================================
  Accuracy  : 0.4362
  Precision : 0.4232  (macro)
  Recall    : 0.4255  (macro)

  Reporte completo (test set):
              precision    recall  f1-score   support

           0       0.49      0.53      0.51      2170
           1       0.36      0.32      0.34      1778

    accuracy                           0.44      3948
   macro avg       0.42      0.43      0.42      3948
weighted avg       0.43      0.44      0.43      3948

=======================================================
EL ACCUARACY TIENE QUE ESTAR EN 0,53 (suba de mercado realista)
Depende de una sola cosa:

baseline de 0,43 (primera)
ahora 0,55

Si tu target está distribuido así:

1 (sube) = 53%
0 (baja) = 47%

Entonces un modelo que siempre prediga “sube” ya tiene 0.53 de accuracy.

Entonces:

0.43 → peor que el mercado
0.55 → apenas mejor que el mercado
0.60+ → empieza a ser interesante
0.65+ → ya es fuerte (si está bien validado)

En NumPy:

calcular retornos diarios.
Guardarlos en un vector.

Calcular:

media
desviación estándar
producto punto entre retornos de MSFT y SPY

Interpretación:
Si el producto punto es alto → se mueven parecido.
Ahí entendés correlación geométrica.
No es matemática abstracta.
Es riesgo de mercado.

Objetivo del día:
Entender cómo crecen los ingresos y el flujo de caja en los últimos 5 años.

🔹 Sobre tu idea tipo Buffett

Warren Buffett no usa magia.
Usa principios:

Ventaja competitiva duradera
ROE alto sostenido
Baja deuda relativa
Flujo de caja predecible
Buen management
Eso se puede cuantificar.

De hecho, lo que estás imaginando se llama:
Value Investing Quantitativo.

Se puede modelar con:

Screener fundamental
Scoring multi-factor
Descuento de flujos (DCF)
Clasificación con ML

Y acá aparece algo potente:

Podrías construir un modelo que:
Lea estados financieros
Extraiga métricas clave
Asigne score de “calidad tipo Buffett”

🔹 Pero ordenemos prioridades

Primero:
Base matemática + manipulación de datos.

Después:
Automatización de métricas fundamentales.

Después:
Modelo predictivo o de scoring.

No al revés.

--------------------------

Día 2 — Vectores en la práctica (álgebra aplicada)

Conceptos:

Qué es un vector.
Producto punto.
Norma.

Implementación:

Crear dos arrays en NumPy.
Calcular producto punto.
Calcular magnitud.

Ejemplo mental:
Si dos activos se mueven parecido → producto punto alto.
Eso ya conecta matemática con finanzas.

Día 3 — Datos reales de mercado

Usá yfinance.
Descargá datos históricos de:
SPY
TSLA
AMZN
MSFT
GALL
YPF

Calcular:

Retornos diarios.
Media.
Volatilidad.

Día 4 — Visualización

Graficar:

Precio histórico.
Retornos.
Media móvil simple.

Entender:

Tendencia.
Ruido.
Variabilidad.
La matemática empieza a tener cara.

Día 5 — Mini proyecto integrador

Construí algo simple pero completo:

“Análisis comparativo de activos”

Descargar 3 activos.

Calcular rendimiento anualizado.
Calcular volatilidad.
Graficar.
Imprimir ranking.
Eso es tu primer output serio.
Subilo a GitHub con:
Código limpio.
README explicando qué hace.
