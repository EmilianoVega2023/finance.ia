
Sobre las dependencias pagas — tenés razón en que hay mucho ruido en ese mundo, pero también hay alternativas realmente buenas y gratuitas. yfinance para datos de mercado, scikit-learn para el Random Forest que ya conocés, y para la parte de agente LangChain tiene una versión open source que no requiere APIs pagas si la conectás con un modelo local o con la API gratuita de Groq por ejemplo. No es lo mismo que GPT-4 pero para análisis financiero estructurado alcanza.

-------------------------------------------------

DIA 1

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

DIA 2

=======================================================
  RESULTADOS DEL MODELO
=======================================================
  Umbral de probabilidad : 0.6
  Accuracy               : 0.5691
  Precision (macro)      : 0.5479
  Recall    (macro)      : 0.5349

  Reporte completo (test set):
              precision    recall  f1-score   support

           0       0.59      0.80      0.68      2119
           1       0.51      0.27      0.36      1629

    accuracy                           0.57      3748
   macro avg       0.55      0.53      0.52      3748
weighted avg       0.55      0.57      0.54      3748

=======================================================
  MÉTRICAS FINANCIERAS (test set)
=======================================================
  Sharpe ratio estrategia  : 0.1564
  MÉTRICAS FINANCIERAS (test set)
=======================================================
  Sharpe ratio estrategia  : 0.1564
=======================================================
  Sharpe ratio estrategia  : 0.1564
  Sharpe ratio estrategia  : 0.1564
  Señales emitidas (pred=1): 876
  Señales emitidas (pred=1): 876
  Falsos positivos         : 360  (41.1% de señales)
  Pérdida media por FP     : -0.0156
  Falsos positivos         : 360  (41.1% de señales)
  Pérdida media por FP     : -0.0156
=======================================================

DIA 3

=======================================================
  RESULTADOS DEL MODELO
=======================================================
  Umbral de probabilidad : 0.6
  Accuracy               : 0.5923
  Precision (macro)      : 0.4856
  Recall    (macro)      : 0.4957

  Reporte completo (test set):
              precision    recall  f1-score   support

           0       0.61      0.92      0.73      5914
           1       0.36      0.07      0.12      3724

    accuracy                           0.59      9638
   macro avg       0.49      0.50      0.43      9638
weighted avg       0.51      0.59      0.50      9638

=======================================================
  MÉTRICAS FINANCIERAS (test set)
=======================================================
  Sharpe ratio estrategia  : -0.0092
  Señales emitidas (pred=1): 731
  Falsos positivos         : 358  (49.0% de señales)
  Pérdida media por FP     : -0.0032
=======================================================

DIA 4

=======================================================
  RESULTADOS DEL MODELO
=======================================================
  Umbral de probabilidad : 0.6
  Accuracy               : 0.6286
  Precision (macro)      : 0.4864
  Recall    (macro)      : 0.4951

  Reporte completo (test set):
              precision    recall  f1-score   support

           0       0.66      0.91      0.76      6380
           1       0.31      0.08      0.13      3261

    accuracy                           0.63      9641
   macro avg       0.49      0.50      0.45      9641
weighted avg       0.54      0.63      0.55      9641

=======================================================
  MÉTRICAS FINANCIERAS (test set)
=======================================================
  Sharpe ratio estrategia  : -0.0304
  Señales emitidas (pred=1): 858
  Falsos positivos         : 418  (48.7% de señales)
  Pérdida media por FP     : -0.0041
=======================================================
