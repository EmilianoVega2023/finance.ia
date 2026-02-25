# finance.ia
learning

Día 1 — Vectores aplicados a rendimientos

Duración: 1.5 h

Parte 1 (30 min)

Repasar conceptualmente:

Vector = lista ordenada de números.

Retornos diarios = vector.

Portafolio = combinación lineal de vectores.

Eso es álgebra lineal aplicada a finanzas.

Parte 2 (1 h práctica)

En NumPy:

Descargar precios de AAPL.

Calcular retornos diarios.

Guardarlos en un vector.

Calcular:

media

desviación estándar

producto punto entre retornos de AAPL y SPY

Interpretación:
Si el producto punto es alto → se mueven parecido.

Ahí entendés correlación geométrica.

No es matemática abstracta.
Es riesgo de mercado.

Día 2 — Introducción real a balances

Acá empezamos a construir ventaja competitiva.

Leer un balance no es “saber contabilidad”.
Es entender:

Revenue (ingresos)

Net Income (ganancia neta)

Free Cash Flow

Debt

ROE

Margen operativo

Te recomiendo empezar con una empresa conocida:

Apple Inc.

Buscar su:

Income Statement

Balance Sheet

Cash Flow Statement

Objetivo del día:
Entender cómo crecen los ingresos y el flujo de caja en los últimos 5 años.

Nada más.

🔹 Sobre tu idea tipo Buffett

Importante.

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

Eso sí es un proyecto con identidad.

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

AAPL

BTC-USD

Calcular:

Retornos diarios.

Media.

Volatilidad.

Acá empezás a pensar en términos de datos reales.

Nada abstracto.

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
