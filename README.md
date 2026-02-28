DIA 2 
mejore el accuracy de 0,43 a 0,55
ajuste los testeos del modelo
hice rules.py para tener un sistema mas solido sin aprendizaje en los resultados

DIA 3
mejore el accuracy de 0,55 a 0,63 pero tenia un sesgo conservador (dejaba de operar para acertar mas) y el ratio de beneficio cayo drastico
lo mejore y ahora el ratio a mirar es TP/FP que actualmente esta en 1.5x y lo tengo que llevar a 2x

DIA 4

sacar roe, yield, mejorar return long. 
anclarme a beta60 y volatility60 para aumentar el ratio TP/FP y accuracy.
generar variables nuevas entre correlacion con el mercado y riesgo sistematico.
conectar mi modelo con un agente ya existente en repositorio de github.


Beta mide cuánto se mueve un activo en relación al mercado (SPY) cuando el mercado se mueve. Es el coeficiente de sensibilidad al riesgo sistémico.

β = 1.0 → el activo se mueve igual que el mercado
β = 1.5 → si SPY sube 10%, el activo sube 15% en promedio. Si SPY cae 10%, cae 15%
β = 0.5 → activo defensivo, se mueve la mitad que el mercado
β < 0 → activo que se mueve en contra del mercado (ej. oro en crisis)

primera prueba bajo el ratio y no creo el output
ahora por conectar los agentes de ia con el main anterior
