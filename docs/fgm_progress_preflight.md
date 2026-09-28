# Comprobación previa de la variable de progreso

Primero se comprueba el FGM de referencia CH4/GRI-Mech 3.0, 300 K, 1 atm,
transporte promediado sin Soret, con 44 composiciones y 241 nodos de progreso.
La definición fijada es `CO2 + H2O + CO + 0.5 H2`, en fracciones másicas.
Esta etapa usa una familia completa y doce llamas independientes: 56
resoluciones científicas y cuatro calentamientos previstos. Las repeticiones
temporales restantes se ejecutarán después de revisar la representación.

## Ejecutar desde la raíz del proyecto

```powershell
python benchmarks/benchmark_fgm_campaign.py --phase all --family-repetition 1 --output runs/thesis_fgm --resume
python tools/postprocess_fgm_campaign.py --input runs/thesis_fgm
```

El primer comando resuelve 44 llamas mediante continuación y doce llamas
independientes. El segundo tabula, compara las cuatro definiciones sobre
construcción, evalúa la definición fijada sobre retención y genera figuras
y cuadros. El informe admite que solo haya una de las cinco familias.

Consultar `runs/thesis_fgm/report/progress_review.md`,
`coordinate_audit.csv`, `progress_candidates.json`, `error_metrics.csv`,
`table_resolution_sensitivity.csv` y `03_fidelidad.pdf`.
Las conclusiones de coste con cinco repeticiones permanecen pendientes.

## Qué se debe decidir

1. Verificar que la coordenada distingue estados: denominador de
   normalización, retrocesos, límites y cambios de campos en tramos planos.
2. Comparar definiciones con las filas de construcción. La reconstrucción
   interna retira una fila cada vez y no utiliza las doce retenidas.
3. Evaluar temperatura, CO2, CO y fuente del progreso fijado con esas doce
   llamas. Revisar media, P95, máximo y localización de errores juntos.
4. Comparar 121, 241 y 481 nodos sobre los mismos perfiles. La sensibilidad
   permite distinguir parte del efecto de resolución en c; una discrepancia
   restante puede requerir revisar las filas en composición o la coordenada.

No hay un umbral universal de gradiente que certifique una coordenada.
La conclusión debe referirse a las magnitudes y condiciones evaluadas.
Si las retenidas motivan cambiar los pesos, pasan a ser datos de desarrollo:
habrá que reservar nuevas composiciones independientes para el control final.

## Evidencia previa disponible

Diagnóstico de 44 perfiles históricos guardados en
`.local/research/FGM/resultados/revision_20260908/v2_adaptive_zc_1pct_final`:

| Combinación másica | Llamas con retrocesos | Peor P95 interno [%] | Máximo puntual interno [%] |
|---|---:|---:|---:|
| CO2 | 19 | No admisible | No admisible |
| CO2 + H2O | 10 | No admisible | No admisible |
| CO2 + H2O + CO + H2 | 0 | 0.433 | 3.444 |
| CO2 + H2O + CO + 0.5 H2 | 0 | 0.436 | 3.537 |

Los errores son normalizados, con cinco campos comunes T, YCO2, YCO, YOH
y qdot; P95 es el mayor entre campos y se evalúan 42 filas interiores.
Los máximos muestran que P95 no acota todos los estados. Ambas combinaciones
de cuatro especies son viables en esta comprobación y dan errores cercanos.
Estos datos apoyan mantener la definición actual para el ensayo independiente,
sin atribuirle optimalidad. Procedencia, escalas y resultados por fila:
`runs/fgm_progress_historical_review/progress_candidates.json`.

## Ampliación de rendimiento acordada

El diseño actual es **cuatro transportes × dos solvers = ocho FGM, sin repeticiones**.
Sustituye el barrido de presión y temperatura, detenido tras detectar
retrocesos pequeños de la coordenada de progreso en los estados de 2 atm.
Los estados, comandos y archivos se describen en
[thesis_fgm_solver_comparison.md](thesis_fgm_solver_comparison.md).

Cada FGM parte de cinco composiciones y ejecuta el refinamiento completo.
Se conservan la continuación nativa y el predictor secante dentro de la
construcción. Las 44 filas son el resultado del estudio previo; el nuevo
número de filas depende de cada condición. La comparación comunica tiempo
individual, coste por etapa y tamaño final de la familia. La aceptación del
progreso en el caso base se vuelve a comprobar en cada condición antes de tabular.
