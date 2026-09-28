# Bloque FGM: diseño, ejecución y lectura

La campaña vigente es **cuatro transportes × dos solvers = ocho FGM, sin repeticiones**.
Su ejecutor adaptativo y los comandos están en
[thesis_fgm_solver_comparison.md](thesis_fgm_solver_comparison.md).
Este documento conserva el protocolo anterior del caso base, utilizado para
comprobar la variable de progreso y la fidelidad con llamas independientes.

## Decisiones de presentación

El bloque sigue familia → manifold → fidelidad → coste. Incluye cuatro
figuras y dos cuadros principales. El control de resolución se conserva
como un cuadro de apéndice. El mapa aprobado `fgm_mapa_adaptativo.pdf`
mantiene su composición: CO2, calor liberado, CO y fuente de progreso.
La imagen histórica no se sobrescribe; la nueva se genera con la primera
familia completa de la campaña, identificada en su pie.

La auditoría de coordenadas se conserva también en el apéndice, en un cuadro
compacto y en `coordinate_audit.csv/json`. `analysis_coordinates.tex` incorpora
al cuerpo los conteos y el peor cambio en tramos casi constantes, indicando
cuántos perfiles y composiciones están disponibles. Se revisan los perfiles originales
de todas las repeticiones completas y de las llamas retenidas, antes de ordenar
o agrupar muestras de progreso: monotonía y límites de c, denominador de
normalización, separación de Zin y variación de T, todas las especies y omega_c
en tramos con |delta c| <= 1e-10. Esa última magnitud permite distinguir un
extremo uniforme de una zona donde cambian los campos con progreso casi fijo;
es un diagnóstico, no una prueba de unicidad global. Los retrocesos mayores
que 1e-7 o valores fuera de [0,1] con ese margen detienen el informe de fidelidad.

Zin y c se almacenan como ejes; Zstar es su coordenada gráfica reescalada.
Los campos actuales son T, u, rho, cp_mass, conductivity, qdot, omega_c, beta
y las fracciones másicas de todas las especies. No se atribuye al archivo el
almacenamiento de h, mu o D_k, que no son campos de esta tabla.

Referencias examinadas y uso en este diseño:

- [van Oijen y de Goey (2000)](https://doi.org/10.1080/00102200008935814):
  construcción a partir de llamas 1D y representación de estados
  termoquímicos. Motiva mostrar la familia y sus coordenadas antes del coste.
- [Ramaekers et al. (2010)](https://doi.org/10.1007/s10494-009-9223-1):
  evaluación a priori de manifolds y especies. Motiva verificar fidelidad
  sobre referencias independientes y explicitar las condiciones físicas.
- [van Oijen et al. (2016)](https://doi.org/10.1016/j.pecs.2016.07.001):
  revisión del enfoque FGM y sus aplicaciones, ya incluida en el marco teórico.
- [Chi, Xu y Thévenin (2022)](https://doi.org/10.1016/j.combustflame.2022.112325):
  distingue pruebas a priori y a posteriori de representaciones del manifold.
  Nuestro estudio se limita a la primera; el coste de consulta no demuestra
  por sí mismo aceleración de una simulación CFD completa.
- [Gupta, Teerling y van Oijen (2021)](https://doi.org/10.1080/13647830.2021.1926544):
  estudia la influencia de la definición de progreso en la velocidad de combustión.
  Justifica examinar esa elección; los pesos concretos de nuestra combinación
  pertenecen a la implementación y no se atribuyen a este artículo.

La elección de cuatro figuras, 44 composiciones y 12 retenciones es propia
de esta tesis, no una exigencia de esos artículos. No se repite la campaña
KFLAME–Cantera ni se añade un punto de amortización sin tareas equivalentes.

## Comparación de variables de progreso

El postprocesado añade un cuadro de apéndice, sin nuevas llamas ni figuras,
con cuatro combinaciones de fracciones másicas: CO2; CO2+H2O;
CO2+H2O+CO+H2; y la actual CO2+H2O+CO+0.5H2. Se utiliza la primera familia
completa, identificada, sin combinar repeticiones de las mismas composiciones.

- Primero se comprueban retrocesos, límites y denominadores sobre perfiles
  originales. Una candidata inválida queda sin reconstrucción; ordenar sus
  muestras no se utiliza para salvar el diagnóstico.
- Se comparan gradientes normalizados y variaciones en tramos casi planos.
  Campos comunes: T, YCO2, YCO, YOH y qdot. Las fuentes químicas de candidatos
  alternativos no se deducen de omega_c, pues sus pesos cambian.
- Se retira cada fila interior (42 para la familia completa) y se reconstruye
  desde las restantes. El indicador adaptativo y sus 241 nodos se calculan de
  nuevo sin la fila retirada. Se comparan los mismos 1001 estados físicos por
  fila, uniformes en el progreso original normalizado, para todas las candidatas.
  Las escalas se obtienen exclusivamente de las filas restantes, con un mínimo
  de 1e-5 para especies. Se informa el peor P95 entre campos.
- El muestreo de composiciones fue refinado previamente con la definición
  actual. Esta comprobación interna caracteriza ese conjunto y no demuestra
  optimalidad universal. No selecciona pesos automáticamente. Las 12 retenidas
  se conservan para la evaluación final del protocolo fijado; utilizarlas para
  ajustar los pesos convertiría ese conjunto en datos de selección.

`progress_candidates.json`, sus CSV de resumen/perfil/fila retirada y
`table_progress_candidates.tex` permiten revisar todo el cálculo.
`analysis_progress.tex` incorpora un resumen breve al capítulo. Los comandos
de campaña y postprocesado permanecen iguales.

Se verificó el procedimiento con los 44 perfiles históricos del mapa de
septiembre de 2026, conservando ese diagnóstico en
`runs/fgm_progress_historical_review`. Es un estudio previo separado de la
nueva campaña; no se incorpora como si fueran resultados nuevos.

La tesis de [Rastogi (2022), sección 2.5](https://pure.tue.nl/ws/portalfiles/portal/218079023/1333836_MasterThesis.pdf)
documenta los criterios de monotonía y gradiente y usa H2O+10HO2 para su
problema de hidrógeno con tres corrientes. Esa combinación pertenece a aquella
configuración y no se traslada al metano de esta tesis.

## Campaña preparada

La primera etapa es la [comprobación de progreso](fgm_progress_preflight.md):
una sola familia y las 12 llamas independientes. La opción
`--phase all --family-repetition 1` selecciona esos dos trabajos sin modificar
el diseño de cinco repeticiones; las otras cuatro pueden completarse después
con los comandos habituales y `--resume`.

La ampliación acordada estudia temperatura y presión, con los cuatro
transportes en cada estado. La construcción completa debe partir de las
cinco composiciones iniciales y repetir su refinamiento adaptativo, con
el mismo criterio y 241 nodos de progreso. Las 44 filas son el resultado
del refinamiento del caso previo, no un número impuesto a cada condición.
La ejecución de esa ampliación se prepara después de revisar el progreso
en el caso base. El estudio 11/22/44 queda fuera de la campaña principal.

- CH4/GRI-Mech 3.0; aire molar O2:N2=1:3.76; 300 K; 101325 Pa.
- Transporte promediado sin Soret, phi de 0.7 a 1.4.
- Calendario existente de 44 composiciones: 5 solicitadas y 39 puentes.
  Se conserva fijo para separar variación temporal de selección de filas.
  Su criterio histórico de selección fue un defecto leave-one-out del 1 %;
  no es una garantía del error independiente de la nueva campaña.
- Cinco familias completas: 220 resoluciones principales.
- Doce llamas de retención, una resolución independiente por composición,
  situadas en puntos medios logarítmicos de 12 intervalos distribuidos por
  el calendario. Se excluyen de tabla, indicador y decisiones de refinamiento.
- Dos calentamientos por proceso; seis procesos previstos = 12 llamas extra.
  Total inicial previsto: 244 resoluciones incluyendo calentamiento.
- Malla FGM del ejemplo de referencia: slope=.04, curve=.08, ratio=2.5,
  prune=.003, hasta 1600 nodos y 300 s por llama. Es distinta de L3.
- 241 nodos de progreso; controles 121 y 481 por postprocesado de los mismos
  perfiles. Esto controla c, no el error espacial ni la discretización en Z.
- Procesos secuenciales, cuatro hilos Numba y uno BLAS/OpenMP; perfiles de
  continuación solo dentro de una familia. Cada familia se inicia en frío
  después del calentamiento, con el registro interno de operaciones activado.

## Comandos PowerShell desde la raíz

```powershell
# Revisar el diseño y dependencias, sin ejecutar ni escribir resultados
python benchmarks/benchmark_fgm_campaign.py --output runs/thesis_fgm --dry-run

# Cinco repeticiones de la familia de 44 llamas
python benchmarks/benchmark_fgm_campaign.py --phase family --output runs/thesis_fgm --resume

# Doce llamas independientes excluidas de la tabla
python benchmarks/benchmark_fgm_campaign.py --phase holdout --output runs/thesis_fgm --resume

# Figuras, errores, sensibilidad y cinco mediciones de consulta por tamaño de lote
python tools/postprocess_fgm_campaign.py --input runs/thesis_fgm --benchmark-queries

# Regenerar posteriormente las figuras con tiempos ya guardados, sin medir de nuevo
python tools/postprocess_fgm_campaign.py --input runs/thesis_fgm

# Compilar la tesis desde su carpeta
Push-Location .local/research/thesis
pdflatex -interaction=nonstopmode -halt-on-error TFM_FGM_FINAL.tex
bibtex TFM_FGM_FINAL
pdflatex -interaction=nonstopmode -halt-on-error TFM_FGM_FINAL.tex
pdflatex -interaction=nonstopmode -halt-on-error TFM_FGM_FINAL.tex
Pop-Location
```

La nueva sección ya está incluida en `main.tex` y `TFM_FGM_FINAL.tex`.
`resultados_FGM.tex` compila el bloque por separado. Los pies y cuadros se
incorporan automáticamente desde `runs/thesis_fgm/report`. El informe
incompleto identifica qué falta y mantiene el balance de costes pendiente
hasta disponer de las cinco familias y de las consultas.

## Persistencia y reanudación

Cada llama guarda su perfil NPZ completo y un registro de trayectoria:
coordenadas, composición, propiedades, fuentes, estado de aceptación,
tiempo, predictor, diagnóstico de monotonía y contadores disponibles.
Los perfiles y trazas se escriben atómicamente después de cada llama.
La tabla final y los resúmenes se guardan tras completar la familia.

El manifiesto conserva parámetros, versiones, equipo, hilos, calendario,
composiciones de retención, semilla y hashes del código/mecanismos. Se copia
el mecanismo y el calendario. La reanudación verifica compatibilidad y
artefactos de ejecuciones completadas.

Las familias aceptadas se omiten. Una familia interrumpida se repite entera
en otro intento para conservar una trayectoria temporal coherente; los
perfiles anteriores quedan disponibles. Los fallos completados se conservan
y no se repiten silenciosamente. Tras revisar su log, `--retry-failed`
permite repetirlos explícitamente. Un bloqueo residual tras cerrar por fuerza
el proceso se encuentra en `campaign.lock`: comprobar antes que no exista
una ejecución activa y retirar ese archivo únicamente en tal caso.

`runs/fgm_campaign_smoke*` contiene exclusivamente pruebas técnicas y no se
integra en la tesis. No se modifica ni se mezcla la campaña L3 anterior.

## Magnitudes y estadísticos

- Figura 1: tiempo por llama, mediana y cuartiles entre familias; predictor
  de una repetición identificada. No se atribuye causalidad a un speedup que
  no haya sido medido con un control equivalente.
- Figura 2: una familia individual, la misma que se evalúa en fidelidad.
  Zstar es (Zin-Zmin)/(Zmax-Zmin), no el índice de fila. La etiqueta Zin
  conserva la convención histórica de corrientes de referencia en base
  másica del generador, mientras la entrada de aire se define en base molar.
- Figura 3: reconstrucción de T, YCO2, YCO y omega_c para la llama retenida
  que minimiza abs(log(phi)), seleccionada por composición antes de evaluar
  errores. El criterio y la repetición tabulada quedan en `fidelity_selection.json`.
  Se añaden distribución y localización del error en todas las llamas retenidas,
  sobre 1001 puntos uniformes de c por llama. El error es
  `100*abs(predicción-referencia)/escala`; para T la escala es el máximo salto
  térmico retenido, para los otros campos el máximo módulo retenido.
  Cada composición pesa igual. La tabla da media aritmética, P95 y máximo;
  la caja gráfica usa mediana, cuartiles y bigotes P5/P95.
  La coordenada c de cada referencia usa sus propios extremos fresco/quemado;
  es una reconstrucción a coordenadas conocidas. Las escalas de error retenidas
  solo intervienen en la evaluación: ni los nodos, ni el indicador, ni Zin, ni
  los campos de la tabla se construyen con perfiles retenidos.
- Figura 4: fases offline en segundos y consulta online en microsegundos
  por estado. Cinco observaciones, mediana y cuartiles. Consulta lineal
  conjunta de T, YCO2, YCO y omega_c, con interpolador en memoria y calentado;
  comparación de llamadas de uno y 10000 estados. Sin escritura en los
  tiempos. No se compara una consulta con una resolución de llama completa.
  Incluye el total offline: primero se suman las fases de cada repetición y
  luego se calculan mediana y cuartiles. El texto añade microsegundos/estado y
  estados/s; el rendimiento se obtiene por observación antes de resumirlo.
  `timing_summary.json` guarda esas observaciones y estadísticos.

La fuente tabulada es omega_c = sum(a_k*omega_k)/(beta_b-beta_u), con
beta=YCO2+YH2O+YCO+0.5YH2. Su signo se conserva en los archivos y en el
cálculo de errores; solo el mapa aprobado representa su módulo.

Después de ejecutar, revisar los fallos, la monotonía, los extremos de
error, la sensibilidad y los tiempos. La interpretación final se ajustará
a esos datos; no se presupone que el error sea inferior al 1 %.
