# Campaña FGM reducida: cuatro transportes × dos solvers

Este es el diseño vigente. Sustituye el barrido de 80 FGM por **8 construcciones**:
cuatro KFLAME y cuatro Cantera, sin repeticiones, para CH4/aire con GRI-Mech 3.0,
300 K, 1 atm y el intervalo de equivalencia 0.7–1.4.

| Transporte | KFLAME | Cantera |
|---|---:|---:|
| Promediado | 1 | 1 |
| Promediado + Soret | 1 | 1 |
| Multicomponente | 1 | 1 |
| Multicomponente + Soret | 1 | 1 |

## Comandos desde la raíz del proyecto

```powershell
# Opcional: revisar los ocho trabajos sin resolver llamas
python benchmarks/benchmark_fgm_solver_comparison.py --dry-run

# Ejecutar o reanudar las ocho construcciones
python benchmarks/benchmark_fgm_solver_comparison.py --output runs/thesis_fgm_comparison --resume

# Postprocesar los archivos guardados, sin resolver llamas
python tools/postprocess_fgm_solver_comparison.py --input runs/thesis_fgm_comparison

# Reunir las figuras y cuadros usados por el capítulo, sin resolver llamas
python tools/assemble_fgm_chapter.py
```

No volver a lanzar los comandos del barrido de presión/temperatura.
Sus perfiles y fallos se conservan en sus directorios originales.

## Carpeta integrada del capítulo

Las cinco figuras en PDF/PNG y todos los cuadros LaTeX están en
`.local/research/thesis/figuras/FGM/`. Su `README.md` identifica la procedencia:
las figuras 01–03 corresponden a la referencia de 44 filas y su validación
independiente; 04–05 corresponden a los ocho FGM adaptativos completados.
`provenance.json` conserva hashes y rutas de los archivos originales.
El capítulo y su apéndice leen directamente de esa carpeta.

La consulta en memoria ya fue medida mediante el ensayo breve
`python tools/assemble_fgm_chapter.py --benchmark-queries`. Sin esa opción,
el ensamblador conserva las observaciones guardadas, por lo que regenerar
figuras y tablas no vuelve a medir tiempos ni a resolver llamas.

## Protocolo

Cada FGM comienza con cinco llamas: phi=0.7, 0.9, 1, 1.1 y 1.4. El refinamiento
selecciona hasta diez puntos medios logarítmicos por ronda, resuelve esas llamas
y reconstruye la tabla, hasta que el indicador interno máximo sea <=1 %.
Se usan 241 nodos adaptativos comunes de progreso. El número final de llamas
se determina por separado para cada solver y transporte; no se impone 44.

Cantera usa **FreeFlame para resolver las ecuaciones**, sus propiedades físicas
y sus estadísticas disponibles. KFLAME usa su solver y sus propiedades nativas.
Ambos comparten el código de definición de progreso, selección de composiciones
y tabulación; esto compara dos construcciones FGM alimentadas por distintos
solvers de llama, no dos bibliotecas de interpolación distintas.

Se conservan beta=Y_CO2+Y_H2O+Y_CO+0.5Y_H2 y la normalización entre extremos
fresco y quemado de cada fila, la misma convención de Z_in y los criterios
espaciales slope=.04, curve=.08, ratio=2.5, prune=.003. El dominio frío inicial
es de .03 m, ampliable durante la resolución. Se solicitan las tolerancias
estacionarias relativas/absolutas 1e-4/1e-9 y transitorias 1e-4/1e-11;
Cantera auto=True puede usar tolerancias de arranque durante sus fases previas.
Se guardan las mallas finales y la configuración efectiva.

KFLAME mantiene el predictor secante nativo con los estados anteriores de la
misma familia. Cantera recibe el perfil previo mediante `set_initial_guess`.
Cada construcción comienza desde cero tras tres llamas de calentamiento,
cuyos perfiles se descartan. El orden KFLAME–Cantera se alterna entre transportes;
se usa un proceso nuevo por FGM, cuatro hilos Numba y uno BLAS/OpenMP.

El tiempo incluye preparación, llamas y propiedades, recuperación, controles
de coordenada, selección y todas las tabulaciones. Excluye calentamiento,
escritura y gráficos. La instrumentación está habilitada y los tiempos
describen este protocolo completo. No son una medida aislada del núcleo Newton.
Se muestra una observación por condición, sin mediana ni intervalo estadístico.

## Datos, reanudación y presentación

Se mantienen manifiesto con hashes, mecanismo copiado, perfiles NPZ originales,
tablas finales e intermedias, rondas de inserción, predictores, mallas,
contadores y tiempos disponibles, logs y registros de fallo. Los campos que
Cantera no expone en este adaptador, como la norma final del paso Newton,
quedan sin dato; no se sustituyen por ceros.

La reanudación omite construcciones completas compatibles, conserva fallos
finalizados y reinicia una construcción interrumpida en otro intento. Un fallo
solo se repite con `--retry-failed`, tras revisarlo. Cambiar código, configuración
o entorno requiere un directorio nuevo. `--case ID` permite seleccionar un FGM
sin modificar el diseño del manifiesto.

El informe genera una figura compacta con tiempos y número de llamas, una tabla
con las ocho configuraciones y costes desglosados, e índices CSV/JSON completos.
Cuando ambos solvers terminan un transporte, compara sus tablas en 101×501
coordenadas comunes (Z_in,c), guardando errores de T, CO2, CO y fuente de progreso.
Esta concordancia entre tablas se distingue de la validación independiente del
caso base ya presentada en la tesis. Las figuras no presentan datos pendientes
como resultados disponibles.

## Causa del fallo en el barrido anterior

Los tres fallos examinados a 2 atm (promediado ±Soret y multicomponente)
ocurrieron en phi=1. La resolución de la llama estaba aceptada, pero el control
de la coordenada detectó c_max entre 1.00002767 y 1.00002901 y retrocesos locales
de aproximadamente 3.1e-6–3.4e-6. El máximo se sitúa a 40.5–42 mm en un dominio
de 60 mm, en una zona quemada donde la liberación de calor es aproximadamente
una millonésima de su máximo. Por tanto, el bloqueo inmediato proviene del
control estricto de monotonía/límites del progreso original, no de un fallo
de convergencia de esas llamas.

Su magnitud pequeña no permite atribuir por sí sola el retroceso a discretización,
dominio o física de la mezcla. No se ha relajado el umbral ni recortado el perfil
para ocultarlo. La campaña reducida se circunscribe a 1 atm y vuelve a aplicar
el mismo control. El informe de diagnóstico conserva los valores por caso en
`runs/fgm_failure_review/failure_audit.json`.
