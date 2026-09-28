# Ampliación L3: cuatro transportes en presión y temperatura

Ejecutar los comandos desde la raíz del proyecto. La comprobación inicial ya
se realizó durante la preparación; puede repetirse sin escribir ni simular:

```powershell
python benchmarks/extend_flame_transport_sweeps.py --campaign runs/thesis_flames_L3 --dry-run
```

Iniciar o reanudar:

```powershell
python benchmarks/extend_flame_transport_sweeps.py --campaign runs/thesis_flames_L3 --resume
```

## Diseño

| Barrido | Estados nuevos por mecanismo | Transportes añadidos | Condiciones nuevas |
|---|---|---|---:|
| Presión | 2, 3, 5, 10 atm; phi=1; 300 K | Los tres ausentes en el diseño original | 24 |
| Temperatura | 350, 400, 450, 500 K; phi=1; 1 atm | Los tres ausentes en el diseño original | 24 |

En CH4/GRI-Mech 3.0 se añaden promediado + Soret, multicomponente y multicomponente
+ Soret; en H2/h2o2 se añaden promediado, promediado + Soret y multicomponente.
Los ocho estados centrales, así como los barridos originalmente disponibles,
se reutilizan. Resultado: 52 condiciones por mecanismo, 104 en total.

Hay **48 condiciones nuevas, cinco pares por condición: 480 resoluciones medidas**,
240 por solver, además de un calentamiento independiente antes de cada resolución.
El total combinado previsto es de 1040 resoluciones medidas. No se ejecuta otro
estudio de sensibilidad como requisito.

Se mantienen L3 (slope=0.01, curve=0.02, ratio=2.5, prune=0), dominio inicial de
0.03 m, tolerancias, límite temporal y aceptación del protocolo original.
CH4 usa Newton/PTC-SER con rescate Euler; H2 usa Newton/Euler implícito.
Se alterna el orden de los solvers, con cuatro hilos Numba y uno BLAS/OpenMP.
Ejecutar una sola campaña a la vez para conservar la comparación temporal.

## Guardado y reanudación

- El manifiesto original se conserva; `transport_extension.json` documenta las
  condiciones adicionales y vincula sus hashes al código, entorno y manifiesto base.
- Los nuevos casos se guardan en `main/<case_id>/pair-XX/attempt-XXX/`, con los
  mismos NPZ, diagnósticos, registros, solicitudes y logs del ejecutor original.
- `transport_extension_progress.json` informa avance y pares no utilizables.
- `Ctrl+C` interrumpe la ejecución; el mismo comando con `--resume` reinicia el
  par interrumpido y conserva su intento anterior.
- Los pares completos se omiten, incluidos los fallidos. Un fallo completado
  requiere revisión de su causa; no se repite silenciosamente.
- Se comprueba compatibilidad de código/entorno y la integridad de los perfiles
  reutilizados. Una discrepancia debe investigarse antes de mezclar resultados.

## Postprocesado sin resolver llamas

Al terminar, ejecutar:

```powershell
python tools/postprocess_extended_flames.py --input runs/thesis_flames_L3 --output runs/thesis_flames_L3/report_expanded
```

También puede ejecutarse con la campaña incompleta. Guarda CSV/JSON, tablas y
`pendientes.csv`; las figuras finales esperan los cinco pares utilizables por
condición. Para producir solo las estadísticas puede añadirse `--tables-only`.
Las sesiones base/ampliación se identifican en `ejecuciones.csv`.

Cuando se completa la campaña genera en `report_expanded/revision_figures`:

- `13_respuesta_fisica` y `14_speedup_global`: cuatro transportes en los tres barridos.
- `09_tiempos_transportes`, `09_tiempos_presion` y `09_tiempos_temperatura`: cuatro
  columnas de transporte y dos filas de mecanismos, cinco tiempos y cuartiles.
- `12_concordancia_global` y `15_precision_speedup`: las 104 condiciones.
- Perfiles, mallas, efectos de transporte, costes y ablaciones centrales reutilizados.
- `revision_graficos.pdf`: conjunto de figuras vectoriales; PDF, PNG y pies TeX individuales.

Para compilar el conjunto con pies:

```powershell
Push-Location runs/thesis_flames_L3/report_expanded/revision_figures
pdflatex -interaction=nonstopmode -halt-on-error revision_con_pies.tex
pdflatex -interaction=nonstopmode -halt-on-error revision_con_pies.tex
Pop-Location
```

El informe original y el capítulo permanecen disponibles para contrastar con
la ampliación antes de integrar sus nuevas interpretaciones.

## Alcance de los controles

Perfiles y coste ya cubren los cuatro transportes de ambos mecanismos a phi=1,
1 atm y 300 K. Las mallas de esos mismos ocho casos también están guardadas.
Las ablaciones estudian las dos optimizaciones previstas. Por tanto, no hace
falta repetir esas campañas para completar las curvas de presión y temperatura.
La sensibilidad espacial existente comprende seis estados y los transportes
de referencia; su conclusión mantiene ese alcance. Si los nuevos casos muestran
discrepancias o fallos destacados, se seleccionarán esos estados para controles
dirigidos después de revisar los resultados.

La comparación temporal es pareada dentro de cada condición. La ampliación se
ejecuta en otra sesión: conservar equipo, alimentación y carga de trabajo permite
interpretar mejor las tendencias entre condiciones, sin equiparar las repeticiones
temporales con incertidumbre de discretización.
