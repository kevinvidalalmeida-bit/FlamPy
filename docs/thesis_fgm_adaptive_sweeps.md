# Campaña FGM adaptativa: 20 estados × 4 transportes

**Campaña detenida y sustituida.** El diseño vigente es de ocho FGM en el caso
base, cuatro por solver: [thesis_fgm_solver_comparison.md](thesis_fgm_solver_comparison.md).
Los comandos siguientes documentan el barrido anterior y no son los de la campaña vigente.

La campaña actual contiene **80 FGM de CH4/aire con GRI-Mech 3.0**, con una
construcción por condición, sin repeticiones estadísticas. Se conserva el
generador nativo, la continuación y el indicador de refinamiento existentes.

| Grupo | Temperatura [K] | Presión [atm] | Estados nuevos |
|---|---|---|---:|
| Base | 300 | 1 | 1 |
| Presión | 300 | 2, 3, 4, 5, 6, 7, 8, 9, 10 | 9 |
| Temperatura | 340, 380, 420, 460, 500, 540, 580, 620, 660, 700 | 1 | 10 |

Cada estado se resuelve con promediado, promediado + Soret, multicomponente y
multicomponente + Soret. Los barridos comparten el caso base y permiten observar
la respuesta separada a presión y temperatura. No constituyen un factorial T–p.

## Ejecución desde la raíz del proyecto

```powershell
# Opcional: comprobar el diseño, sin resolver ni escribir resultados
python benchmarks/benchmark_fgm_adaptive_campaign.py --output runs/thesis_fgm_sweeps_max10 --dry-run

# Ejecutar las 80 construcciones; el mismo comando reanuda una interrupción
python benchmarks/benchmark_fgm_adaptive_campaign.py --output runs/thesis_fgm_sweeps_max10 --resume

# Regenerar figura, tablas e índices sin resolver llamas, también con campaña parcial
python tools/postprocess_fgm_adaptive_campaign.py --input runs/thesis_fgm_sweeps_max10
```

El protocolo vigente permite hasta diez inserciones por ronda. Se usa
`runs/thesis_fgm_sweeps_max10`, separado del ensayo anterior con cuatro
inserciones (`runs/thesis_fgm_sweeps`), cuyos archivos se conservan.
Cambiar el tamaño de las tandas puede cambiar la selección y el coste adaptativo;
las construcciones anteriores no se mezclan con este protocolo.
También se mantiene separado de `runs/thesis_fgm`, que conserva la
validación anterior de la variable de progreso y las 12 llamas independientes.
Las cuatro opciones de transporte son `P`, `PS`, `M`, `MS` en los identificadores.
Para ejecutar solo un estado físico concreto sin alterar el diseño:

```powershell
python benchmarks/benchmark_fgm_adaptive_campaign.py --output runs/thesis_fgm_sweeps_max10 --resume --case CH4_T300_p10_MS
```

## Qué se construye y qué se mide

Cada FGM comienza con cinco composiciones: 0.7, 0.9, 1, 1.1 y 1.4. Se tabulan
sus perfiles sobre 241 nodos adaptativos comunes de progreso. El indicador
`kflame.fgm.refine._defects` compara cada fila interior con la interpolación
de sus vecinas y combina temperatura, especies, calor liberado y fuente de progreso.
Si algún intervalo supera el 1 %, se insertan hasta diez puntos medios
logarítmicos en los intervalos con mayor indicador y se resuelven esas llamas.
Después se reconstruye el eje de progreso y se evalúa nuevamente el indicador.

**El número final de llamas es un resultado adaptativo.** No se impone 44.
Las filas ya calculadas dentro de esa misma construcción se conservan; las nuevas
usan los dos estados resueltos inferiores más próximos cuando están disponibles.
El predictor nativo decide cuándo emplear la secante o la continuación simple.
No se cargan perfiles de una construcción anterior. Las tres llamas de
calentamiento (0.9, 1, 1.1) tampoco se usan para iniciar la familia medida.

Los controles por llama mantienen slope=0.04, curve=0.08, ratio=2.5, prune=0.003,
aceptación nativa y convergencia espacial solicitada. Se comprueba el progreso
original antes de ordenar o tabular sus valores. Se conserva
beta=Y_CO2+Y_H2O+Y_CO+0.5Y_H2, con su normalización por fila.
Un fallo de coordenada se guarda como fallo y permite revisar ese estado;
el éxito del caso base no certifica por anticipado los demás.

Se ejecutan secuencialmente procesos nuevos, con cuatro hilos Numba y uno para
BLAS/OpenMP. El tiempo comunicado suma preparación, resolución con propiedades,
control del progreso, todas las tabulaciones y la selección adaptativa. Incluye
la instrumentación de contadores del solver. Excluye calentamiento, escritura y
postprocesado. `process_wall_s` conserva además el tiempo completo del proceso,
que sí incluye calentamiento y escritura. Hay límites explícitos de 64 rondas
de inserción, 256 filas y 300 s por llama; alcanzarlos sin cumplir el objetivo
produce un fallo documentado, nunca una tabla aceptada silenciosamente.

Los tiempos son **individuales**, sin mediana, cuartiles ni incertidumbre de
repetición. Cambiar el estado puede cambiar tanto el coste por llama como el
número de filas que necesita el adaptativo; la figura muestra ambos. El objetivo
del 1 % corresponde al indicador interno, no a una garantía de error puntual
ni a una nueva validación independiente de cada tabla.

## Archivos y reanudación

- `manifest.json`: 80 condiciones, opciones numéricas, hashes de código y datos,
  versiones, hilos y protocolo; `inputs/gri30.yaml` conserva el mecanismo.
- `cases/<condición>/attempt-NNN/profiles/`: perfiles completos en NPZ, incluidos
  los rechazados disponibles; nombres de especies y variables físicas originales.
- `trace.json`: predictor, referencias de continuación, recuperaciones y contadores
  disponibles, tiempos, malla final y diagnóstico de progreso por llama.
- `rounds.json`, `round_tables/`: inserciones, indicadores y tablas intermedias;
  `fgm_table.npz` solo se publica cuando converge el adaptativo.
- `warmup.json`, `progress.json`, `state.json`, `result.json`, `process.log`:
  progreso, estados, tiempos y causas de fallo. Guardado atómico con reintentos
  para bloqueos transitorios de Windows/OneDrive.
- `report/conditions.csv`, `conditions.json`, `attempts.json`: índice completo,
  tiempos desglosados y todos los intentos, además de PDF, PNG y fragmentos LaTeX.

`--resume` omite construcciones terminadas compatibles, incluidos los fallos.
Reinicia desde cero una construcción interrumpida, conservando el intento anterior.
Para volver a intentar un fallo revisado, añadir `--retry-failed` y, si procede,
`--case ID`. Cambiar código, entorno o configuración requiere otro directorio.
No eliminar `campaign.lock` mientras haya un proceso activo; una terminación
forzada puede dejar ese archivo como aviso de interrupción.

El modo `--smoke` usa un protocolo técnico diferente y exige un directorio
separado; sus resultados no se incorporan a la tesis.
