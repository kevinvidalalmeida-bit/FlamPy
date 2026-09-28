# Resultados L3 y diagnóstico del coste

La campaña `runs/thesis_flames_L3` contiene 560 resoluciones aceptadas: 56
condiciones, cinco pares KFLAME–Cantera por condición. Incluye promediado y
multicomponente, con y sin Soret, para CH4/GRI30 e H2/h2o2. No hace falta
otra campaña de 28+28 condiciones para incorporar Soret.

## Regenerar figuras y tablas, sin resolver llamas

Desde la raíz del proyecto, en PowerShell:

```powershell
python tools/postprocess_flame_sweeps.py --input runs/thesis_flames_L3 --output runs/thesis_flames_L3/report
python tools/postprocess_flame_diagnostics.py --input runs/thesis_flames_L3_diagnostics --campaign runs/thesis_flames_L3 --conditioning runs/thesis_flames_L3_conditioning_v2 --output runs/thesis_flames_L3/report
```

El informe contiene ocho figuras con resultados (02–07, 09 y 10), en PDF y PNG:
perfiles, composición, presión, temperatura, transporte/Soret y tiempos de
todas las opciones y mallas finales. La figura 10 usa exactamente la misma
selección de perfiles que la figura 02: caso 1 (CH4 promediado sin Soret) y
caso 9 (H2 multicomponente con Soret), primera repetición utilizable de cada
solver. Muestra el dominio completo, el frente ampliado y el espaciado local
sin interpolar ni reconstruir los nodos. `mallas_referencia.csv` conserva
identificador, repetición, ruta, hash del perfil y medidas de malla.
Los archivos 01 y 08 identifican verificaciones o ablaciones
pendientes y no se insertan como figuras de resultados en la tesis.

`summary.json` conserva intervalos bootstrap pareados y las cinco observaciones.
`diagnosticos.json`, `desglose_diagnostico.csv`, `etapas_diagnosticas.csv` y
`condicionamiento.csv` contienen las tablas complementarias. La tesis incluye
fragmentos `figure_*.tex`, `table_*.tex` y `table_diagnostic_*.tex`, organizados
con la interpretación en `.local/research/thesis/capitulo_resultados_L3.tex`.
Los informes completos `report.tex` y `diagnostics_tables.tex` se conservan
para consultas independientes; el capítulo usa sus fragmentos para evitar
duplicar tablas y separar las figuras de su explicación.

Los resultados previos y FGM se han retirado del documento compilado.
El capítulo principal sigue la campaña L3 de llamas individuales.

## Ejecuciones diagnósticas realizadas

- Diez estados, una repetición instrumentada por solver, con calentamiento
  excluido: ocho centrales (dos combustibles × cuatro transportes), H2 a 10 atm
  multicomponente/Soret y H2 a phi=0.9 promediado/Soret.
- Los veinte perfiles coinciden exactamente con los correspondientes perfiles
  originales en coordenada, temperatura, velocidad y fracciones másicas.
- Cuatro estados centrales adicionales para extraer ocho Jacobianos: ambos
  combustibles, promediado sin Soret y multicomponente con Soret. Son diagnósticos
  separados, sin incorporar sus tiempos a las cinco observaciones originales.
- Matrices dispersas, perfiles y escalas de equilibrado se guardan en
  `runs/thesis_flames_L3_conditioning_v2`. La primera prueba de extracción con
  la interfaz adjunta no era compatible con el backend disperso de Cantera;
  su log queda preservado en `runs/thesis_flames_L3_conditioning`.

Comandos reproducibles; omiten los diagnósticos ya completados:

```powershell
python benchmarks/diagnose_flame_campaign.py --resume
python benchmarks/diagnose_flame_conditioning.py --output runs/thesis_flames_L3_conditioning_v2
```

Los scripts comprueban la identidad del código y entorno de la campaña. Las
modificaciones de instrumentación actúan solo en sus procesos de diagnóstico.

## Lectura de los contadores

Los tiempos nativos de Jacobiano, residual, factorización y sustituciones son
categorías sin solapamiento, asignadas a la llamada exterior. Los subtimers
internos del solver están anidados: no deben sumarse entre sí. Las fases
preliminares multicomponente están incluidas en los contadores globales.

Cantera comunica tiempos y contadores retenidos al terminar. Algunas etapas
interrumpidas al ampliar el dominio no permanecen en esos contadores; los
callbacks sí registran los pasos transitorios aceptados a lo largo del solve.
Por ejemplo, CH4 central tiene 190 callbacks transitorios, frente a 110 pasos
en las estadísticas finales. Las llamadas internas usadas por diferencias
finitas tampoco tienen el mismo alcance que el contador de residual nativo.
El resto del tiempo conserva todas las tareas sin desglose disponible.

PTC aceptado es una corrección aceptada; Euler aceptado es un paso convergido.
Los fallos Euler de Cantera se extraen de los mensajes de fallo del paso
temporal, y los pasos Newton estacionarios de los resultados de dampStep
impresos en el log de nivel 2. La columna Iter enumera ensayos de
amortiguamiento dentro de un paso; sus filas no son iteraciones Newton.
Las iteraciones internas de Newton durante Euler no están disponibles.

`quimica_transporte_subtiempos.csv` y `table_diagnostic_subtimes.tex` desglosan
termoquímica del residual, transporte y flujos, preparación del Jacobiano,
derivadas termoquímicas y ensamblaje espacial de KFLAME, con segundos y
llamadas al bloque, sumando una vez la etapa preliminar. Son subtotales ya
incluidos en las categorías globales. Cantera expone residual y Jacobiano
agregados, sin separar aquí química y transporte.
Fuente del alcance del log:
[MultiNewton 3.2](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/MultiNewton.cpp).

## Sensibilidad espacial L3, para ejecutar por el usuario

```powershell
python benchmarks/benchmark_flame_mesh_sensitivity.py --input runs/thesis_flames_L3 --output runs/thesis_flames_L3_sensitivity --resume
```

Seis estados: central, 10 atm y 500 K de CH4/GRI30 promediado sin Soret y
H2/h2o2 multicomponente/Soret. Mantiene la estrategia no lineal por mecanismo.
Reutiliza el primer par aceptado L3 en cada estado. Resuelve L2, L4 y L5,
y después L3 con dominio inicial doble del mayor dominio final del par.
Son 48 nuevas resoluciones medidas y 48 calentamientos excluidos, sin las
cinco repeticiones temporales de la campaña principal. Comprueba perfiles,
configuración, código, entorno e inputs antes de comenzar. Guarda pares
con la escritura atómica y reanudación del ejecutor principal.

El proceso regenera automáticamente `11_sensibilidad_L3.pdf/.png`,
`sensibilidad_L3.csv/.json` y `sensitivity_report.tex` en el informe principal.
El capítulo ya tiene el punto de inserción; aparece pendiente hasta recibir
datos. Las variaciones se muestran con signo y escala lineal, relativas a L3;
se informa además el cambio L4–L5 y el dominio. El umbral del 0,5% describe
sensibilidad, sin cambiar la campaña L3 ni bloquearla.

Solo para regenerar sin resolver:

```powershell
python tools/postprocess_flame_mesh_sensitivity.py --input runs/thesis_flames_L3_sensitivity --output runs/thesis_flames_L3/report
```

Los tiempos por transporte y el espaciado de malla usan eje logarítmico;
velocidades, efectos de Soret y cambios espaciales con signo mantienen
escala lineal. Las discrepancias absolutas parten de cero.

## Condicionamiento

Se estima `||J||_1 * onenormest(J^-1)` mediante LU dispersa, con semilla fija.
También se estima sobre `Dr J Dc`, normalizando máximos absolutos primero por
fila y luego por columna; las escalas se guardan. Es una estimación inferior,
sujeta a errores de aritmética, no un cálculo exacto por SVD.

KFLAME reconstruye su linealización estacionaria en el estado final. Cantera
termina su resolución original con el solver de banda y después realiza una
corrección Newton dispersa en la misma malla para exportar su última matriz.
Esta corrección cambia Su menos del 0.00075% y T menos de 0.0043 K. Se registran
los cambios, el tamaño y el hash de cada matriz. Las formulaciones y mallas
difieren: los números de condición no clasifican por sí solos a los solvers
ni prueban la causa de una diferencia temporal.

Referencias de las interfaces y del estimador:

- [Cantera: Jacobianos](https://cantera.org/3.2/python/utilities.html#jacobians)
- [SciPy: onenormest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.onenormest.html)
- [Cantera: solución no lineal 1D](https://cantera.org/stable/reference/onedim/nonlinear-solver.html)

## Pendiente: ablaciones

Son veinte resoluciones adicionales, diez por mecanismo (cinco pares de
configuración normal/modificada), independientes de la campaña Cantera:

```powershell
python benchmarks/benchmark_flame_sweeps.py --phase ablation --spatial-policy fixed-L3 --pairs 5 --output runs/thesis_flames_L3 --resume
```

Después se regenera el informe offline. La campaña L3 no certifica por sí misma
independencia espacial; los cinco tiempos repetidos describen dispersión temporal.
Las tolerancias transitorias configuradas y las etapas automáticas de cada
solver se explicitan en el texto de resultados.
