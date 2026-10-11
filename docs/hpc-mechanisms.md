# HPC por mecanismo: transporte multicomponente y Soret

**Mecanismos grandes, 10 de octubre:** [Jacobiano estructurado, Newton–Krylov y transporte vectorizado](hpc-large-mechanisms.md): AramcoMech 3.0 completa pasa de no terminar a 115 s; JetSurF 3,6×; GRI 1,5×.

**Jacobiano térmico, 10 de octubre:** [derivadas implícitas y tiempos inclusivos](hpc-thermal-jacobian.md), con validación contra Cantera y opción experimental desactivada.

**Continuación del 10 de octubre:** [derivadas y actualizaciones algebraicas](hpc-analytic-transport.md), con tiempos de la fuente integrada, controles contra Cantera y opciones experimentales desactivadas por defecto.

**Última continuación del 9 de octubre:** [perfil de Aramco3 y columnas de transporte por bloques](hpc-jacobian-breakdown.md), con fuente final, Cantera y reparto interno del Jacobiano.

**Mejora anterior del 9 de octubre:** [banco de transporte compartido entre propiedades y Jacobiano](hpc-transport-bank.md), con tiempos de la fuente final y comprobaciones dentro de Newton.

**Nueva continuación del 9 de octubre:** [reutilización exacta de coeficientes entre columnas de temperatura](hpc-temperature-reuse.md), con tiempos de la fuente integrada y controles sin reutilización.

**Continuación del 9 de octubre:** [reparto por rangos en mecanismos grandes y mallas pequeñas](hpc-small-grids.md), con la fuente final validada y sus límites de medición.

**Revisión posterior del 9 de octubre:** [nuevo perfil, optimizaciones integradas y tiempos reales contra Cantera](hpc-reprofile.md). Esta revisión usa una nueva referencia de 70 archivos; las mediciones históricas que siguen conservan su alcance original.

[Guía de lectura](README.md) · [Mecanismos](mechanisms.md) · [Estado de las estrategias](hpc-strategy-status.md)

La fuente integrada conserva los resultados anteriores y supera la comparación
contra Cantera en los nueve mecanismos instalados. Se seleccionaron dos cambios
de memoria: scratch privado reutilizado por bloques de cuatro caras y acceso
por filas en el ensamblaje térmico de mecanismos de al menos 400 especies.
Los tiempos corresponden a evaluación de propiedades; no son tiempos de nueve
llamas convergidas.

## Antes y después, con los mismos cuatro hilos

32 caras, presión de 101325 Pa, Numba=4 y BLAS=1; siete pares con orden
alternado después de dos calentamientos completos por versión. Se incluye
preparación termoquímica, asignaciones y los cinco resultados públicos:
densidad, conductividad, peso molecular, difusión multicomponente y Soret.
El cociente es la mediana de los siete cocientes emparejados; puede diferir
del cociente entre las dos medianas mostradas.

| Mecanismo | Especies | Antes (ms) | Ahora (ms) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|---:|
| H2-O2 | 10 | 0.062 | 0.062 | 0.980× | 3/7 |
| DRM19 | 21 | 0.234 | 0.243 | 0.956× | 0/7 |
| DRM22 | 24 | 0.201 | 0.212 | 1.001× | 4/7 |
| FFCM-1 | 38 | 0.464 | 0.457 | 1.001× | 4/7 |
| GRI-Mech 3.0 | 53 | 1.079 | 0.995 | 1.105× | 7/7 |
| USC Mech II | 111 | 5.928 | 6.072 | 0.993× | 3/7 |
| JetSurF 2.0 | 348 | 129.698 | 121.873 | 1.064× | 7/7 |
| AramcoMech 2.0 | 493 | 358.524 | 320.065 | 1.118× | 7/7 |
| AramcoMech 3.0 | 581 | 591.495 | 510.481 | 1.159× | 7/7 |

Sólo se interpreta como ganancia consistente una mediana de al menos 1,03×
y cinco victorias de siete. Los controles sin cambios de algoritmo pueden
variar por ruido. Los intervalos finales intersectan 600–2300 K con el
intervalo común NASA de cada archivo; los intervalos exactos y las muestras
están en [los datos completos](assets/chemistry/hpc-mechanisms-2026-10-09.json).
La máquina es un Ryzen 9 7845HX con 32 GiB de RAM.

## Camino residual y Jacobiano

La evaluación de propiedades no equivale al paso completo. En el estado fijo
de nueve nodos, con Jacobiano multicomponente congelado en ambas versiones,
el resultado es:

| Mecanismo | Antes (ms) | Ahora (ms) | Cociente emparejado |
|---|---:|---:|---:|
| H2-O2 | 0.662 | 0.877 | 0.855× |
| DRM19 | 1.559 | 1.730 | 0.939× |
| DRM22 | 1.551 | 1.691 | 0.951× |
| FFCM-1 | 3.067 | 3.067 | 1.000× |
| GRI-Mech 3.0 | 6.453 | 6.457 | 0.959× |
| USC Mech II | 31.936 | 32.037 | 0.987× |
| JetSurF 2.0 | 842.943 | 816.860 | 1.039× |
| AramcoMech 2.0 | 1922.684 | 1894.358 | 1.014× |
| AramcoMech 3.0 | 3055.030 | 2993.201 | 1.023× |

El camino inclusivo no mejora uniformemente: los controles pequeños tienen
pérdidas locales y las ganancias de los mecanismos grandes son menores que
en la tabla de propiedades. No se atribuye 1,159× a una llama completa de
Aramco. La regresión de llamas existentes comprueba exactitud; su orden
baseline→candidata y dos muestras no bastan para atribuir cambios pequeños
de tiempo a una optimización nueva.

## Dónde se va el tiempo

El perfil inicial de AramcoMech 3.0, ocho caras y un hilo, midió 472,47 ms
en transporte multicomponente frente a 1,27 ms en cinética fusionada.
Para una cara representativa: 2,92 ms en términos de temperatura,
11,53 ms en la inversa ordinaria y 37,75 ms en ensamblaje/Schur/LU térmico.
Estas categorías tienen distinto alcance y no se suman como un perfil cerrado.
La sonda LU sobre una matriz ya factorizada se excluye de esta atribución.

## Qué cambia en memoria

El kernel de bloques reserva scratch una vez por bloque y procesa cuatro
caras consecutivas. Cada bloque conserva sus buffers privados; no comparte
scratch entre trabajadores. La ruta se usa a partir de 24 especies y 32
caras, cuando no se solicita el sistema denso. Conserva los strides de las
salidas, el fallback y las ramas de simetría certificada.

En `L11`, el recorrido interior por columnas de una fila hace contiguas las
lecturas de coeficientes. Las sumas de cada columna mantienen el orden
original de especies mediante un vector de acumulación. La selección sólo
se activa con al menos 400 especies y una matriz C contigua; los demás
layouts mantienen el recorrido anterior. Esto mejora localidad y evita
trabajo repetido; no implica que todas las matrices quepan en L1 o L2.
No se habilita `fastmath`, FMA ni precisión reducida.

## Comprobación de resultados

- **853 pruebas aprobadas**, incluidas 65 nuevas pruebas de transporte.

- Los cinco arrays públicos de transporte y los seis arrays de residual/Jacobiano
  coinciden byte a byte con la referencia independiente en los nueve mecanismos.
- Cantera verifica densidad, conductividad, difusión multicomponente y Soret;
  el benchmark independiente verifica también las tasas químicas. Tolerancias:
  transporte 3×10⁻⁷ relativo/10⁻¹² absoluto y tasas 3×10⁻⁹/10⁻⁸.
- Seis casos existentes H₂/GRI, incluyendo Soret y radiación: doce pares
  cronometrados de llamas completas y seis pares de calentamiento conservan
  los 21 campos exportados, malla y aceptación originales.

El Jacobiano se comprueba con estados fijos de nueve nodos. Para los cuatro
mecanismos grandes, la inicialización de esa prueba usa `CH4:1,N2:4`;
no es una validación de nueve llamas adaptativas de metano puro. USC tiene
intervalo común 300–2000 K: Cantera obtiene 2230,33 K para la entrada pura
y 1848,93 K para la diluida; la inicialización nativa diluida coincide en
menos de 2×10⁻⁸ K. El fallo original del bracket se conserva como evidencia.

La suite y las mediciones registran los mismos 69 archivos de fuente/datos;
sólo cambian dos módulos de transporte y la documentación del consumidor
compilado que invalida su caché dependiente. Los YAML y sus hashes no cambian.

## GPU: evidencia y decisión

El prototipo CUDA FP64 con LU ordenado y sin FMA conserva LU, pivotes y
solución térmica byte a byte en 27 pruebas matriciales. Los lotes 8/32 repiten
una matriz capturada por mecanismo; no representan distintas caras de una
llama. El total incluye copias y retorno de todas las salidas verificadas.
La comparación inicial usa CPU serial y no autoriza una migración a GPU.

La comparación adicional con CPU paralela de cuatro/ocho hilos contiene 12 casos; gate completo: aprobado.
Los cocientes CPU/GPU totales van de 0.165× a 0.577×; valores inferiores a uno favorecen a CPU.
CPU fue entre 1.73× y 6.07× más rápida en esta etapa. La migración de este kernel a la RTX 4060 no compensa en los casos medidos.

GPU permanece como experimento local: falta una integración de transporte
completo con comprobación contra Cantera y comparación de llamas. El backend
de producción sigue en CPU. Este informe no atribuye tiempos de kernel a la
resolución completa.

## Reproducir en otro equipo

```sh
python examples/benchmark_mechanisms.py --faces 32 --threads 4 --repeats 7 --validate-cantera --output runs/mechanism_timings.json
python -m pytest -q tests/test_large_transport_memory.py
```

La salida debe ser nueva. Aramco utiliza los archivos locales instalados;
si faltan, el benchmark informa su ausencia y las pruebas correspondientes
se omiten. La primera llamada incluye JIT y se registra aparte. Los datos
publicados guardan manifiestos, muestras y hashes de evidencia; las pruebas
fallidas y los experimentos descartados se conservan en la campaña local.
