# HPC en mallas pequeñas de mecanismos grandes

**Última mejora del 9 de octubre:** [banco de transporte compartido entre propiedades y Jacobiano](hpc-transport-bank.md), con tiempos de la fuente final y comprobaciones dentro de Newton.

**Nueva continuación del 9 de octubre:** [reutilización exacta de coeficientes entre columnas de temperatura](hpc-temperature-reuse.md), con tiempos de la fuente integrada y controles sin reutilización.

[HPC por mecanismo](hpc-mechanisms.md) · [Estado de las estrategias](hpc-strategy-status.md) · [Datos y pruebas](assets/chemistry/hpc-small-grids-2026-10-09.json)

Se integra un reparto con más rangos independientes para el Jacobiano multicomponente congelado de JetSurF y Aramco en mallas pequeñas. En nueve nodos y cuatro hilos disponibles, pasa de dos rangos a cuatro. La referencia es una nueva copia de los 70 archivos **posterior** a las tres mejoras de [la revisión anterior](hpc-reprofile.md). Este cambio adicional modifica únicamente el despacho Python de `diffusion.py`; todos los kernels científicos conservan sus bytes.

## Antes y después de la fuente integrada

Una pasada de propiedades, residual con esas propiedades y Jacobiano con esas propiedades: el ciclo utilizado por Newton. Se incluyen asignaciones, los sistemas de las caras y las copias necesarias. Es un estado fijo con transporte multicomponente y Soret, sin resolver una llama completa. Son siete pares alternados después de dos calentamientos, Numba=4 y BLAS=1. El snapshot y la fuente integrada tienen globals de sus drivers separados y llaman a los mismos kernels, cuya fuente no cambió.

| Mecanismo | Antes (ms) | Ahora (ms) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|
| H2-O2 | 0.858 | 0.820 | 0.997× | 3/7 |
| DRM19 | 1.829 | 1.633 | 1.103× | 6/7 |
| DRM22 | 1.635 | 1.690 | 1.015× | 4/7 |
| FFCM-1 | 2.916 | 2.980 | 1.025× | 5/7 |
| GRI-Mech 3.0 | 5.728 | 5.602 | 0.981× | 3/7 |
| USC Mech II | 25.225 | 24.987 | 1.008× | 5/7 |
| JetSurF 2.0 | 437.054 | 297.229 | 1.485× | 7/7 |
| AramcoMech 2.0 | 1119.590 | 769.041 | 1.473× | 7/7 |
| AramcoMech 3.0 | 1889.324 | 1354.943 | 1.367× | 7/7 |

La selección se considera consistente con una mediana emparejada ≥1,03× y al menos cinco victorias de siete. **El cambio se activa en los tres mecanismos de ≥256 especies de esta tabla.** Los otros seis son controles con el mismo reparto; sus variaciones no se atribuyen a esta optimización, incluso si una fila supera ese criterio. Las medianas separadas y la mediana de cocientes emparejados pueden diferir. No se multiplican estas ganancias por las de campañas anteriores. La tabla anterior del residual y Jacobiano públicos incluía dos pasadas de propiedades: sus tiempos no se comparan directamente con este ciclo de una pasada.

Todos los nueve mecanismos conservan propiedades, residual, cinco arrays de bloques y columna Lambda byte a byte, tanto en A-B-A como en cada par cronometrado. Las propiedades de transporte se contrastan además con Cantera 3.2.0 en tres estados por mecanismo, con los límites existentes. Esta comprobación es de propiedades; no es una nueva comparación de nueve llamas convergidas. H2 se fuerza a frozen como control del perfil; su modo automático de producción sigue exacto.

## Selección y mallas mayores

Se aplica únicamente al ensamblador por bloques químicos con química analítica, ≥256 especies, Jacobiano frozen y **8–15 nodos**. El mínimo baja de cuatro a dos nodos por rango. En el resto de mallas se conserva el mínimo anterior; cada rango sigue poseyendo sus buffers y rotando sus caras privadas. Se mantiene el presupuesto y tamaño de los bloques químicos. No cambia el residual, las tolerancias, la precisión ni el orden de las sumas. PTC/SER conserva el estado de la revisión anterior.

Con cuatro hilos disponibles:

| Mecanismo | Nodos | Antes (ms) | Candidata (ms) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|---:|
| JetSurF 2.0 | 13 | 472.285 | 403.645 | 1.186× | 7/7 |
| JetSurF 2.0 | 33 | 905.295 | 909.549 | 0.989× | 2/7 |
| AramcoMech 2.0 | 13 | 1198.210 | 1029.449 | 1.199× | 7/7 |
| AramcoMech 2.0 | 33 | 2389.711 | 2429.152 | 0.988× | 2/7 |
| AramcoMech 3.0 | 13 | 1868.280 | 1687.523 | 1.103× | 7/7 |
| AramcoMech 3.0 | 33 | 3988.420 | 4055.391 | 0.988× | 2/7 |

En 13 nodos el reparto pasa de tres a cuatro rangos. En 33 nodos ambos usan cuatro rangos y ejecutan el mismo trabajo; son controles y no ganancias atribuidas. Estos pilotos se ejecutaron antes de integrar; el AST de la fuente integrada coincide con la candidata seleccionada.

También se comprueba la configuración de **24 hilos disponibles**, correspondiente a los 24 procesadores lógicos del Ryzen 9 7845HX. En 13 nodos los rangos pasan de tres a seis, con BLAS=1:

| Mecanismo | Antes (ms) | Candidata (ms) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|
| JetSurF 2.0 | 473.544 | 355.837 | 1.325× | 7/7 |
| AramcoMech 2.0 | 1200.378 | 902.264 | 1.316× | 7/7 |
| AramcoMech 3.0 | 1931.142 | 1594.904 | 1.201× | 7/7 |

Los tres casos conservan todos los bytes y ganan los siete pares. Esto compara los rangos que realmente se ejecutan, no 24 workers simultáneos. El selector de mallas pequeñas evita extender este cambio a un reparto de 16 rangos en 33 nodos con la configuración de 24 hilos.

## Dónde se gasta el tiempo

Medianas diagnósticas de tres llamadas por fase, fuera de la selección del ciclo completo. Las fases se cronometraron por separado y en bloques sucesivos; no son porcentajes aditivos de una única llamada.

| Mecanismo, nueve nodos | Propiedades antes (ms) | Residual compartido antes (ms) | Jacobiano antes (ms) | Jacobiano ahora (ms) |
|---|---:|---:|---:|---:|
| JetSurF 2.0 | 58.836 | 0.107 | 369.409 | 253.101 |
| AramcoMech 2.0 | 159.742 | 0.122 | 968.422 | 594.102 |
| AramcoMech 3.0 | 247.176 | 0.090 | 1502.541 | 991.690 |

La mejora aprovecha más paralelismo del ensamblaje por nodos. No se midieron fallos L1/L2/L3 ni se atribuye un porcentaje a una caché particular. Tener más rangos requiere **más scratch simultáneo**:

| Especies | Un workspace (MiB) | Dos rangos (MiB) | Cuatro rangos (MiB) |
|---|---:|---:|---:|
| 348 | 27.90 | 55.80 | 111.61 |
| 493 | 55.89 | 111.78 | 223.56 |
| 581 | 77.57 | 155.14 | 310.27 |

Estos tamaños se obtienen de los arrays reales de `_node_workspace`, uniendo sus intervalos para evitar contar aliases dos veces. Son payload de scratch, excluyen el Jacobiano, parciales químicos, backend, caché JIT y overhead del allocator; no son mediciones de RSS máximo. La ganancia de tiempo de esta selección no implica una reducción de memoria. Los datos contienen también el payload de tres y seis rangos para 13 nodos.

## Qué ocurrió con las otras propuestas de los tres agentes

La disposición transpuesta privada del flujo conserva los resultados. Su primer piloto, sobre la fuente anterior a este nuevo despacho, dio:

| Mecanismo | Antes (ms) | Candidata (ms) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|
| GRI-Mech 3.0 | 4.719 | 4.868 | 0.977× | 3/7 |
| JetSurF 2.0 | 423.173 | 392.655 | 1.076× | 7/7 |
| AramcoMech 3.0 | 2001.823 | 1937.662 | 1.020× | 5/7 |

JetSurF supera el criterio local; GRI y Aramco 3 no. Esta variante **no se integra**: falta seleccionarla con mallas mayores y sobre la fuente final. Su oráculo independiente aprobó los **540 casos seleccionados y 2160 invocaciones**, con matrices C/F/strided, entradas readonly y modificables, composiciones densas/dispersas/trazas, cancelación, ceros con signo, NaN e infinitos. Son 540 de los 8064 casos posibles del driver, no una ejecución de los 8064. Todos los bytes finitos, clases no finitas, inputs y guardas pasan; no hubo diferencias de codificación NaN.

El agente del banco residual→Jacobiano dejó un diseño compacto con límite de 32 MiB y un prototipo de ownership/presupuesto. No calcula transporte y no se ejecutó como mejora numérica. Falta establecer propiedad inmutable de toda la configuración efectiva y medir construcción, escrituras, copias e invalidación dentro del ciclo. Su variante conservadora que copia el witness completo consume 32,42/65,02/90,28 MiB sólo en coeficientes para 348/493/581 especies: no entra en ese presupuesto. Esto no demuestra que un banco con configuración inmutable sea imposible; identifica el coste de ese prototipo.

El agente de arranque dejó un perfilador de importación, constructor, `compile_kernels` y primeras/calientes llamadas de H2/GRI con cachés cold/warm separadas. Sólo se verificó su preparación y sintaxis. La campaña numérica de arranque sigue pendiente y no aporta tiempos a estas tablas.

Las 15 estrategias originales conservan sus pendientes. Este cambio concreta un caso adicional del reparto por tamaño; no cierra las comparaciones de banco completo, mallas mayores ni políticas no lineales. La decisión de GPU mantiene su evidencia anterior: no hay nuevas mediciones GPU en esta campaña ni un cambio de backend.

## Validación y reproducción

Se aprobaron **111 pruebas seleccionadas**, incluidas seis nuevas comparaciones de propiedad por rangos en JetSurF/Aramco: 9 y 13 nodos, A-B-A, Soret on/off, temperatura libre/fija y radiación. Se incluyen también las regresiones de rotación de caras, buffers entre bloques, memoria y llamas de difusión. Este número es de la ejecución actual; las 853 pruebas completas y las 168 de la revisión anterior mantienen su alcance histórico.

Los raw están bajo `.local/hpc-layout-2026-10-09/`. `integrated-n9.json` contiene la fuente integrada; `thread-n13-n33.json` y `thread-default-n13.json` contienen los pilotos acotados; `selected-tests.log` contiene la salida de pytest. Los datos públicos incluyen hashes de fuentes y archivos brutos, estados/mecanismos, muestras, fingerprints y la selección del oráculo. Los primeros borradores fallidos del comparador y del guard de hilos se conservan y se excluyen de las tablas.

El baseline tiene manifest `54ada3ecb0ea42f7bc1fd5136a691e8ed975d3cb668136cfca9a7e0a967270d9`. `prepare_integrated.py` comprueba que el único cambio científico de fuente es el despacho de `diffusion.py` y que el AST coincide con la candidata seleccionada. El kernel original y el de workspace conservan fuente y firma. No se presenta una ganancia nueva de llama completa para JetSurF ni Aramco.
