# Banco de transporte compartido entre propiedades y Jacobiano

La fuente integrada evita recalcular los coeficientes base que la evaluación de propiedades ya obtuvo. Conserva difusión multicomponente, Soret y conductividad en un banco propio de la invocación y los copia por filas al scratch existente del Jacobiano. Las perturbaciones de temperatura siguen usando el cierre completo y el mismo orden de operaciones.

La referencia son los 70 archivos aceptados después de [la reutilización entre columnas de temperatura](hpc-temperature-reuse.md). Se conserva congelada para comparar la fuente real final. Los cocientes siguientes son mejoras adicionales respecto a esa referencia; no se multiplican por ganancias de campañas anteriores. [Datos completos y procedencia](assets/chemistry/hpc-transport-bank-2026-10-09.json).

## Antes y después

Numba=4, BLAS=1, siete pares alternados después de dos calentamientos. Cada muestra incluye propiedades, residual y Jacobiano, además de asignación, exportación, snapshots, validación y copias del banco. Compilación, comprobación de bytes e I/O quedan fuera del timer.

| Mecanismo | Nodos | Antes (ms) | Ahora (ms) | Cociente emparejado | Victorias | Banco (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| JetSurF 2.0 | 9 | 274.601 | 232.112 | 1.183× | 7/7 | 7.413 |
| JetSurF 2.0 | 33 | 848.408 | 697.028 | 1.220× | 7/7 | 29.652 |
| AramcoMech 2.0 | 9 | 718.845 | 597.232 | 1.204× | 7/7 | 14.865 |
| AramcoMech 2.0 | 33 | 2292.833 | 1915.786 | 1.198× | 7/7 | 59.459 |
| AramcoMech 3.0 | 9 | 1278.810 | 1063.815 | 1.190× | 7/7 | 20.639 |
| AramcoMech 3.0 | 33 | 3885.200 | 3320.877 | 1.170× | 7/7 | 82.554 |

El cociente es la mediana de los siete cocientes emparejados; puede diferir del cociente entre las medianas mostradas. Se exige al menos 1,03× y cinco victorias para atribuir una ganancia consistente.

## Dentro de Newton

Se midieron llamadas reales de tres iteraciones, con edad máxima cuatro, incluyendo factorización, solve y amortiguación. En ambos lados se obtiene el mismo estado byte a byte, un Jacobiano y el resultado `iteration_limit`. Estos ensayos no certifican una llama convergida.

| Mecanismo, 9 nodos | Antes (s) | Ahora (s) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|
| JetSurF 2.0 | 0.553252 | 0.503711 | 1.103× | 7/7 |
| AramcoMech 2.0 | 1.369692 | 1.212367 | 1.132× | 7/7 |
| AramcoMech 3.0 | 2.186625 | 2.020669 | 1.088× | 7/7 |

Newton construye el banco al evaluar una base que requiere Jacobiano nuevo. Durante las pruebas de amortiguación lo exporta sólo cuando la edad del modelo anticipa una renovación. Una renovación inesperada utiliza el camino anterior con las propiedades correctas. Los ensayos rechazados nunca aportan el banco de la base aceptada. El banco se libera tras formar los bloques y no queda almacenado en el problema ni en el modelo; se comprueba mediante referencias débiles.

## Dónde se ahorra el tiempo

| Mecanismo, 33 nodos | Propiedades antes (ms) | Propiedades + banco (ms) | Jacobiano antes (ms) | Jacobiano ahora (ms) |
|---|---:|---:|---:|---:|
| JetSurF 2.0 | 141.342 | 142.996 | 700.066 | 562.540 |
| AramcoMech 2.0 | 368.172 | 356.625 | 1895.271 | 1530.454 |
| AramcoMech 3.0 | 568.134 | 559.908 | 3165.826 | 2669.542 |

Las fases son diagnósticos separados y no se suman para reconstruir otro benchmark. Exportar el banco añade tráfico de memoria a las propiedades; el ahorro procede de evitar repetir el cierre base del Jacobiano. Las factorizaciones de caras perturbadas y del sistema lineal siguen siendo costes pendientes.

## Memoria y selección

El banco C-contiguo contiene `D[caras,Ns,Ns]`, `DT[caras,Ns]` y `lambda[caras]`: `8*(n-1)*(Ns*Ns+Ns+1)` bytes. El límite efectivo es **96 MiB**. Sólo se usa en mecanismos de al menos 256 especies, caras compiladas, composición congelada, Schur y química analítica por bloques. Si no cabe el banco completo se usa el camino anterior, sin asignar una parte gigantesca.

En 65 nodos, JetSurF necesita 59,303 MiB; Aramco 2 y 3 necesitarían 118,917 y 165,109 MiB y quedan fuera del límite. Son cálculos de payload, no nuevas mediciones de rendimiento de 65 nodos. Se conservan los workspaces privados anteriores: el banco no sustituye su memoria. El plan de arrays incluye snapshots y una referencia diagnóstica de 4 GiB; esa referencia no es una política de admisión de producción ni una cota del RSS/JIT, factores o memoria temporal del proceso.

La validación exige identidad del problema propietario, backend y datos compilados, configuración, presión y los bits exactos de T/Y/grid/conductividad. Las vistas `uint64` distinguen +0 y −0. Un banco de otro estado, configuración o dueño se rechaza. Los coeficientes son de sólo lectura y las invocaciones independientes no comparten memoria mutable.

El código oficial de Cantera 3.2 invalida los datos dependientes de temperatura y composición por separado cuando cambia su estado. La oportunidad aquí consiste en conservar explícitamente el resultado de la evaluación de la misma base durante el ensamblado del Jacobiano. [MultiTransport.cpp, Cantera 3.2](https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/MultiTransport.cpp). La amortiguación y la renovación de Jacobianos siguen la política anterior del proyecto; la documentación de Cantera describe por qué un Jacobiano envejecido requiere renovación. [Solver no lineal de Cantera](https://www.cantera.org/3.2/reference/onedim/nonlinear-solver.html).

## Controles y comprobaciones

| Temperatura por encima de Tref, 9 nodos | Antes (ms) | Ahora (ms) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|
| JetSurF 2.0 | 300.033 | 254.410 | 1.173× | 7/7 |
| AramcoMech 2.0 | 850.873 | 684.899 | 1.265× | 7/7 |
| AramcoMech 3.0 | 1266.326 | 1056.580 | 1.196× | 7/7 |

Estos estados no tienen coincidencias de temperatura perturbada entre nodos. La reutilización del nuevo banco base puede actuar igualmente; es una oportunidad distinta de la mejora térmica anterior. H₂, DRM19, DRM22, FFCM-1, GRI y USC mantienen el camino anterior y sus seis controles se conservan aparte, sin atribuirles ganancia del banco.

| Control inactivo, 9 nodos; lote de 20 | Antes (ms/ciclo) | Ahora (ms/ciclo) | Cociente emparejado |
|---|---:|---:|---:|
| h2o2.yaml | 0.756 | 0.709 | 1.000× |
| drm19.yaml | 1.568 | 1.578 | 0.993× |
| drm22.yaml | 1.487 | 1.435 | 0.997× |
| ffcm1.yaml | 2.939 | 2.871 | 1.001× |
| gri30.yaml | 5.305 | 5.471 | 0.970× |
| uscmech2.yaml | 25.174 | 25.085 | 1.004× |

La medida de una sola llamada dio 0,817× en DRM19; su lote de 20 dio 0,993×. GRI dio 0,970× con 20 ciclos y 1.012× al repetir con 100. Se conservan todos los resultados: no se atribuye aceleración ni una regresión consistente del banco a estos caminos inactivos. En los lotes se temporizan todos los ciclos y se informa la media por ciclo; sólo se comparan los 14 resultados del último ciclo de cada lote, además de los gates de calentamiento y A–B–A. No equivale a comprobar individualmente todas las salidas internas.

La fuente final pasa **140 pruebas seleccionadas**, sin omisiones; incluyen siete casos nuevos del banco y regresiones de propiedades reutilizadas, rangos, scratch, caras, temperaturas y llamas de difusión. Los quince escenarios principales de una llamada comparan 14 arrays: siete propiedades, residual, cinco bloques y la columna del borde. Se exige igualdad de bytes, shapes, tipos, strides y finitud durante cada par, además de A–B–A. Los seis lotes de control y la repetición GRI se conservan aparte con el alcance indicado arriba.

Seis escenarios finales independientes comprueban malla no uniforme, radiación, composición dispersa con trazas OH, Soret activado/desactivado, temperatura fijada, cruce NASA a 1000 K y propiedad de la memoria. En 36 evaluaciones de cara se compara el propio banco con un gas nuevo de Cantera 3.2: D, conductividad y Soret con rtol 3e-7 y atol 1e-12, leyendo primero `multi_diff_coeffs`. También se verifican las propiedades de los nueve mecanismos en tres estados por escenario. El calentamiento de JetSurF precarga el productor de un solve y los consumidores C/A: uniformidad, perfil lineal y temperatura >Tref no añaden especializaciones. No se presenta como una medición de arranque sin caché.

## Alcance y trabajo pendiente

Es una nueva variante limitada del banco C, validada sobre la fuente actual; las pruebas históricas C/full y C-seeds conservan su estado y sus errores previos en [el inventario](hpc-strategy-status.md). No da por cerradas las quince estrategias originales.

Los tres agentes alcanzaron su límite de uso. Sus helpers de memoria, validación y análisis fueron revisados y ejecutados por el agente principal; no se atribuyen pruebas ejecutadas a los agentes interrumpidos. Las campañas privadas, las finales y un error inicial de nombres de mecanismos permanecen separados en los datos.

Quedan pendientes nuevas llamas completas convergidas de JetSurF/Aramco y su tiempo total frente a Cantera. No hay una nueva campaña GPU ni contadores L1/L2. La decisión GPU anterior está en [HPC por mecanismo](hpc-mechanisms.md). La siguiente oportunidad es un banco parcial o por bloques que conserve los bits cuando el banco completo excede el límite, y perfilar los cierres de temperatura que siguen requiriendo nueva factorización.
