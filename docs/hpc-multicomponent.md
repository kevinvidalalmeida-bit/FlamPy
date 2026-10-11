# HPC del transporte multicomponente: medición y mejoras

[Guía de lectura](README.md) · [Campaña anterior](hpc-next-strategies.md) · [Muestras y auditoría](assets/diffusion/hpc-multicomponent-2026-10-08.json)

[Estado de las propuestas de los agentes: aplicadas, pendientes y retiradas](hpc-strategy-status.md)

La revisión del 8 de octubre de 2026 se concentra en el transporte Dixon–Lewis
y Soret de la versión local ya optimizada. Se conserva el modelo y el orden
de las acumulaciones; se eliminan cálculos sin consumidores y se reutilizan
buffers. **554 pruebas aprobadas y 18 pares de llamas completos con
los 21 campos guardados exactamente iguales.** Hay además seis pares exactos
en el control final con el código anterior.

## Antes y después en esta campaña

| Llama completa | Antes (s) | Ahora (s) | Antes / ahora |
|---|---:|---:|---:|
| CH₄, multicomponente + Soret | 11,820 | 10,554 | 1,12× |
| H₂, multicomponente + Soret | 0,887 | 0,905 | 0,98× |
| CH₄, mezcla (control) | 2,912 | 3,033 | 0,96× |
| H₂, mezcla (control) | 0,503 | 0,551 | 0,91× |
| CH₄, mezcla + radiación (control) | 3,412 | 2,900 | 1,18× |
| H₂, mezcla + radiación (control) | 0,573 | 0,554 | 1,04× |

Son medianas de tres ejecuciones completas por caso, cuatro hilos Numba y
un hilo BLAS/OMP/MKL. Cada configuración se calienta antes de medir en su
proceso. El tiempo nativo incluye resolución, adaptación y diagnóstico;
la preparación, compilación y exportación quedan fuera. El JSON conserva
también el tiempo de API y pared. El punto de partida es la versión posterior
a caras compartidas y derivadas por bloques; no se comparan campañas distintas.

El código anterior se repitió al final para los dos casos multicomponente. CH₄, multicomponente + Soret: mediana 11,426 s, intervalo 11,317–11,543 s; H₂, multicomponente + Soret: mediana 0,930 s, intervalo 0,911–0,980 s.

Estas muestras describen esta máquina y entorno. La variación observada limita
la atribución de diferencias pequeñas; no constituyen una garantía de tiempo
ni un intervalo de confianza. Las mediciones completas y las de kernels
responden a preguntas diferentes.

La mejora se concentra en CH₄ multicomponente. En H₂ multicomponente la mediana
inicial pasó de 0,887 a 0,905 s y no demuestra aceleración; el control anterior
final fue 0,930 s. Los controles de mezcla también presentan resultados mixtos,
con una mediana más lenta en H₂ sin radiación y otra más rápida en CH₄ radiativo.
No se atribuyen esas variaciones a una optimización de su transporte.

En estados finales idénticos, siete muestras alternadas de los kernels
anteriores y nuevos, con compilación y comparación de bytes fuera del tiempo:

| Caso y operación | Antes (ms) | Ahora (ms) | Cociente |
|---|---:|---:|---:|
| H₂, multicomponente + Soret, caras | 3,242 | 3,052 | 1,06× |
| H₂, multicomponente + Soret, Jacobiano | 70,202 | 68,726 | 1,02× |
| CH₄, multicomponente + Soret, caras | 61,958 | 52,560 | 1,18× |
| CH₄, multicomponente + Soret, Jacobiano | 472,942 | 408,816 | 1,16× |

El tiempo del Jacobiano incluye un paso completo de caras base. No se suman
ambas columnas de tiempo. Los módulos anteriores de esta comparación se
compilan desde el árbol preservado y llaman a sus propios helpers: la referencia
no utiliza la implementación nueva de transporte.

## Dónde estaba el tiempo

Una resolución diagnóstica de CH₄ multicomponente y Soret de la versión
anterior dio 12,248 s y conservó exactamente las salidas de la campaña previa:

| Categoría | Tiempo acumulado (s) | Porcentaje del tiempo nativo |
|---|---:|---:|
| Transporte compilado de caras | 5,790 | 47,3 % |
| Jacobiano completo | 4,904 | 40,0 % |
| Factorización del sistema de la llama | 0,825 | 6,7 % |
| Sustituciones lineales de la llama | 0,542 | 4,4 % |

Son categorías inclusivas que se solapan. En particular, el Jacobiano vuelve
a calcular propiedades de caras; su coste ya incluye parte del transporte.
Este perfil es una muestra diagnóstica y no la mediana de la tabla anterior.

El desglose del transporte por cara usa 96 caras muestreadas de estados
convergidos y siete tandas calentadas, en recorrido serial independiente:

| Fase anterior, 53 especies | Mediana por cara |
|---|---:|
| Ajustes de temperatura y cp | 22,58 µs |
| Construcción e inversión de L00 | 58,85 µs |
| Sistema térmico de Schur | 109,09 µs |
| Aplicación del flujo multicomponente | 2,41 µs |

Las fases reciben sus entradas preparadas; no representan porcentajes de la
resolución completa. Se usó una lista tipada Numba para evitar que convertir
una lista Python a la entrada añadiera decenas de microsegundos ajenos al
cálculo. El primer ensayo con listas reflejadas se retiró de la evidencia.

El perfil posterior, también una sola muestra, dio 10,730 s: caras 4,865 s y Jacobiano 4,207 s. Los campos permanecieron exactos. Este segundo perfil permite localizar trabajo restante, pero comparar dos muestras diagnósticas no cuantifica por sí solo el efecto de cada cambio.

## Implementación seleccionada

| Cambio | Trabajo que se evita y conservación numérica |
|---|---|
| Preparación exclusiva de viscosidad en caras multicomponente | Calcula solamente `root` y `phi` del cierre Wilke con las expresiones originales. Conductividad, difusión ordinaria y Soret siguen viniendo del cierre multicomponente |
| Menos copias entre caras compartidas | En multicomponente no copia `cond`, `invB`, `diagB` ni `scale`, que no tienen consumidores en esa ruta |
| Reutilización del peso molecular de mezcla | Usa el `wm` devuelto por el cierre, calculado con la misma acumulación y el mismo clamp, en vez de repetir la suma para viscosidad |
| Productos Schur en salidas preasignadas | Tres `np.dot` escriben en scratch: dos matrices ya inactivas y el producto conservado. Mantiene las copias transpuestas y los flags de BLAS del original; resta cada elemento en el orden `(L11−producto01)−producto12` |
| Actualización LU con vistas de filas contiguas | Para matrices C de al menos 24 especies, expone actualizaciones independientes del tramo restante de fila. El orden de pivotes y cada multiplicación/resta siguen iguales; la ruta pequeña/F/con strides mantiene el bucle original |
| Ocho acumuladores independientes en el flujo | Para más de 20 especies procesa ocho salidas a la vez, conservando el orden ascendente de cada suma. Las especies restantes y mecanismos pequeños usan el recorrido escalar |

En 53 especies, la preparación exclusiva de viscosidad elimina 1.484
evaluaciones polinómicas de grado cuatro por preparación de cara, incluyendo
conductividad y datos binarios de mezcla que antes se calculaban sin usarse.
También elimina 1.378 recíprocos de pares. La ruta de mezcla mantiene su
preparación completa. No se sustituye `root` por `sqrt(mu)` ni se toma la
fracción molar truncada del sistema de difusión para viscosidad: ambas
sustituciones cambiarían los valores o el redondeo.

La reducción de copias base evita dos matrices de 53×53 y dos vectores de
53 elementos por transición reutilizada: 45.792 bytes de payload por
transición. Es una cuenta de arrays copiados; no se midieron bytes de DRAM
ni fallos L1/L2. Los placeholders del workspace se conservan para limitar
los cambios de interfaces; no se afirma una reducción del RSS.

En los productos Schur se reutilizan las matrices 0/1 cuando los buffers
son C contiguos; la inversa permanece en la matriz 2. Para otros layouts
del workspace se conserva el cálculo original. Cada rango tiene scratch
privado. Ningún array de entrada se modifica ni se retiene una salida
prestada a través de llamadas concurrentes.

Todos los kernels mantienen `fastmath=False`. La variante LU experimental
generó instrucciones SIMD de multiplicación/resta sin FMA en esta CPU;
esa observación del compilador no sustituye las pruebas de igualdad.
No se cambiaron Newton, pseudo-transitorio, adaptación, aceptación ni
tolerancias. Los cambios productivos de esta campaña se limitan a
[multicomponent_kernels.py](../src/kflame/chemistry/multicomponent_kernels.py)
y [diffusion_kernels.py](../src/kflame/flame/diffusion_kernels.py).

## Ensayos que no se seleccionaron

El agente térmico probó recorrido por filas en L11 conservando el orden
de cada suma: pasó las comparaciones exactas, pero su mediana emparejada
del kernel fue 0,916× respecto al anterior, por lo que se mantuvo el
recorrido por columnas. Pasar transpuestas como vistas a BLAS tampoco
se adoptó: se conservan las copias y orientación originales.

El agente de temperatura obtuvo igualdad byte a byte al evaluar solo el
triángulo de pares con coeficientes simétricos y reflejar los resultados.
El kernel mejoró aproximadamente 1,17× en GRI y 1,11× en H₂ en una tanda
corta. No se integró: exige certificar los cinco arrays reales de pares
en una copia privada e inmutable, con fallback para datos asimétricos o
mutables. Un nombre de mecanismo o un booleano en arrays modificables
no garantizan esa simetría. Es un candidato pendiente de mayor alcance.

## Literatura y siguientes estrategias

| Fuente primaria | Resultado pertinente y aplicación a este solver |
|---|---|
| [Chi, Hui y Wang, 2026](https://doi.org/10.3390/app16073611) | Organización para SIMD y localidad, tratamiento de varios RHS y operaciones por bloques en Maxwell–Stefan/OpenFOAM. Aquí la inversa grande y productos ya usan LAPACK/BLAS; se prioriza quitar trabajo y exponer operaciones independientes. Sus cocientes de velocidad no se trasladan a este código |
| [Ambikasaran y Narayanaswamy](https://arxiv.org/html/1512.01338v2) | Compresión de rango bajo de difusividades recíprocas y Sherman–Morrison–Woodbury con precisión solicitada. Incluye Soret a través de coeficientes térmicos suministrados; no elimina el coste de calcularlos ni es un reemplazo byte exacto del Schur actual |
| [Fillo et al.](https://arxiv.org/html/1808.05463) | Streaming de contribuciones y precondicionador semiimplícito; los casos del artículo omiten Soret/Dufour. Inspira vida corta de datos y evitar repetición, pero no valida esta ruta térmica ni su equivalencia byte a byte |
| [Córdoba y Arias-Zugasti, 2022](https://doi.org/10.1016/j.combustflame.2022.112202) | Algoritmo iterativo para difusión térmica y conductividad, implementado en MuTLib. Es una vía de investigación directamente relevante al coste térmico; requiere reconciliar modelos de colisiones/relajación, tolerancia de iteración y fallback directo |

Se consultaron texto completo HTML de los dos trabajos arXiv y texto
primario indexado del editor/abstract para los otros dos. No se verificaron
los detalles de parada ni la implementación de MuTLib. Los métodos iterativos,
compresión y cambios de precondicionador pueden conservar las ecuaciones
objetivo y aun así alterar la solución a tolerancia finita o sus últimos bits;
no están activados como parte de esta mejora exacta.

El próximo cambio de mayor interés es **evitar la segunda evaluación base
cuando el residual y un Jacobiano actualizado usan el mismo estado**.
Actualmente `jacobian_blocks(x)` vuelve a calcular química y todas las caras
base tras `residual(x)`. La propuesta es pasar explícitamente propiedades
producidas por el mismo kernel/estado al Jacobiano, con propiedad y vida útil
limitadas a esa actualización. Hace falta comprobar configuración, presión,
malla y que las evaluaciones C/F conservan bytes. No se implementa una caché
global o basada solo en identidad de un array mutable.

Después conviene medir rotación de buffers de caras en vez de copiarlos,
certificación de pares simétricos y reparto de rangos por mecanismo. Las
alternativas iterativas/rango bajo se deben evaluar por separado con errores
de coeficientes, flujos, conservación de masa y convergencia; mantener Soret
y la ecuación energética completa sigue siendo un requisito.

## Validación y reproducción

La suite final dio **554 passed, 1 warning in 478.11s**. Las nuevas
[pruebas del transporte](../tests/test_multicomponent_hpc.py) comparan contra
[kernels originales congelados](../tests/_multicomponent_reference.py),
independientes de los helpers nuevos. Cubren H₂/GRI, temperaturas de
200–3500 K, especies traza, composiciones no normalizadas, matrices C/F/con
strides, datos de solo lectura, scratch contaminado, ceros con signo,
pivoteo y fallo singular. La prueba de ensamblaje de caras reproduce el
protocolo anterior con Soret activado/desactivado y particiones independientes.
La suite existente cubre además Jacobianos exacto/congelado y el camino denso.

Las llamas completas comparan mallas, formas, tipos y bytes de los 21 campos
en seis configuraciones y tres repeticiones. Las muestras del código anterior
se conservaron antes de editar esta campaña. Las mediciones finales son
secuenciales; no compiten agentes o workers temporizados por CPU.

```powershell
python examples/benchmark_memory_layout.py --baseline-src .local/hpc-multi-2026-10-08/baseline-current/src --output runs/hpc_multicomponent --repeats 3 --threads 4
```

La carpeta de salida debe ser nueva. El benchmark es portable; el árbol
anterior y los scripts exploratorios de `.local/hpc-multi-2026-10-08/`
pertenecen a esta copia local ignorada por Git. Para la suite:

```powershell
$env:NUMBA_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m pytest -q
```
