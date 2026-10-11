# Localidad de memoria y buffers del solver

[Guía de lectura](README.md) · [Difusión](diffusion-flames.md) · [Revisión previa](reviews/repository-hpc-2026-10-08.md)

[Continuación: caras compartidas y derivadas por bloques](hpc-next-strategies.md)
presenta la campaña posterior con 463 pruebas y sus propios tiempos antes/después.
Este documento conserva los resultados de la primera campaña.

Esta revisión del 8 de octubre de 2026 trabaja sobre el código local, incluida
la familia radiativa y la regularización del estancamiento que ya existían.
El objetivo es reducir tráfico de memoria y asignaciones conservando las
ecuaciones discretas, el orden de las operaciones y las tolerancias.

Resultado de esta primera campaña: **269 pruebas aprobadas** y **18 comparaciones de llamas con
los 21 campos guardados exactamente iguales**. Las sustituciones por bloques
mejoran 1,13–1,78× en el microbenchmark emparejado. Las mediciones del solver
completo presentan variación entre campañas y no demuestran una aceleración
general. La [evidencia numérica](assets/diffusion/memory-layout-2026-10-08.json)
conserva también esos resultados mixtos.

## Cómo se organizan los arrays

La forma pública sigue siendo `(especies, nodos)` o `(especies, caras)`.
El orden de almacenamiento se elige según el recorrido del kernel:

| Datos | Recorrido caliente | Disposición |
|---|---|---|
| Composición para termoquímica y transporte | Nodo/cara exterior, especie interior | Se presta el array C/F existente; las vistas con strides se empaquetan en Fortran |
| Fuentes y entalpías del Jacobiano local | Estado perturbado exterior, especie interior | Columnas contiguas en el workspace |
| Ventanas de cinco nodos y cuatro caras | Nodo/cara exterior, especie interior | Backing array transpuesto con columnas contiguas |
| Bloques del sistema lineal | Un bloque denso por nodo | Bloques contiguos en orden C para BLAS |
| Vectores de sumas y scratch | Recorrido lineal de especies | Filas contiguas dentro de buffers por rango/hilo |

Para una matriz `float64` de 53 especies y 1322 nodos, el salto entre especies
del mismo nodo pasa de 10 576 bytes en orden C a 8 bytes en orden Fortran.
La columna completa ocupa 424 bytes. Esto favorece el acceso por nodo sin
cambiar los valores ni la forma de los arrays. Empaquetar datos también cuesta:
los arrays adecuados se reutilizan y las vistas con strides se empaquetan
una vez al entrar en la ruta compilada.

En la evaluación completa de propiedades se comparte el mismo array preparado
entre química y transporte de caras. Para la vista de composición del estado
Newton se elimina una segunda copia: 560 528 bytes por evaluación en el ejemplo
de 53 especies y 1322 nodos. Si el array ya es C/F contiguo se presta sin copia.
Las salidas públicas de transporte y los flujos mantienen su orden C para
preservar las reducciones posteriores de NumPy.

## Buffers con vida útil limitada

Las sustituciones por bloques reutilizan un vector de RHS y uno de producto;
los productos BLAS escriben en salidas preasignadas. El scratch pertenece a
cada llamada, incluida la ejecución concurrente. Las matrices inversas y de
acoplamiento mantienen su forma y su propiedad.

Se ensayó también scratch reutilizable en los barridos de factorización.
La muestra de 1322 bloques de 56 variables fue aproximadamente un 4,6 % más
lenta, aunque la de bloques pequeños mejoró. Esa variante se retiró: la
factorización conserva su implementación anterior y la optimización seleccionada
se concentra en las sustituciones y los kernels de propiedades/Jacobiano.

El Jacobiano conserva su ventana de cinco nodos. Los vectores de sumas de
mezcla, Soret y polinomios de termoquímica se reúnen en workspaces por rango y
se reutilizan entre nodos y columnas. Las entradas de especies ausentes se
reinicializan antes de reutilizar el scratch de Soret. El residual se reserva
sin rellenarlo de ceros porque todas sus filas se escriben antes de leerlas.

Los arrays públicos devueltos por una llamada son independientes de los de
llamadas posteriores. Los buffers temporales no se comparten entre rangos.
Las entradas prestadas solo se leen; las pruebas incluyen entradas de solo
lectura y composiciones con trazas y perturbaciones de Newton.

Para el solve por bloques, la reserva de temporales evitada en float64
contiguo es `40*(n-1)*nv` bytes por llamada: aproximadamente 2,82 MiB para
1322 bloques de 56 variables. Es volumen de asignaciones evitadas; el pico
de memoria residente requiere una medición distinta.

## Qué demuestra la validación

El [benchmark de comparación](../examples/benchmark_memory_layout.py) carga el
código anterior y el candidato en procesos separados. Usa las mismas entradas,
calienta la compilación de cada configuración y guarda todos los campos de las
llamas. Compara mallas y bytes de los campos numéricos sin introducir una nueva
tolerancia para esconder diferencias. Los tiempos de resolución, API y llamada
completa se conservan por separado.

```powershell
python examples/benchmark_memory_layout.py --baseline-src RUTA_BASELINE/src --output runs/memory_layout --repeats 3
python -m pytest -q --durations=10
```

`RUTA_BASELINE` debe ser una copia completa del código anterior, incluidos los
datos de mecanismos. En esta sesión se conserva localmente en
`.local/hpc-memory-2026-10-08/baseline/`; los resultados completos se guardan
en la misma carpeta de trabajo local.

La suite final tardó 21,12 s, con el aviso del caso deliberado `troe_zero` de
Cantera. Incluye geometría, fuentes, termodinámica, Jacobianos, radiación,
transporte y FGM, además de los nuevos casos de layouts C/F/strided, entradas
de solo lectura, concurrencia y propiedad de arrays. Los tests permanentes
funcionan sin la copia privada del código anterior.

Se comprobó además que los AST de los balances, radiación, Newton,
pseudo-transitorio, adaptación y aceptación final coinciden con el snapshot.
La comparación completa usa tres flamelets independientes por caso, sin
semillas de continuación, con cuatro hilos Numba y un hilo BLAS/OMP/MKL.

## Mediciones y alcance

El microbenchmark de sustituciones utiliza sistemas algebraicos sintéticos de
tamaños representativos, un hilo y siete pares alternados en el mismo proceso.
La compilación, calentamiento y comparación de resultados quedan fuera del
cronómetro. Los factores y soluciones se comprobaron byte a byte.

| Bloques × variables | Antes | Después | Antes/después |
|---|---:|---:|---:|
| 1322 × 56 | 6,792 ms | 6,023 ms | 1,13× |
| 1071 × 13 | 1,002 ms | 0,564 ms | 1,78× |

La factorización seleccionada coincide con la anterior. Sus pequeñas
diferencias de tiempo entre aliases son variación de medida, no una mejora.
El experimento C/F de química mostró beneficios dependientes del kernel y
del layout de entrada; por eso la política final conserva los arrays ya
contiguos y empaqueta solo vistas con strides.

En resolución completa se conservaron las medianas de todas las campañas:

| Caso | Código anterior, primera campaña | Versión final | Código anterior, repetición posterior |
|---|---:|---:|---:|
| H₂, mezcla | 0,700 s | 0,659 s | — |
| CH₄, mezcla | 3,649 s | 4,322 s | 4,363 s |
| H₂, multicomponente + Soret | 1,030 s | 1,286 s | 1,275 s |
| CH₄, multicomponente + Soret | 14,363 s | 18,281 s | — |
| H₂, mezcla con radiación | 0,671 s | 0,714 s | — |
| CH₄, mezcla con radiación | 3,500 s | 4,052 s | — |

El mismo código anterior varió 19,6 % en CH₄ y 23,7 % en H₂ multicomponente
entre campañas. En la comprobación posterior, CH₄ queda cerca de la versión
final y H₂ multicomponente es ligeramente más rápido con el código anterior.
Estos datos impiden atribuir una mejora global o una regresión general a la
disposición de memoria. El resultado demostrado es equivalencia exacta,
menos copias/asignaciones y mejora del kernel de sustituciones en los tamaños
ensayados; la velocidad total necesita una campaña con variabilidad controlada.

Los informes completos, logs, snapshot y scripts de auditoría/microbenchmark
están en `.local/hpc-memory-2026-10-08/`. La evidencia portable registra las
muestras, versiones, máscaras de hilos, hashes y cada diferencia de campo,
incluida la variante de factorización descartada. El hash final del paquete es
`26176d380892da4ca69c481a2c146da1c3a691dd6cea447668698ee751329327`.

El host de esta sesión es un Ryzen 9 7845HX, con 12 núcleos y 24 hilos lógicos.
El firmware informa 12 MiB de L2 y 64 MiB de L3 agregadas. La organización
propuesta favorece L1/L2 mediante recorridos contiguos y scratch pequeño.
Los tiempos no miden directamente fallos de caché: para atribuir una mejora
a L1/L2 haría falta una campaña adicional con contadores de hardware.

## Próximas oportunidades

1. Medir pico de memoria al crecer el mecanismo y limitar los factores
   retenidos según presupuesto, conservando la misma matriz lineal.
2. Evaluar derivadas químicas por rangos para evitar materializar todos los
   Jacobianos nodales a la vez; mantener el orden de las sumas de especies.
3. Medir recorridos y empaquetado por kernel antes de extender el orden
   Fortran a otras rutas: una disposición óptima depende del bucle que la lee.
4. Comparar rangos de trabajo con diferentes tamaños y afinidad solo después
   de establecer una medición estable con hilos efectivos registrados.
