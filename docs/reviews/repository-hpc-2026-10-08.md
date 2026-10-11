# Estado del repositorio y prioridades HPC

Fecha local: 8 de octubre de 2026, America/Guayaquil.

[Guía de lectura](../README.md) · [Difusión](../diffusion-flames.md) · [Protocolo de benchmark](diffusion-benchmark.md)

## Lectura en un minuto

La optimización de difusión ya está confirmada en `93b7d8b`: Jacobiano local
compilado, eliminación por bloques desde dos extremos, modelo lineal persistente
y radiación. Los cambios locales añaden FGM radiativo `(Z,C,h)`, continuación,
regularización del gradiente entálpico cerca del estancamiento y contaminantes.
La memoria del 7 de octubre que describe HPC como trabajo sin commit y el FGM
radiativo como pendiente quedó desactualizada.

**207 pruebas pasan**, pero se reprodujeron casos límite fuera de su cobertura.
La prioridad es verificar la solución lineal y reparar la interoperabilidad de
contaminantes con FGM 3D. Después conviene mejorar muestreo, control efectivo de
hilos y memoria. Las propuestas de este informe no son aceleraciones implementadas.

La revisión se dividió entre tres agentes: HPC, validación científica y
documentación. El agente principal comprobó Git, ejecutó la suite completa,
repitió benchmarks de H₂ y reprodujo los casos de factorización, muestreo,
pliegues, valores sin cobertura y configuración tardía de Numba. Los cambios de esta revisión son documentación
y evidencia JSON; los cambios numéricos existentes pertenecen al trabajo previo.

## Estado de Git y del entorno

| Elemento | Estado observado al iniciar la revisión |
|---|---|
| Rama | `feature/nonadiabatic-enthalpy` |
| HEAD | `93b7d8b` |
| Referencia remota local | 16 commits por delante de `origin/feature/nonadiabatic-enthalpy`; sin consultar ni actualizar el remoto |
| Archivos seguidos modificados | 8; principalmente difusión, FGM, README y tests |
| Entradas sin seguimiento | 11 entradas de `git status`, algunas carpetas con varios archivos |
| Suite | `python -m pytest -q --durations=10`: 207 aprobadas, 1 aviso, 20,17 s |
| Aviso | Cantera en el caso deliberado `troe_zero`; no hubo fallos |
| Integridad del diff | `git diff --check` sin errores |
| Python / NumPy / SciPy | 3.11.0 / 2.4.6 / 1.17.1 |
| Numba / Cantera / pytest | 0.65.1 / 3.2.0 / 9.0.3 |

Los archivos sin seguimiento incluyen el módulo y tests de contaminantes,
ejemplos nuevos, evidencia de tablas radiativas y material de tesis en la raíz.
Antes de preparar commits, separar por alcance: solver, tablas FGM,
contaminantes y documentación. `TFM_DEPURAR/`, `main_original.tex`, `src.zip`
y `output/` necesitan decidirse explícitamente como material de tesis,
instantánea o resultados; en esta revisión se conservan en sus rutas.

## Rendimiento comprobado en esta revisión

Se ejecutó el benchmark existente sin instrumentación, con calentamiento y
tres resoluciones independientes por caso y configuración. Primero se midió
un hilo y después cuatro; dentro de cada pareja, siempre nativo y luego
Cantera. OpenBLAS, OMP y MKL se configuraron a un hilo en ambos procesos.
No se ejecutaron otros benchmarks ni la suite a la vez. Son muestras locales,
sin aleatorización entre campañas y sin garantía estadística de escalabilidad.

| Caso adiabático | Hilos Numba | Nativo, mediana | Rango nativo | Cantera, mediana |
|---|---:|---:|---:|---:|
| H₂, mezcla | 1 | 0,804 s | 0,784–0,871 s | 0,893 s |
| H₂, mezcla | 4 | 0,672 s | 0,655–0,852 s | 0,843 s |
| H₂, multicomponente + Soret | 1 | 2,458 s | 2,360–2,660 s | 3,366 s |
| H₂, multicomponente + Soret | 4 | 1,240 s | 1,187–1,423 s | 3,345 s |

Las **12 comparaciones físicas pasan**. El cociente de medianas nativas entre
un hilo y cuatro es 1,20× en mezcla y 1,98× en multicomponente con Soret.
H₂ con un hilo queda cerca de Cantera; una muestra nativa fue ligeramente más
lenta. No se debe presentar una ventaja universal. El cociente de medianas
es distinto de la mediana de cocientes por pareja guardada en el benchmark.

[Evidencia portable con muestras, entorno, métricas y hash](../assets/diffusion/review-hpc-2026-10-08.json).
Hash del código en ambas campañas: `70f89271c815aa01b365fc6c2fcfd2654bc458a5b1d4a9b6da6bdf727b8d3976`.
Los informes completos y perfiles están localmente en
`.local/reviews/hpc-2026-10-08-t1/` y `hpc-2026-10-08-t4/`.
El log de tests está en `.local/reviews/pytest-2026-10-08.log`.

Los [datos adiabáticos del 7 de octubre](../assets/diffusion/hpc-adiabatic.json)
registran CH₄ en 3,62 s frente a 29,79 s y CH₄ multicomponente con Soret en
17,37 s frente a 84,56 s. Tienen dos repeticiones; el
[informe radiativo](../assets/diffusion/hpc-radiation.json) tiene una.
Sus hashes difieren del árbol actual. No se repitieron CH₄ ni radiación en
esta revisión y no se ha medido una campaña en un clúster.

## Hallazgos que requieren cambios

P1 indica prioridad alta; P2 indica el siguiente nivel. Los ejemplos algebraicos
aislan errores del algoritmo y no representan llamas físicas resueltas.

| Prioridad | Hallazgo y evidencia | Acción concreta y validación necesaria |
|---|---|---|
| P1 | **Solución lineal inexacta sin fallback.** [`BorderedFactor.solve`](../../src/kflame/flame/diffusion.py#L504) retorna la eliminación por bloques sin comprobar `J·dx − rhs`. Los pivotes se invierten en [`block_linear.py`](../../src/kflame/flame/block_linear.py#L56). Un sistema de 3 nodos y 4 variables, condición global ≈4,05, devolvió residuo infinito 1 frente a 0 con LU; el fallback quedó desactivado. | Calcular error hacia atrás con producto por bloques y recurrir a LU de la misma matriz si excede el límite. Probar pivotes locales pequeños con matrices globales bien condicionadas y medir el coste y frecuencia del fallback. La guarda no lineal final sigue activa; este caso demuestra una dirección incorrecta, no una llama falsa aceptada. |
| P1 | **Contaminantes y FGM 3D no interoperan con cobertura parcial.** [`pollutants.py:150`](../../src/kflame/pollutants.py#L150) pasa arrays completos a `pollutant_profiles`; [`diffusion.py:510`](../../src/kflame/fgm/diffusion.py#L510) guarda NaN fuera del manifold. Se reprodujo `ValueError: Require finite aligned species profiles and positive molecular weights`. | Aceptar ambos formatos de comparación: arrays 2D ya recortados y arrays 3D completos; seleccionar `covered` en estos últimos y preservar la evaluación de cobertura. Probar la auditoría directamente con la salida real del validador 3D y colas significativas sin cobertura. |
| P2 | **Muestreo adaptativo incumple su tolerancia cuando los dominios difieren.** [`_adaptive_mixture_grid`](../../src/kflame/fgm/diffusion.py#L56) elimina nodos exteriores antes de interpolar los extremos comunes; la tabla final usa el perfil completo. Dominios `[0,1]` y `[0.1,0.9]` producen error 0,375 con tolerancia 0,001. | Interpolar límites desde el perfil completo y medir reconstrucción sobre los mismos datos usados para construir la tabla. Probar dominios distintos y todos los campos controlados; informar fallo si el presupuesto de puntos impide cumplir la tolerancia. |
| P2 | **Los pliegues acumulados de Z pueden pasar como redondeo.** [`_mixture_columns`](../../src/kflame/fgm/diffusion.py#L113) examina incrementos individuales. Tres incrementos de `8e-7` acumulan una inversión de `2.4e-6`, mayor que `FOLD_TOLERANCE=1e-6`; se aceptó el perfil descartando tres puntos. | Medir la excursión acumulada respecto de la envolvente y comprobar también la variación de T, Y y fuentes antes de eliminar estados como redondeo. Añadir inversión acumulada y perfiles que cambian químicamente a Z casi constante. |
| P2 | **Los límites de hilos dependen del orden de importación.** [`__init__.py:5`](../../src/kflame/__init__.py#L5) establece variables de entorno al importar. El agente reprodujo NumPy previamente inicializado con 24 hilos efectivos aunque el entorno indicase 1. El principal confirmó Numba con 24 efectivos y 4 configurados; `numba.core.config.reload_config()` lanza `RuntimeError`. | Establecer política antes de inicializar bibliotecas o aplicar controles compatibles con pools existentes; registrar hilos efectivos. Probar órdenes de importación y overrides explícitos. Los benchmarks actuales configuran variables antes del proceso Python y no ejercitan ese caso límite. |
| P2 | **Ventanas de perfilado y `native_s` distintas.** [`compile_kernels`](../../src/kflame/flame/diffusion.py#L330) evalúa residual/Jacobiano antes del cronómetro de resolución de la línea 710; [`instrument_native`](../../examples/benchmark_diffusion_solver.py#L120) ya está activo. | Separar preparación, resolución y exportación; atribuir categorías dentro de la misma ventana. Las categorías inclusivas se solapan y no deben sumarse. Mantener tiempos sin instrumentación como referencia. |
| P2 | **Memoria del camino principal sin presupuesto global.** [`jacobian_blocks`](../../src/kflame/flame/diffusion.py#L370), las tres matrices de bloques y [`factor_two_ended`](../../src/kflame/flame/block_linear.py#L33) reservan arrays densos; el modelo permite dos factores. Estimación estática para 1.600 nodos y 500 especies: ≈21,1 GiB entre bloques y dos factores, antes de otros datos. | Estimar memoria antes de reservar, limitar la caché y evaluar derivadas por rangos. Medir pico de memoria, tiempo y precisión al aumentar especies/nodos. El límite de 96 MiB corresponde al camino coloreado alternativo. |
| P2 | **Dos referencias distintas para déficit de entalpía.** El máximo de generación usa `h_ad(strain,Z) − h(strain,loss,Z)` en [`diffusion.py:338`](../../src/kflame/fgm/diffusion.py#L338), mientras `delta_h` usa `h_ad(Z,C) − h` en la línea 457. | Alinear el diagnóstico con la definición declarada o nombrar ambas cantidades. Validar con una familia donde radiación cambie composición y progreso, no solo temperatura. |

Para reproducir el primer hallazgo, insertar en la primera variable de tres
bloques identidad la matriz tridiagonal `[[1e-20,1,0],[1,1,1],[0,1,1]]`,
con RHS `[1,2,3]`. Mantener una columna de Λ en el momento del nodo interior
y una fila de borde que seleccione ese momento. `BorderedFactor` devuelve
`[0,1,2]`; LU devuelve `[-1,1,2]`. El sistema ampliado sigue bien condicionado.

## Estado de los hallazgos (11 de octubre de 2026)

| Prioridad | Hallazgo | Estado |
|---|---|---|
| P1 | Solución lineal inexacta sin fallback | **Corregido.** Las eliminaciones por bloques comprueban el error hacia atrás normado de su primera resolución con un producto matriz-vector. Si supera 10⁻⁸, pasan a LU pivotada: LU dispersa en difusión; Thomas por bloques y, si también falla, LU dispersa en la premezclada. Se aplica a la factorización densa y a la estructurada. Las factorizaciones por tramos o in situ ya no guardan A, así que no se verifican, pero sus pasos se aceptan sobre el residual completo. `tests/test_linear_backward_error.py` reproduce el caso de esta revisión: ahora sale `[-1,1,2]` en lugar de `[0,1,2]`. Coste medido: 0–6 % en llamas completas |
| P1 | Contaminantes y FGM 3D con cobertura parcial | **Corregido.** La auditoría acepta las dos formas de comparación: estados cubiertos recortados (tabla `(Z,C)`) y arrays completos con NaN fuera de la tabla (`(Z,C,h)`). La auditoría de la tabla radiativa de CH₄ pasa con cobertura del 100 % en CO, NO, NO₂ y NOₓ. Hay prueba de regresión con el formato completo |
| P2 | Muestreo adaptativo con dominios distintos | **Corregido.** El muestreador interpola desde el perfil completo, como la tabla, y mide el error en los nudos interiores. Devuelve el error alcanzado (`mixture_sampling.achieved_relative_error`) y falla si el presupuesto de puntos no basta. Con el código anterior, la prueba de regresión daba un error almacenado de 0,71 con tolerancia 10⁻³ |
| P2 | Pliegues acumulados de Z | **Corregido.** Se rechaza cualquier caída de Z bajo su envolvente mayor que `FOLD_TOLERANCE`, en un paso o acumulada. Tres inversiones de 8·10⁻⁷ ya no pasan como redondeo |
| P2 | Dos referencias para el déficit de entalpía | **Nombrado.** `maximum_enthalpy_deficit_definition` documenta el déficit de generación (adiabático de la misma deformación a igual Z) frente a `delta_h = h_ad(Z,C) − h` de la consulta |
| P2 | Memoria sin presupuesto global | **Atendido en el camino de mecanismos grandes.** Factorización por tramos sin guardar A, inversas una sola vez y depósito de búferes acotado: Aramco 3.0 tiene un pico de 5,2 GB, frente a la estimación de 21 GiB ([detalle](../hpc-large-mechanisms.md)) |
| P2 | Hilos según el orden de importación | **Corregido.** Si NumPy o Numba se importaron antes que FlamPy, la política se aplica en tiempo de ejecución: BLAS a 1 hilo con `threadpoolctl` (nueva dependencia ligera) y Numba a 4 con `numba.set_num_threads`. Se respetan los valores explícitos del usuario. Antes quedaban 24 y 24. `kflame.effective_threads()` informa los hilos reales, que se guardan en los metadatos de cada llama. Hay pruebas para ambos órdenes de importación |
| P2 | Ventanas de perfilado | **Corregido.** `benchmark_diffusion_solver.py` registra bajo `preparation:` lo que ocurre durante `compile_kernels`. Las categorías de resolución comparten ventana con `native_s`; en H₂, la preparación son 0,57 s y la resolución 0,505 s |

## Orden propuesto para avanzar HPC

1. **Cerrar P1.** Verificación lineal y compatibilidad de contaminantes/FGM 3D,
   con regresiones que reproduzcan los casos anteriores.
2. **Consolidar la precisión del FGM.** Límites del dominio común, pliegues y
   definición de déficit. Una consulta más rápida necesita una tabla validada.
3. **Medir costes actuales.** Hilos efectivos, tiempos de preparación y
   resolución, Jacobianos construidos/reutilizados y frecuencia de LU. Comparar
   1, 2, 4 y 8 hilos con orden alternado y casos nuevos; tres a cinco repeticiones
   son un punto de partida, aumentando muestras si la variabilidad lo exige.
4. **Controlar memoria y asignaciones.** Evaluación química por rangos y caché
   limitada, conservando salidas y derivadas. Medir tiempo y pico de memoria.
5. **Evaluar paralelismo entre familias.** Comparar procesos para ramas
   independientes de strain y continuación secuencial dentro de cada rama de
   pérdidas. Limitar hilos por proceso para evitar sobreasignación y conservar
   semillas, aceptación y metadatos. Es una propuesta, sin aceleración medida.

El siguiente cambio debe mantener las guardas físicas y mostrar coste y
precisión en condiciones reservadas que no participaron en su ajuste.
La campaña actual permite evaluar H₂ localmente; CH₄ y las familias radiativas
necesitan sus propias mediciones después de cada cambio.

## Organización aplicada para lectura rápida

Se añadió [un índice por objetivo](../README.md), accesible desde el README de
la raíz. El catálogo de ejemplos incluye difusión, radiación, contaminantes y
benchmark, y corrige el mapa H₂ a 86 flamelets y 2001 puntos de progreso.
Los tiempos históricos se identifican como tales en las guías de difusión y
contaminantes. Las revisiones antiguas conservan sus checkpoints y se leen como
antecedentes, junto a este estado vigente.
