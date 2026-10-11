# HPC: caras compartidas y derivadas por bloques

[Guía de lectura](README.md) · [Primera campaña de memoria](hpc-memory-layout.md) · [Evidencia completa](assets/diffusion/hpc-next-2026-10-08.json)

[Campaña posterior de transporte multicomponente](hpc-multicomponent.md):
perfil, mejoras del sistema térmico y 554 pruebas. Este documento conserva
las mediciones anteriores de caras compartidas y derivadas por bloques.

La continuación del 8 de octubre de 2026 aplica dos mejoras: reutilizar la
cara base compartida entre nodos y calcular las derivadas químicas en bloques
de 32 nodos con almacenamiento contiguo por nodo. **Pasan 463 pruebas y las
18 comparaciones de llamas completas conservan los 21 campos byte a byte.**
El punto de partida ya contiene las mejoras de la primera campaña; estas
mediciones comparan esa versión local con la nueva implementación.

## Antes y después

Tiempos de resolución nativa, incluida la adaptación y los diagnósticos del
solver, con compilación previamente calentada. Medianas de tres ejecuciones
por caso, cuatro hilos Numba y un hilo BLAS. Las versiones se ejecutaron en
procesos separados y secuenciales. La preparación y exportación de archivos
quedan fuera de este tiempo; el JSON guarda además los tiempos de API y pared.

| Llama completa | Antes (s) | Ahora (s) | Antes / ahora |
|---|---:|---:|---:|
| H₂, transporte de mezcla | 0,7262 | 0,6307 | 1,15× |
| CH₄, transporte de mezcla | 3,7052 | 3,5662 | 1,04× |
| H₂, multicomponente y Soret | 1,1336 | 1,0574 | 1,07× |
| CH₄, multicomponente y Soret | 15,1351 | 13,4185 | 1,13× |
| H₂, mezcla con radiación | 0,7019 | 0,5657 | 1,24× |
| CH₄, mezcla con radiación | 3,6257 | 3,0507 | 1,19× |

Estos cocientes describen esta campaña local. Se repitió el código anterior
después del candidato para CH₄: las medianas fueron 3,8076 s en mezcla y
14,9923 s en multicomponente. En este último control las muestras abarcan
13,7036–15,3412 s, por lo que existe variación apreciable. Las seis muestras
de control también conservaron los resultados exactos. Tres repeticiones
no establecen una aceleración universal ni un intervalo de confianza.

El microbenchmark del Jacobiano usa siete muestras alternadas en el mismo
proceso, estados finales conservados y comparación de matrices fuera del
temporizador. Mide caras y derivadas seleccionadas sin la resolución completa:

| Jacobiano | Ruta anterior (ms) | Ruta seleccionada (ms) | Cociente |
|---|---:|---:|---:|
| H₂, mezcla | 13,159 | 12,786 | 1,03× |
| CH₄, mezcla | 187,394 | 181,765 | 1,03× |
| CH₄, multicomponente y Soret | 661,279 | 557,336 | 1,19× |

En CH₄ de mezcla, reutilizar caras sin bloques dio 173,483 ms, mejor que
los 181,765 ms de la ruta seleccionada. Los bloques aportan una reducción
considerable del almacenamiento y favorecieron el caso multicomponente;
no son la variante más rápida en cada microbenchmark.

## Trabajo de los tres agentes

| Área | Implementación y decisión |
|---|---|
| Transporte y Jacobiano | Reutilización de la cara base derecha como izquierda del nodo siguiente; pruebas contra la ruta sin reutilización |
| Sistema por bloques | API de sustitución con buffers aportados por el llamador; la reutilización automática entre llamadas se retiró por falta de ventaja estable |
| Química y paralelismo | Derivadas por rango, matrices contiguas por nodo, integración por bloques y comprobación contra el kernel paralelo original |

Las mediciones finales se ejecutaron secuencialmente una vez terminados los
agentes. Los cambios productivos de esta continuación están en
[analytic.py](../src/kflame/chemistry/analytic.py),
[diffusion.py](../src/kflame/flame/diffusion.py),
[diffusion_kernels.py](../src/kflame/flame/diffusion_kernels.py) y
[block_linear.py](../src/kflame/flame/block_linear.py).

## Localidad de memoria y caras compartidas

La forma lógica de las derivadas continúa siendo `(especies, nodos, especies)`.
La nueva ruta por rango usa un array base `(nodos_del_bloque, especies, especies)`
y una vista transpuesta: la matriz de un nodo queda contigua para su consumo
inmediato por el Jacobiano. Se calcula un bloque, se ensambla y se liberan sus
derivadas antes de producir el siguiente. La ruta completa sigue disponible
con `problem.chemistry_tile_nodes = 0` para diagnóstico y comparación.

Para 53 especies y 1322 nodos, contando las derivadas respecto a especies y
temperatura en `float64`:

| Almacenamiento de derivadas | Bytes | MB decimales |
|---|---:|---:|
| Banco de toda la malla anterior | 30 268 512 | 30,27 |
| Un bloque de 32 nodos | 732 672 | 0,73 |
| Hasta cuatro bloques simultáneos | 2 930 688 | 2,93 |

Cada matriz de derivadas de especies ocupa 22 472 bytes. El presupuesto de
derivadas por bloque es 1 MiB; para 53 especies permite hasta 45 nodos, aunque
el valor predeterminado es 32. En los ensayos el ajuste solicitado de 64 quedó
limitado a 45. La cota admite al menos un nodo para mecanismos cuyo nodo ya
supere el presupuesto. La tilificación se activa para más de 20 especies.

Esto reduce el conjunto de datos de derivadas que permanece vivo y acerca
producción y consumo. Es una mejora de localidad pertinente para L1/L2;
no se han medido fallos de caché, ancho de banda ni RSS. Los números excluyen
concentraciones preparadas, bloques del Jacobiano, factores y scratch de
transporte, y no describen la memoria total del solver.

En el recorrido de nodos, la cara derecha base se copia al workspace de la
izquierda del siguiente nodo. Se reutilizan propiedades base de mezcla,
multicomponente y coeficientes congelados; las caras perturbadas mantienen
sus cálculos propios. Cada rango y cada bloque comienza con ambas caras:
para N nodos interiores repartidos en R rangos/bloques se evitan N−R cálculos
de cara izquierda base. El scratch pertenece a cada rango, sin estado global
compartido. El paso completo de caras del residual mantiene su implementación.

La preparación y validación químicas se hacen una vez para toda la malla,
conservando el orden de las reducciones de NumPy. Los bloques usan un kernel
serial `nogil` dentro del reparto externo del Jacobiano, evitando introducir
un segundo reparto paralelo dentro de cada bloque.

## Conservación numérica y variantes retiradas

Mover un `np.dot` del kernel paralelo original a un helper serial produjo
diferencias pequeñas en las llamas completas, aunque las primeras pruebas
entre las rutas nuevas coincidían. Esa variante se rechazó. En el entorno
Numba 0.65.1 usado, dos acumulaciones de especies se restauraron mediante
bucles escalares en el orden del kernel paralelo original, sin `fastmath`.
La equivalencia se comprobó contra una copia independiente del kernel
original en [la referencia de pruebas](../tests/_analytic_partials_reference.py).
La distinción entre reducciones paralelas y serialización de bucles anidados
está documentada en [Numba](https://numba.readthedocs.io/en/stable/user/parallel.html);
la igualdad de esta implementación se establece mediante nuestras pruebas.

La reutilización automática de buffers entre sustituciones tampoco se
seleccionó: sus medianas fueron aproximadamente un 1–4 % peores en los tres
casos del microbenchmark. Se conserva `twisted_solve_into` para llamadores
que aporten buffers privados; `BorderedFactor` conserva la ruta de asignación
por llamada anterior. Una salida anterior no se sobrescribe al resolver otro
RHS, incluidas llamadas concurrentes.

Las rutinas Newton, pseudo-transitorio, aceptación, adaptación y ecuaciones
del residual conservan su implementación respecto al punto de partida.
La auditoría AST guarda esas invariantes; para la química se usa además la
referencia numérica independiente. No se relajaron tolerancias.

## Verificación y reproducción

La suite completa ejecutada con cuatro hilos Numba y BLAS/OMP/MKL limitados
a uno dio **463 passed, 5 warnings in 56.92s**. Cuatro avisos de rendimiento
proceden de arrays deliberadamente no contiguos en las referencias de pruebas;
el otro es el aviso existente de Cantera para `troe_zero`.

Las pruebas cubren layouts C/F/vistas con strides, entradas de solo lectura,
aliasing y vida útil de salidas, scratch contaminado y llamadas concurrentes.
Doce casos de derivadas comparan ambas rutas con la referencia paralela
original, incluidos estados con signos y fronteras de polinomios NASA.
Los seis casos completos se comparan tres veces: misma malla, formas, tipos
y bytes de los 21 campos guardados, incluidos ceros con signo.

Para repetir la comparación desde esta copia local, conservando el árbol de
código anterior y usando una carpeta de salida nueva:

```powershell
python examples/benchmark_memory_layout.py --baseline-src .local/hpc-next-2026-10-08/baseline-current/src --output runs/hpc_next --repeats 3 --threads 4
```

Para repetir la suite en PowerShell:

```powershell
$env:NUMBA_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m pytest -q
```

El benchmark es portable, pero el árbol anterior y los scripts exploratorios
de `.local/hpc-next-2026-10-08/` pertenecen a esta copia de trabajo y están
ignorados por Git. El JSON enlazado conserva muestras, versiones, hashes del
código, controles, perfil diagnóstico y ensayos de hilos.

## Próximas estrategias, por prioridad

| Estrategia | Evidencia disponible | Próximo experimento y criterio |
|---|---|---|
| Ajustar bloques y hilos por mecanismo | En esta campaña 32 nodos favorecen CH₄ multicomponente; en mezcla solo caras fue más rápido en el microbenchmark | Repetir barrido de 16/32/45 nodos y 2/4/8 hilos sobre el código final con controles alternados; aceptar según tiempo completo y bytes exactos |
| Reducir trabajo de transporte multicomponente | El perfil anterior atribuye 6,71 s a caras en una muestra de 14,81 s | Perfilar de nuevo la versión final; buscar sumas repetidas y datos inmutables del mecanismo que puedan reutilizarse por rango, conservando el orden de acumulación |
| Optimizar factorización en mallas pequeñas | Cerca del 90 % de las llamadas de factor se realizaron con 31 nodos en el perfil anterior; falta el coste agrupado por tamaño | Medir tiempo por tamaño antes de ensayar kernels para bloques pequeños; mantener operaciones y fallback, exigir equivalencia exacta |
| Paralelizar flamelets independientes | La resolución individual ya usa varios hilos; el barrido anterior no favorece ocho hilos para H₂ | Ensayar procesos con pocos hilos por llama sobre casos independientes y comparar tiempo total de una familia; respetar dependencias de continuación y comprobar cada salida |

El perfil diagnóstico es una muestra por caso del código anterior: sus tiempos
incluyen categorías anidadas que se solapan y no se deben sumar. El barrido
anterior dio H₂ 0,603 s con dos hilos frente a 1,066 s con ocho; CH₄ dio 3,500 s
con ocho frente a 3,514–3,689 s con cuatro. Es una razón para medir por mecanismo,
no para cambiar globalmente a ocho hilos. No se activan todavía estas propuestas.
