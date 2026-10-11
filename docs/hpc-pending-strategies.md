# Pruebas de las estrategias HPC pendientes

[Inventario actualizado de las 15 propuestas](hpc-strategy-status.md) · [Transporte multicomponente](hpc-multicomponent.md)

[Muestras de la campaña, todavía parcial](assets/diffusion/hpc-pending-2026-10-09.json)

Campaña iniciada el 9 de octubre de 2026. **En curso: todavía no se han probado
ni aceptado todas las estrategias.** El punto de partida está preservado en
`.local/hpc-pending-2026-10-09/baseline-current/src`; se mantienen candidatas
separadas para medir cada cambio. Los cambios del árbol de trabajo son
candidatas hasta superar las comprobaciones numéricas y de rendimiento.

La ampliación posterior ya tiene una campaña independiente cerrada de
validación: **853 pruebas, nueve mecanismos contra Cantera y doce pares de
llamas completas exactos**. [HPC por mecanismo](hpc-mechanisms.md) muestra el
antes/después de las dos mejoras de memoria y la decisión de mantener CPU
tras medir GPU contra cuatro/ocho hilos. Esa referencia tiene 69 archivos;
los pilotos originales siguientes conservan la referencia de 60 archivos.

| Estrategia | Evidencia obtenida | Trabajo restante |
|---|---|---|
| E: compartir propiedades residual → Jacobiano | 14 pruebas de la candidata aislada aprobadas; siete pares de kernels exactos y 66 pares exactos de llamas completas, incluidos 30 pares alternados nuevos | Cerrar la aceptación junto con las demás candidatas |
| B: rotar buffers completos | Pruebas numéricas de la candidata conjunta aprobadas; seis pares de llamas completas exactos de B aislada | Control alternado, medición conjunta y decisión de selección |
| D: compactar workspace multicomponente | Seis pares aislados exactos; H₂-MS 0,973× y CH₄-MS 0,959×, sin aceleración observada | Incluir asignaciones y decidir selección |
| Simetría certificada de pares | 50 pruebas específicas aprobadas; conexión compilada validada en las 145 pruebas afectadas del conjunto | Medirla separadamente en llamas completas, con certificación/preparación incluidas |
| Schur con transpuestas como vistas | 590 composiciones físicas exactas; tres de 68 productos matriciales difieren en bits | Variante rechazada bajo el requisito de igualdad exacta; no se ejecuta su benchmark completo |
| Coeficientes base y semillas entre rangos | Ocho registros exactos por modo tras corregir Jcopy. Banco completo: 1,074–1,107× en CH₄-MS y pérdida en H₂-MS. Seeds construye todavía el banco completo y no da ganancia uniforme | Los timings anteriores a Jcopy están retirados. Falta banco limitado a fronteras y ensayo de llamas completas |
| Bloques/hilos | 32 configuraciones más cuatro controles de hilos, exactos en residual/Jacobiano/solve. Se conserva tile32; no hay ganancia uniforme de otro tile | Mayor cobertura y comparación de llamas completas; cuatro hilos frente a uno es configuración, no una nueva ganancia de código |
| Factor por tamaño | 18 configuraciones exactas y backward error válido; ventajas dependen de tamaño/shift | Repetición y selección sobre llama completa |
| Presupuesto de memoria e importación | Siete transiciones de caché exactas y presupuesto de 4 GiB aprobado en GRI/65 nodos; 36 diagnósticos de hilos/importación aprobados | No generalizar el presupuesto a todos los mecanismos ni confundir arrays retenidos con RSS/JIT |
| Procesos entre familias | Tres tandas completas y exactas: 48,7/49,2/49,6 s en serie frente a 26,7/26,8/26,7 s con dos procesos; mediana emparejada 1,835×. Memoria y cierre de trabajadores aprobados | Cobertura de otros mecanismos/continuaciones; no atribuir throughput a aceleración de una llama |
| Alternativas de papers: rango bajo, iteración térmica, precondicionador | Recíproco con diagonal: rango completo en nueve mecanismos; SMW cambia bits. Neumann converge pero cambia bits. Los 60 solves GMRES convergen, cambian bits y son más lentos | No se seleccionan bajo igualdad estricta. Los prototipos inspirados en papers no equivalen a reproducir todas sus formulaciones |

El seguimiento terminado está en `completed_followup` de
[las muestras publicadas](assets/diffusion/hpc-pending-2026-10-09.json).
Las muestras y el ledger anteriores se conservan como checkpoints históricos.
El primer intento del barrido de tiles completó doce configuraciones exactas
antes de fallar la sustitución atómica del informe por un bloqueo temporal de
Windows. El segundo intento, con reintentos de escritura fuera de los timers,
terminó las 32 configuraciones y los cuatro controles; no cambió la fuente
numérica congelada.

El presupuesto medido de memoria para GRI/65 nodos distingue 10,92 MiB de
arrays/factores retenidos, 401,05 MiB del plan con reservas y 901,10 MiB de
pico de working set del proceso, incluido JIT. El plan no acota ese pico.
El piloto de familias alcanzó 3571,32 MiB sumados entre trabajadores,
dentro de su presupuesto explícito de 8192 MiB; esas observaciones no son
una cota para todas las mallas o mecanismos.

## Primer experimento: reutilización de propiedades

Medianas de siete mediciones alternadas del **paso residual + Jacobiano
actualizado**, no de la llama completa:

| Caso | Preparación independiente (ms) | Reutilización (ms) | Cociente |
|---|---:|---:|---:|
| H₂ multicomponente + Soret | 57,71 | 53,28 | 1,083× |
| CH₄ multicomponente + Soret | 455,14 | 415,33 | 1,096× |

Se incluye preparar las propiedades y convertir las salidas nodales a C;
se excluyen compilación y comparación de bytes. Se usan cuatro hilos Numba
y uno BLAS/OMP/MKL. Los estados guardados tienen 1.043 y 1.273 nodos,
respectivamente. Muestras: `reuse-micro.json` en la carpeta local de campaña.

La primera implementación compartía también datos nodales en la ruta NumPy
alternativa. **Seis casos fallaron la comparación exacta**: su preparación F
del residual redondea distinto a la preparación C anterior del Jacobiano.
Se conservó ese resultado en `pytest-base-reuse-initial-layout-failure.log`.
La candidata corregida conserva la evaluación nodal original cuando no se
usa el kernel nativo fusionado, y comparte las caras. En la ruta fusionada
comparte ambas partes. No se relajaron tolerancias.

El driver guarda las propiedades del estado base en variables locales de
Newton. Las tentativas rechazadas no sustituyen esa tupla. No se crea una
caché global por identidad de arrays; no se retiene la tupla entre llamadas,
mallas o problemas. Los tests incluyen rechazo de tentativa y actualización
con Jacobiano envejecido. Las rutas de problemas genéricos o residual sin
caras compiladas conservan su evaluación original.

## Medición y aceptación

El primer benchmark completo usa el protocolo anterior de calentamiento sin
adaptación. El primer caso H₂ produjo una muestra inicial más lenta que las
siguientes; se conserva como diagnóstico y no se borra. Se añadió
`--warm-refined` al benchmark para repetir con una resolución adaptativa
completa fuera de los tiempos. Las comparaciones deberán distinguir ambos
protocolos; estos tiempos no acreditan por sí solos una ganancia final.

La aceptación exige formas, tipos y bytes idénticos contra la fuente
preservada, sin cambiar ecuaciones ni criterios de convergencia. Después se
decide según mediciones calientes por estrategia y llama completa. Una
variante probada puede quedar sin seleccionar; preparar un driver no equivale
a haber ejecutado el experimento.

La repetición con `--warm-refined` terminó: las 18 comparaciones adicionales
conservaron exactamente los 21 campos. En CH₄ multicomponente la mediana pasó
de 9,609 a 8,863 s (1,084×). El primer protocolo dio 9,283 a 8,332 s (1,114×).
Son tandas secuenciales de esta máquina. H₂ multicomponente y los controles de
mezcla/radiación muestran variación entre tandas; no se afirma una ganancia
uniforme ni se atribuye a E toda diferencia pequeña.

### Control alternado terminado: E aislada

Se completaron cinco pares por cada uno de los seis casos, con ambos
recorridos calientes y la misma fuente aislada de E. Las 30 comparaciones
conservan formas, tipos y los bytes de los 21 campos. Junto a las tandas
anteriores hay 66 pares completos comprobados para E.

| Caso | E desactivada, mediana (s) | E activada, mediana (s) | Mediana de cocientes por par | Pares más rápidos |
|---|---:|---:|---:|---:|
| H₂ | 0,4584 | 0,4558 | 1,011× | 3/5 |
| CH₄ | 2,5966 | 2,4756 | 1,063× | 4/5 |
| H₂-MS | 0,7721 | 0,7522 | 1,026× | 4/5 |
| CH₄-MS | 8,7271 | 8,1789 | 1,069× | 5/5 |
| H₂ con radiación | 0,4988 | 0,4921 | 0,993× | 2/5 |
| CH₄ con radiación | 2,5636 | 2,4664 | 1,038× | 5/5 |

La mediana de los cocientes emparejados puede diferir del cociente entre
medianas de tiempos. Estos resultados respaldan el ahorro de E en CH₄-MS;
no prueban una mejora uniforme en todos los mecanismos. Evidencia local:
`reuse-full-paired/comparison.json`; copia en las muestras publicadas.

### B aislada y pruebas del conjunto

B aislada terminó tres comparaciones completas por caso, todas exactas:
H₂-MS pasó de 0,7993 a 0,7728 s (1,034×) y CH₄-MS de 8,7785 a 8,8052 s
(0,997×). El protocolo ejecuta baseline y después candidata; falta un
control alternado antes de atribuir diferencias pequeñas a B. La reducción
de copias no acredita por sí sola una ventaja temporal.

La integración inicial dio 294 pruebas aprobadas y dos fallos al desempaquetar
la tupla histórica `multi` de 14 entradas. Se restauró esa interfaz y se
añadió `multi_certified` con 15 entradas para el consumidor privado. Después
aprobaron las 145 pruebas afectadas en 27,79 s. No se suman ambos recuentos
porque se solapan; sigue pendiente la suite final de la versión seleccionada.

La cola `core-queue.json` terminó sus cinco pasos. B+D dio seis pares exactos,
con cocientes 1,021× en H₂-MS y 1,002× en CH₄-MS. El conjunto E+B+D+simetría
dio 18 pares exactos; CH₄-MS pasó de 8,960 a 8,288 s (1,081×). Los pequeños
cambios de B+D no demuestran una aceleración consistente. Estos resultados
son anteriores a la ampliación del catálogo químico y del soporte PLOG/SRI:
se conservan los hashes de aquella fuente y sigue pendiente comprobar el
árbol posterior antes de cerrar la selección final.

Los informes de los agentes están en las subcarpetas locales `rotation/`,
`symmetry/` y `experiments/`. El inventario de ejecución es `campaign.json`.
Estas carpetas son locales e ignoradas por Git. Este documento mantendrá
separados los resultados terminados y el trabajo pendiente.
