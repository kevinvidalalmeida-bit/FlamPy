# Estado de las propuestas HPC de los agentes

**Mecanismos grandes, 10 de octubre:** [Jacobiano estructurado, Newton–Krylov y transporte vectorizado](hpc-large-mechanisms.md): AramcoMech 3.0 completa pasa de no terminar a 115 s; JetSurF 3,6×; GRI 1,5×.

**Jacobiano térmico, 10 de octubre:** [derivadas implícitas y tiempos inclusivos](hpc-thermal-jacobian.md), con validación contra Cantera y opción experimental desactivada.

**Continuación del 10 de octubre:** [derivadas y actualizaciones algebraicas](hpc-analytic-transport.md), con tiempos de la fuente integrada, controles contra Cantera y opciones experimentales desactivadas por defecto.

**Última continuación del 9 de octubre:** [perfil de Aramco3 y columnas de transporte por bloques](hpc-jacobian-breakdown.md), con fuente final, Cantera y reparto interno del Jacobiano.

**Mejora anterior del 9 de octubre:** [banco de transporte compartido entre propiedades y Jacobiano](hpc-transport-bank.md), con tiempos de la fuente final y comprobaciones dentro de Newton.

**Nueva continuación del 9 de octubre:** [reutilización exacta de coeficientes entre columnas de temperatura](hpc-temperature-reuse.md), con tiempos de la fuente integrada y controles sin reutilización.

**Continuación del 9 de octubre:** [reparto por rangos en mecanismos grandes y mallas pequeñas](hpc-small-grids.md), con la fuente final validada y sus límites de medición.

**Revisión posterior del 9 de octubre:** [nuevo perfil, optimizaciones integradas y tiempos reales contra Cantera](hpc-reprofile.md). Esta revisión usa una nueva referencia de 70 archivos; las mediciones históricas que siguen conservan su alcance original.

[Guía de lectura](README.md) · [Mediciones multicomponente](hpc-multicomponent.md) · [Campaña anterior](hpc-next-strategies.md)

[Campaña iniciada el 9 de octubre](hpc-pending-strategies.md)

## Revisión actual: 9 de octubre de 2026

**Las 15 estrategias originales no están todas completadas ni aceptadas.**
Esta revisión incorpora los reportes medidos C-full/seeds-Jcopy,
small-factor-first, thread-policy-first, tiles-inprocess y memoria. Distingue exactitud local, ganancia
local y aceptación en llama completa. Preparar o corregir un driver no equivale
a ejecutar su validación. El checkpoint del 8 de octubre se conserva íntegro
más abajo, incluidas sus tablas y su orden de continuación histórico.

Las mejoras A (`wm`), D (solo viscosidad y copias muertas), `dot_out`, LU
condicional y flujo con ocho acumuladores pertenecen a la campaña aceptada del
8 de octubre. No forman parte de los quince pendientes originales del ledger
`campaign.json`. Ese ledger y algunos índices locales todavía describen etapas
anteriores; aquí se usa el contenido de los reportes posteriores para actualizar
el estado, sin reescribir la evidencia histórica.

La fuente posterior a la ampliación de mecanismos tiene **853 pruebas
aprobadas**, transporte/cinética contra Cantera en nueve mecanismos y doce
pares exactos de llamas completas. Las dos mejoras nuevas de memoria y la
comparación GPU frente a CPU paralela están en [HPC por mecanismo](hpc-mechanisms.md).
Esa campaña usa una referencia de 69 archivos; los pilotos originales de
esta tabla que citan 60 archivos mantienen su referencia anterior.

### Inventario actual de las 15 estrategias y trabajo restante

| Nº / ID original | Estado comprobado | Tarea todavía pendiente |
|---|---|---|
| 1. `E`: propiedades residual → Jacobiano | Candidata con 66 pares exactos de llamas completas; 30 pares alternados. CH₄-MS: 8,727 → 8,179 s y 1,069× emparejado | Cerrar selección y validación del árbol final; no atribuirle automáticamente tiempos de otras candidatas |
| 2. `B`: rotación completa de buffers | Seis pares aislados exactos: H₂-MS 1,034×; CH₄-MS 0,997×. B+D también tiene seis pares exactos, con 1,021× y 1,002× | Control alternado e integración seleccionada; no hay ganancia consistente demostrada de B |
| 3. `D-compact`: workspace | Seis pares aislados exactos: H₂-MS 0,973×; CH₄-MS 0,959× | Decidir selección con coste de asignaciones y llama completa; reducir memoria no demuestra aceleración |
| 4. `symmetry`: pares certificados | Snapshot privado/inmutable y fallback comprobados; pruebas específicas y del conjunto. El conjunto E+B+D+simetría tiene 18 pares completos exactos | Medir simetría aislada en llama completa con preparación/certificación incluidas; seleccionar sobre fuente final |
| 5. `dot_views`: transpuestas BLAS | Rechazada bajo igualdad estricta: 590 composiciones físicas exactas, pero tres de 68 productos difieren en bits | No activar ni medir la integración completa de esta variante rechazada; reconsiderarla sólo con una variante que pase el gate original |
| 6. `C`: banco completo | Histórico Jcopy: ocho registros exactos, con ganancia local sólo en CH₄-MS. Nueva variante D/DT/lambda para Ns≥256 integrada y validada sobre la fuente actual: [banco limitado a 96 MiB](hpc-transport-bank.md) | La variante nueva tiene ensamblados exactos y Newton limitado exacto; faltan nuevas llamas completas grandes, bancos que exceden el límite y cierre de la propuesta C original |
| 7. `C-seeds`: fronteras de rangos | Jcopy: ocho registros exactos; ganancia sólo en CH₄-MS con siete particiones. Construye el mismo banco completo que C | Probar preparación limitada a fronteras como variante separada, mallas mayores y llamas completas; no atribuirle menor memoria todavía |
| 8. `tiles-threads`: bloques/hilos por mecanismo | Barrido completo: CH₄ y CH₄-MS, 33/65 nodos, tiles 0/16/32/64 y 1/4 hilos; 32 configuraciones más cuatro comparaciones de hilos, todas exactas en residual, cinco bloques y solve | Se mantiene tile32: no hay ganancia uniforme por cambiar tile. Cuatro hilos dan 2,09–3,02× frente a uno en estos estados; ya eran la configuración por defecto y no son una nueva ganancia de código |
| 9. `small-factor`: factor por tamaño | Piloto de 18 configuraciones exactas, con backward error válido; ventajas dependientes de caso/shift | Repetir con control de variación, cubrir tamaños/shifts restantes y decidir política mediante llama completa |
| 10. `flame-processes`: familias independientes | Piloto completo: tres tandas de cuatro llamas H₂/CH₄-MS; uno frente a dos procesos, Numba=1 y BLAS=1 por proceso. Ganancia emparejada 1,835×; perfiles, warmup, memoria y cierre de nueve trabajadores aprobados | No es aceleración de una llama ni una política automática de producción. Falta cobertura de continuaciones, cuatro procesos y otros mecanismos |
| 11. `memory-budget`: memoria/factores | Diagnóstico preservado y validación separada aprobada: siete transiciones nativas, soluciones exactas y presupuesto de 4 GiB para un proceso GRI/65 nodos. La condición original del driver fallaba en un cache hit válido | La política de producción no cambia. El plan de arrays más reservas no es una cota del RSS/JIT; ampliar cobertura de mecanismos/mallas y concurrencia requiere otra medición |
| 12. `thread-policy`: hilos/orden de importación | 36 diagnósticos completos: gates de política y producto exacto aprobados | Comprobar control del proceso real y rendimiento de llamas con hilos; no resuelve por sí sola un pool Numba ya inicializado |
| 13. `research-low-rank`: PPACA/rango bajo | Diagnóstico en nueve mecanismos: el recíproco de difusión, incluida la diagonal modificada, conserva rango completo en cuatro tolerancias. Las inversas SMW no coinciden byte a byte | Esta representación no se selecciona. No reproduce el PPACA optimizado ni demuestra que una representación distinta sea inviable |
| 14. `research-thermal`: iteración térmica | Neumann con escalado diagonal converge en las nueve matrices Schur reales, pero las nueve soluciones difieren en bits | Rechazada bajo igualdad estricta. El diagnóstico no reproduce las ecuaciones reducidas ni el modelo de colisiones completo de MuTLib/Córdoba–Arias |
| 15. `research-preconditioner`: alternativa inspirada en Fillo et al. | Ensayo completo: H₂-MS/CH₄-MS, nodos 9/33/65, shifts 0/1e7 y cinco muestras: 60 solves GMRES convergen, pero ninguno conserva los bits. El camino inclusivo es más lento en los doce grupos | No se selecciona bajo igualdad estricta. Es un precondicionador inspirado en el paper, no una reproducción de su formulación semiimplícita sin Soret |

### C y C-seeds: retirada histórica y resultados con Jcopy

**Todos los resultados y timings C/full anteriores a Jcopy están retirados como
evidencia de aceptación: UNTRUSTWORTHY / NOT ACCEPTED.** Esto incluye los ocho
registros aparentemente exactos de `C-ordinary-smoke-budget4g.json`. Full y seeds
compartían `experimental_ordinary_bank`, que calculaba `flux` pero no copiaba
`J[k,f]`: el residual recibía el array creado con `np.empty` sin inicializar.
La coincidencia con memoria reutilizada es una explicación posible de los pases
aparentes, no una demostración retrospectiva. Los reportes previos se conservan;
`C-seeds-smoke-budget4g.json` y `C-seeds-original-fallback.json` permanecen
**COMPLETE / FAILED**. Sus timings tampoco validan el candidato corregido.

La corrección añade la copia por cara del kernel legacy a ambos modos, sin
cambiar su aritmética. `C-full-Jcopy.json` y `C-seeds-Jcopy.json` terminan con
`complete=true`, `exact_gate=true` y ocho registros cada uno: H₂-MS/CH₄-MS,
9 nodos, tile de química 4 y particiones 1/2/4/7. Todos pasan comparación
ordinaria residual/bloques, inmutabilidad, estado stale, modo Jacobiano exacto,
fallback denso y contador de cierres. Es validación local del overlay congelado,
no validación de una llama completa ni de su integración en el árbol actual.

Se midieron siete pares alternados por configuración, con Numba=1 y BLAS=1.
Las particiones se recorren secuencialmente; no son mediciones con 2/4/7 hilos.
El tiempo incluye residual + Jacobiano, construcción/asignación/copias del
banco y las propiedades de cara normales del Jacobiano. No incorpora la
reutilización general E. Compilación, comparación de bytes e I/O no forman
parte de esos tiempos. El cociente es la mediana de los cocientes emparejados:

| Modo / caso | Partición 1 | Partición 2 | Partición 4 | Partición 7 |
|---|---:|---:|---:|---:|
| full / H₂-MS | 0,865× | 0,845× | 0,800× | 0,779× |
| full / CH₄-MS | 1,078× | 1,074× | 1,084× | 1,107× |
| seeds / H₂-MS | 0,829× | 0,792× | 0,788× | 0,725× |
| seeds / CH₄-MS | 0,968× | 0,996× | 0,988× | 1,031× |

Por ejemplo, full CH₄-MS/partición 1 tiene medianas 9,070 → 8,610 ms;
seeds CH₄-MS/partición 7, 10,323 → 10,043 ms. El gate local de ganancia
exige al menos 1,03× y cinco victorias de siete: full CH₄-MS lo pasa en las
cuatro configuraciones, con 7/7 victorias; seeds sólo lo pasa en CH₄-MS/7,
con 6/7. Ninguno lo pasa en H₂-MS. No se extrapolan esas diferencias a la
llama adaptativa completa.

Full elimina los cierres congelados del Jacobiano: 8/9/11/14 → 0 en H₂-MS
según partición y 10/10/11/14 → 0 en CH₄-MS. Seeds mantiene ocho por caso,
eliminando sólo duplicados de frontera. CH₄-MS/partición 1 aún tiene dos seeds
por sus tiles `[0,4)`, `[4,8)`, `[8,9)`; siete particiones usan seis seeds.
Las propiedades de cara normales del Jacobiano siguen evaluándose.

**Seeds todavía reserva y construye el banco de todas las caras**, no sólo las
fronteras: ambos modos informan 55.744 bytes para H₂-MS y 1.462.016 bytes
para CH₄-MS en esta malla de nueve nodos. Son bytes de arrays del banco,
no RSS ni una cota del pico total. La variante de banco limitado a fronteras
sigue pendiente y necesitará su propia validación.

### Small-factor: evidencia conservadora

`small-factor-first.json` termina completo y exacto: H₂, CH₄ y CH₄-MS;
9/33/65 nodos; `rdt=0` y `1e7`; 18 configuraciones, siete pares por
configuración, Numba=4 y BLAS=1. Compara el constructor `BorderedFactor`
completo (incluidas asignaciones y Schur de Lambda), y mide solve por separado.
Soluciones, inversas y acoplamientos vivos son exactos. El máximo backward
error observado es 7,605×10⁻²⁰, frente al gate 5×10⁻¹²; no se activó fallback
sparse en ninguna de las tres variantes de esas configuraciones.

Se conserva la conclusión más limitada: serial favorece H₂/9 en los dos
shifts (1,210× y 1,150×), pero no da una mejora uniforme. En nueve nodos el
baseline ya usa la ruta serial: esos cocientes no demuestran por sí solos una
optimización nueva. En CH₄-MS/33 y /65, serial queda aproximadamente en
0,60–0,65×; forzar dos barridos muestra ventajas aisladas y variación, no un
corte universal aceptado. No se elige el mejor dato aislado como política.

Faltan tamaños 5/17/31/129 y el shift intermedio `rdt=1` del barrido preparado,
repeticiones para controlar variación y una decisión basada en **llama completa**.
El perfil serial de fases es inclusivo y no se suma al tiempo del constructor.
No se publican speedups de llama a partir de tiempos de factorización aislados.

### Política de hilos: diagnóstico completado, rendimiento pendiente

`thread-policy-first/thread-policy.json` termina con 36 registros y
`policy_gate=true`, `product_exact_gate=true`. Cruza Numba solicitado
1/4/8, BLAS 1/2 y tres órdenes (`numpy-scipy-numba`, `numba-scipy-numpy`,
`scipy-numba-numpy`), con controles previos y diagnóstico tardío: 18 procesos
por etapa. Los procesos se ejecutan secuencialmente y preservan hashes del
producto BLAS compilado en este ensayo.

Con controles previos se observan los 1/4/8 hilos Numba solicitados y los pools
BLAS controlados. El diagnóstico tardío arranca deliberadamente con máximo
Numba=1 y permanece en un hilo aunque después se pida 4 u 8; el límite BLAS
sí se aplica en runtime. El gate tardío aprobado confirma esta limitación,
no que se haya ampliado el pool. No demuestra igualdad de todos los solves
ni una aceleración multihilo en llamas.

### Memoria y validaciones todavía en cola

La ejecución anterior de memoria terminó tras unos 96 s con
`AssertionError: Factor cache gate failed`, antes del checkpoint. No permite
atribuir retrospectivamente el fallo al solve ni aprobar el presupuesto.
La política del baseline congelado sí explica un desajuste del gate:
`_LinearModel.factor` expulsa las claves transitorias sólo en un cache miss;
tras `0 → 1 → 2 → 0`, el hit de cero conserva `[0,2]`. El gate anterior
exige claves en `(0,rdt)` y rechaza ese estado en `rdt=0`.

El driver local corregido conserva el gate original y guarda **antes del
assert** rdt, claves previas/actuales, hit/miss, gate, solve exacto, memoria
retenida y, si difiere el solve, SHA256 y máximo error absoluto. Registra por
separado la política real del baseline y una corrección propuesta basada en
transiciones, identidades y un máximo de dos factores; no la presenta como
aceptación obtenida. `memory-budget-diagnostic.json` todavía no existe al
cierre de esta revisión: el rerun está pendiente. No hay un gate global de
memoria validado con este driver.

El precondicionador de investigación está listo, pero no corrido: mantiene
el A/RHS nativo multicomponente con Soret y usa un clon fresco mixture-averaged
sin Soret sólo para P. Está inspirado en el paper de difusión de masa que
omite Soret/Dufour; no reproduce su formulación completa. Convergencia del
sistema no implica mismos bytes. Fallback sparse y diferencias de bits se
reportarán sin convertirlos en una aprobación de GMRES.

Las cifras históricas 294+dos fallos y 145 pruebas afectadas se solapan; las
554 pruebas del checkpoint siguiente pertenecen al día 8. **No se afirma un
gate de 853 pruebas ni una suite final nueva aprobada.** Las validaciones
GPU, los tiempos de llama con multihilo y la suite final siguen pendientes;
no se asignan tiempos o resultados a esas tareas aún no corridas.

Todos los reportes nuevos citados proceden del baseline preservado de
60 archivos y manifiesto `c96155c75d388c9da79789d9f69912939877a4deb52c4f10911cdf02e46a273c`.
No validan automáticamente el árbol posterior con ampliación de mecanismos y
PLOG/SRI. Las mediciones de E/B/D y del conjunto anteriores a esa ampliación
conservan sus hashes y su alcance original.

La evidencia local nueva está en
`.local/hpc-pending-2026-10-09/experiments/reports/`: `C-full-Jcopy.json`,
`C-seeds-Jcopy.json`, `small-factor-first.json` y
`thread-policy-first/thread-policy.json`. La retirada pre-Jcopy se conserva en
`experiments/diagnostics/ordinary-jcopy-scope-2026-10-09/status.json` y el estado
estático del precondicionador en
`experiments/diagnostics/research-preconditioner-2026-10-09/static-readiness.json`.
Son artefactos locales ignorados por Git; esta revisión publica su alcance sin
reescribirlos. Sólo se actualiza este documento; no se lanzan benchmarks.

## Inventario original: checkpoint del 8 de octubre

Las tablas siguientes conservan el estado **anterior** a la campaña del día
9 y los nombres exactos de las variantes escritas. Para el estado actual de
E, B, compactación y simetría debe consultarse la tabla anterior.

Se contrastaron los informes y scripts guardados de los tres agentes de
transporte con el código actual y los informes de las campañas anteriores.
Las 554 pruebas y las comparaciones exactas citadas en este checkpoint
corresponden a la campaña final del día 8, no a la candidata actual ni a una
nueva ejecución durante esta auditoría.

## Agente de orquestación y literatura

Su informe local es `.local/hpc-multi-2026-10-08/literature.md`. Las letras
A–E son las del documento original. Priorizó D, después E, después A/B.

| Propuesta escrita | Estado comprobado | Qué se hizo o qué falta |
|---|---|---|
| A. Reutilizar el peso molecular `wm` devuelto por el cierre | **Aplicada** | `_face_state_ws` lo usa para viscosidad sin repetir la suma. No sustituye las fracciones molares por las truncadas del sistema de difusión |
| B. Rotar los buffers completos de cara izquierda/derecha | **Pendiente** | Hay reutilización de caras y menos copias, pero todavía se copian datos vivos multicomponente/congelados. Falta intercambiar conjuntamente los slots privados sin aliasing |
| C. Compartir coeficientes base entre residual y Jacobiano | **Pendiente, sin benchmark** | No existe el banco propuesto de matrices/coeficientes por cara. Requiere procedencia exacta del estado y comprobar el coste de memoria O(caras × especies²) |
| C, alternativa. Compartir semillas en fronteras de rangos | **Pendiente, sin benchmark** | Podría evitar las evaluaciones base repetidas entre rangos; no se evaluó si compensa preparar y compartir las semillas |
| D. Preparar solo viscosidad en modo multicomponente | **Aplicada** | `face_viscosity_temperature_data` evita datos de mezcla sin consumidores; el cierre multicomponente y Soret se mantienen |
| D. Omitir copias de arrays muertos de mezcla | **Aplicada** | Se omiten `cond`, `invB`, `diagB` y `scale` en esa ruta |
| D, continuación. Compactar el workspace | **Pendiente** | Los placeholders siguen reservados para conservar las interfaces; quitar escrituras/copias no implica eliminar esas reservas |
| E. Pasar las propiedades base del residual al Jacobiano actualizado | **Pendiente; primera candidata a continuar** | `residual` acepta `cached_properties`, pero `jacobian_blocks` no. Este último vuelve a calcular termoquímica y caras base. Falta reutilización explícita para el mismo estado/configuración, con invalidación correcta |

E reutiliza las salidas existentes `(rho, cp, omega, hk, lam, J, mu)`;
C conservaría además matrices de transporte por cara. Son propuestas distintas
y ninguna está completada. No se puede atribuirles todavía una aceleración.

El agente también comprobó que las columnas de caudal y velocidad **ya no
recalculan transporte**. No hay una optimización pendiente que consista en
eliminar esas llamadas inexistentes. Las perturbaciones de temperatura sí
necesitan sus cierres. El streaming de caras con scratch privado ya existía.

## Agente de temperatura y simetría

Sus entregables locales son `temperature_candidate.py` y
`temperature_notes.md`, en la misma carpeta de campaña.

| Variante escrita | Estado comprobado | Evidencia y motivo |
|---|---|---|
| Evaluar el triángulo de pares y reflejar inmediatamente | **Pendiente de integración, experimento favorable** | Los seis arrays de salida coincidieron byte a byte en H₂/GRI a 11 temperaturas. Kernel local: 1,105× en H₂ y 1,168× en GRI; no es una mejora medida de llama completa |
| Reflejar mediante un segundo recorrido de copias | **No seleccionada** | No mejoró la variante inmediata y empeoró ligeramente H₂ |
| Certificar simetría y conservar fallback original | **Solo prototipo local** | Hace falta integrar copias privadas e inmutables de los cinco arrays reales de pares. Datos asimétricos, mutables, layouts/aliasing no certificados y temperaturas fuera del dominio seguro deben usar la ruta original |

La simetría no se descartó por producir resultados distintos con los datos
nativos ensayados. Quedó pendiente porque la certificación y propiedad de los
datos no se integraron en producción. La comprobación Python por llamada del
prototipo cuesta más que el trabajo ahorrado; el camino rápido medido presupone
datos ya certificados.

## Agente del sistema térmico de Schur

Su script local `schur_candidate.py` conserva `RECORDED_RESULTS`.
Las cinco variantes iniciales pasaron 296 casos locales cada una, sin fallos.
Los cocientes siguientes son medianas emparejadas del kernel Schur de
53 especies: mayores que uno indican aceleración.

| Variante escrita | Estado comprobado | Evidencia y motivo |
|---|---|---|
| `dot_out`: productos en salidas preasignadas | **Aplicada** | 1,050× local; conserva copias transpuestas y orientación BLAS originales |
| `dot_views`: pasar transpuestas como vistas | **No seleccionada; posible ensayo posterior** | Pasó los casos locales y dio 1,076×. Se eligió conservar la orientación/copias originales en el cambio final; no falló por exactitud en ese experimento ni tiene validación completa propia de llamas |
| `row_l11`: construir L11 por filas | **Retirada por rendimiento** | 0,916× local, aunque pasó las comparaciones exactas |
| `lu_rows`: aliases de filas completas en LU | **Retirada por rendimiento** | 0,947× local; no es la misma variante que la LU finalmente integrada |
| `combined`: combinar vistas y recorrido L11 por filas | **No seleccionada por rendimiento** | 0,965× local en Schur |
| Ensayo posterior `lu_simd` / `dot_lu` | **Integrado con condición de tamaño/layout** | El ensayo posterior pasó 296 casos por variante. Producción usa vistas del tramo restante de fila para matrices C de al menos 24 especies, junto con `dot_out`; mantiene el bucle original en los otros casos |

El texto `status` dentro de `RECORDED_RESULTS` quedó desactualizado: describe
el ensayo LU posterior como interrumpido. `schur-lu-trial.json`, el código
actual y la validación final muestran que después sí se completó. No debe
usarse esa frase antigua como estado final.

El coordinador añadió también el experimento de flujo con 2/4/8 acumuladores,
con y sin empaquetado transpuesto. Se eligieron **ocho acumuladores sin
empaquetado**, para más de 20 especies, conservando el orden de cada suma.
Las otras variantes quedaron sin seleccionar. El tiempo del ensayo empaquetado
excluía la preparación de la matriz; no demuestra una ventaja completa de
adoptar ese layout.

## Propuestas de las campañas HPC anteriores

| Estrategia | Estado comprobado |
|---|---|
| Arrays y ventanas con columnas/especies contiguas según el recorrido; preparación compartida de composición | **Aplicada**, con rutas para C/F/vistas y salidas públicas conservadas |
| Scratch privado por rango y productos/sustituciones en buffers | **Aplicada**; existe también la API `twisted_solve_into` para buffers del llamador |
| Reutilizar la cara derecha base como izquierda del siguiente nodo | **Aplicada**; todavía usa copias, no rotación completa de buffers |
| Derivadas químicas por bloques y matrices de nodo contiguas | **Aplicada**; predeterminado de 32 nodos para mecanismos grandes, con presupuesto de derivadas por bloque |
| Scratch adicional en los barridos de factorización | **Retirado**: el caso grande resultó aproximadamente un 4,6 % más lento |
| Retener automáticamente scratch entre sustituciones | **No seleccionado**: medianas aproximadamente un 1–4 % peores; se mantiene propiedad independiente por llamada |
| Primer helper químico serial con `np.dot` | **Rechazado por diferencias en los últimos bits**; sustituido por acumulaciones ordenadas validadas contra el kernel original |
| Ajustar tamaño de bloque y número de hilos por mecanismo | **Explorado, sin política automática**: se probaron bloques e hilos; falta un ajuste aceptado por tiempo completo sobre el código final |
| Optimizar factorización en mallas pequeñas | **Pendiente**: falta medir coste agrupado por tamaño; el porcentaje de llamadas no equivale al porcentaje de tiempo |
| Paralelizar flamelets/familias independientes con procesos | **Pendiente**: no hay campaña de aceleración validada; deben respetarse las dependencias de continuación |
| Presupuesto global de memoria y límite de caché de factores | **Pendiente como política global**: las derivadas por bloques atienden solo una parte de la propuesta original |
| Política de hilos efectivos independiente del orden de importación | **Pendiente de solución general**: configurar los benchmarks antes de iniciar Python no resuelve por sí solo ese caso límite |

## Estrategias de papers que permanecen en investigación

Compresión de rango bajo/PPACA, algoritmo térmico iterativo tipo MuTLib y
precondicionadores alternativos se documentaron, pero **no se activaron**.
Requieren validar errores, conservación y convergencia; no tienen garantía
de mantener los mismos bytes. Las técnicas de localidad/SIMD sí inspiraron
los cambios concretos anteriores. La inversión grande y productos ya usan
LAPACK/BLAS: no se añadió una inversión propia por bloques ni otro backend.

Las fuentes y sus límites están en [el informe multicomponente](hpc-multicomponent.md).
Los entregables exploratorios de `.local/` son locales e ignorados por Git;
esta tabla deja su estado accesible junto a los informes públicos.

## Orden para continuar

1. Ensayar E: compartir propiedades exactas residual → Jacobiano en cada actualización.
2. Ensayar B: rotar slots privados completos y medir si retirar copias compensa.
3. Integrar la certificación de pares y probar simetría en llamas completas.
4. Medir bloques/hilos y fronteras de rangos sobre la versión resultante.

Cada paso necesita comparación con el código anterior y medición alternada
del tiempo completo. Este orden es una propuesta de continuación; ninguna
de las mejoras pendientes se presenta como ya aplicada.
