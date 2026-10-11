# Aramco 3.0: perfil del Jacobiano y columnas de transporte por bloques

**Resultado integrado:** Aramco3 / 33 nodos pasa de 3.163 s a 2.733 s por ciclo de propiedades, residual y Jacobiano; cociente emparejado 1.167×. Las salidas conservan sus bytes. No es el tiempo de una llama completa.

**Hallazgo:** antes de agrupar las columnas, composición concentra aproximadamente el 68 % del trabajo de los hilos del Jacobiano y temperatura el 29 %. La química analítica ronda el 2 %. La nueva mejora conserva diferencias finitas y agrupa el trabajo de composición; la derivada analítica térmica queda documentada para validación futura.

Fecha: 9 de octubre de 2026. AramcoMech 3.0 instalado: 581 especies y 3037 reacciones. La referencia es la fuente aceptada inmediatamente después del banco base; se congelaron 70 archivos y se cargaron copias independientes de ambos módulos de llama. Las dependencias químicas compartidas conservan sus hashes. La fuente final añade un helper y modifica sólo `flame/diffusion.py` y `flame/diffusion_kernels.py` dentro de este alcance.

## Dónde se va el trabajo del Jacobiano

Estado lineal de 33 nodos, banco activo, 4 hilos Numba y BLAS a 1 hilo. Copias instrumentadas separan cinco regiones sin cambiar los resultados: seis bloques/columnas comparados por bytes, dtype, shape, strides y finitud. Las derivadas químicas se miden alrededor de sus llamadas reales. Tres muestras después de JIT.

| Región | Fracción aproximada del trabajo medido de los hilos |
|---|---:|
| Columnas de composición: flujos, viscosidad y filas | 67.66 % |
| Columna de temperatura: transporte y filas | 28.89 % |
| Derivadas químicas | 1.69 % |
| Ventana, termoquímica y filas base | 0.66 % |
| Columnas de flujo/velocidad | 0.70 % |
| Caras base y copia del banco | 0.40 % |

**Estos porcentajes suman tiempos de los trabajadores, no tiempo de pared.** Incluyen filas, restauración y escritura de bloques en las regiones de columnas; no atribuyen todo el 68 % exclusivamente a los productos de difusión. La instrumentación adquiere brevemente el GIL al tomar marcas. El JSON conserva los tiempos por nodo, la pared instrumentada y tres paredes sin instrumentación. No son contadores de CPU ni L1/L2. El perfil corresponde a la referencia anterior a la nueva agrupación.

## Un cierre térmico físico, con matrices frescas

Cara 16 de la misma malla: T_f = 1321.875092489 K, perturbación real de temperatura y composición de cara del fixture. Cada muestra reconstruye L00 y Schur antes de invertir/factorizar; nunca se refactoriza una matriz que ya contenga LU. Nueve muestras, un hilo BLAS. D, conductividad y Soret concuerdan con Cantera 3.2 (rtol 3e-7, atol 1e-12), y el cierre instrumentado conserva los bytes del original.

L00/normalización: **0.745 ms**; inversión ordinaria: **13.039 ms**; cierre térmico completo instrumentado: **38.909 ms**; salida D/DT: **0.402 ms**.

| Dentro del cierre térmico | Mediana (ms) |
|---|---:|
| Ensamblaje L01 | 0.852 |
| Ensamblaje L11 | 1.134 |
| Ensamblaje L12 | 0.520 |
| Ensamblaje L22 | 0.743 |
| Escalado y tres productos matriciales | 24.182 |
| LU del Schur recién ensamblado | 10.305 |
| Solve triangular | 0.270 |
| Recuperación de conductividad/Soret | 0.395 |

Las medianas de componentes no reconstruyen exactamente la mediana total; faltan la preparación pequeña y el RHS, conservados en el JSON. Es una cara representativa, no el reparto térmico de toda la llama.

## Newton real: tres iteraciones, nueve nodos

Cinco muestras de la referencia, max_age=4 y max_damping=6. Los wrappers restan llamadas anidadas para evitar doble conteo. Cada llamada evalúa cuatro estados, arma un Jacobiano y una factorización y realiza seis solves; termina en `iteration_limit`, sin certificar convergencia.

| Trabajo | Mediana exclusiva (ms) | Proporción de la suma de medianas |
|---|---:|---:|
| Propiedades/transporte de los estados evaluados | 1128.813 | 48.56 % |
| Jacobiano | 904.977 | 38.93 % |
| Factorización del sistema | 262.585 | 11.30 % |
| Solves lineales | 18.392 | 0.79 % |
| Residual con propiedades disponibles | 0.404 | 0.02 % |
| Resto de Newton | 9.299 | 0.40 % |

Las propiedades incluyen los estados iniciales y de prueba de amortiguación. Por eso su peso aquí es mayor que en un único ensamblado. Este reparto no puede sumarse al perfil de 33 nodos ni extrapolarse a una llama completa.

## Mejora integrada y medición final

El consumidor del banco calcula los flujos y las sumas de Wilke para ocho composiciones perturbadas juntas. Los temporales de entrada tienen forma `[Ns,8]` en C, con columnas vecinas contiguas; cada reducción mantiene el orden original de especies. La cola del bloque se rellena con el último carril válido y sus salidas adicionales se descartan. Se conservan el paso de diferencias finitas, los clamps, Soret, las filas y todas las operaciones de cada columna. No se usa fastmath ni GEMM para reemplazar estas reducciones.

Las dos caras base siguen usando el banco validado. `mleft/mright`, que esa ruta no consume, pasan a payload vacío; `mpert`, Wilke, coeficientes congelados y scratch Schur permanecen privados y completos. Los bancos inválidos, la ruta sin banco y los mecanismos pequeños mantienen sus cálculos anteriores. El cap de 96 MiB no cambia: Aramco3 a 65 nodos no selecciona esta mejora.

Un ciclo completo incluye propiedades/banco, residual con propiedades y Jacobiano, con reservas, validaciones, copias y temporales. Siete pares alternados, JIT y gates fuera del cronómetro:

| Mecanismo, 33 nodos | Antes (ms) | Fuente final (ms) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|
| JetSurF 2.0 | 1041.194 | 1019.524 | 1.034× | 5/7 |
| AramcoMech 2.0 | 1907.640 | 1803.942 | 1.073× | 6/7 |
| AramcoMech 3.0 | 3162.809 | 2733.412 | 1.167× | 7/7 |

El JSON también conserva nueve nodos. Los cocientes son medianas de pares; no necesariamente coinciden con dividir las dos medianas. Son ganancias respecto al solver propio anterior, no speedups contra el solver completo de Cantera.

| Newton, tres iteraciones / nueve nodos | Antes (s) | Fuente final (s) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|
| JetSurF 2.0 | 0.491893 | 0.476480 | 1.028× | 4/7 |
| AramcoMech 2.0 | 1.275985 | 1.201615 | 1.062× | 7/7 |
| AramcoMech 3.0 | 1.995522 | 1.898015 | 1.049× | 7/7 |

Estados finales, outcomes y número de Jacobianos coinciden por bytes entre ambas fuentes. Estos tiempos incluyen propiedades, factor, solve y amortiguación; siguen sin ser llamas convergidas.

En JetSurF, el Newton medido gana sólo 4/7 pares: ese resultado no demuestra una mejora temporal uniforme. Aramco2 y Aramco3 ganan 7/7. La aceptación temporal del ensamblado se comprueba aparte (al menos 1.03× y 5/7 pares en los seis casos).

Para Ns=581 se eliminan 20.630 MiB de scratch por rango y se añaden hasta 5.266 MiB de tablas/temporales locales por nodo activo: ahorro neto planificado **15.364 MiB por rango**, aproximadamente **61.454 MiB** con cuatro rangos simultáneos. Esto es payload de arrays, no RSS, memoria de JIT/factores ni un contador de caché.

## Opciones probadas y vía analítica

- Cambiar la matriz de flujo a F fue más lento en la microprueba (8 carriles: C 79.7 µs, F 111.5 µs); descartado.
- Agrupar con temporales completos `[Ns,Ns]` fue más lento en Aramco3/9 (0.926×, 1/7); descartado.
- Temporales estrechos `[Ns,8]` dieron ganancia en los tres mecanismos del piloto de 33 nodos, con igualdad de bytes. La tabla anterior usa la fuente final integrada.

La química ya usa derivadas analíticas. Para el transporte térmico, diferenciar implícitamente `A(T)u(T)=b(T)` permite reutilizar factores mediante `A u_T = b_T - A_T u`, evitando factorizar de nuevo por cada perturbación. Habría que derivar los coeficientes, NASA, normalización y ramas de modos internos, y también la salida D que depende de la inversa. El agente térmico documentó el sistema eliminado y las acciones O(N²). **No está implementada esta derivada.** No reproduce el cociente de diferencias finitas byte a byte y necesita pruebas de tangentes y convergencia antes de sustituirlo.

Cantera 3.2 documenta un Jacobiano numérico; la documentación de desarrollo 4.0.0a2 describe columnas analíticas de especies interiores y conserva diferencias finitas en temperatura/fronteras. Se consultan como referencia, sin atribuir esa API a la versión 3.2 instalada: [Cantera 3.2](https://www.cantera.org/3.2/reference/onedim/nonlinear-solver.html), [documentación de desarrollo](https://www.cantera.org/dev/reference/onedim/nonlinear-solver.html). La derivada térmica completa y un Jacobiano analítico de flujos son las siguientes rutas algorítmicas a validar.

## Verificación y artefactos

Regresión final: **153 passed in 902.37s (0:15:02)**. Doce pruebas nuevas comparan todas las columnas contra las operaciones originales, incluyendo umbrales 23/24 especies, 581 especies, cola de ocho, ceros, trazas, Soret y entradas readonly; una comprueba scratch compacto y rotación. Se añaden las regresiones existentes de banco, propiedades, temperatura, rangos, tiles, memoria y llamas.

Seis escenarios independientes comparan ciclos completos A–B–A entre fuentes: malla no uniforme, radiación, composición dispersa con OH traza, Soret on/off, temperatura fijada y cruce NASA a 1000 K. El banco se comprueba en 36 caras contra Cantera; se validan entradas intactas, propiedad e invalidación y el fallback de banco antiguo. La auditoría AST confirma que Newton, PTC, factor y solver principal conservan sus definiciones anteriores.

No se ha ejecutado una nueva llama Aramco/JetSurF completa convergida ni una comparación de sus tiempos totales frente a Cantera. No hay nueva campaña GPU ni contadores L1/L2.

Datos completos: [JSON publicado](assets/chemistry/hpc-jacobian-breakdown-2026-10-09.json). Scripts, muestras y logs: `.local/hpc-jacobian-breakdown-2026-10-09/`. Revisiones de los tres agentes: `agent-thermal.md`, `agent-memory.md` y `agent-validation.md` en esa carpeta; son revisiones estáticas, separadas de las campañas ejecutadas por el agente principal.
