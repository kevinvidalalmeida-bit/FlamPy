# Comparación reproducible de tiempos

Fecha: 2026-10-07. Punto de partida: `70708b4` (checkpoint de `94baec9`).

La última medición sin instrumentación antes de esta revisión fue 33.37/27.63 s
para CH4, 1.27/0.57 s para H2 y 4.00/2.22 s para H2 multicomponente con Soret
(nativo/Cantera). Son muestras individuales conservadas en
[timing-review.json](../assets/diffusion/timing-review.json); no demuestran una
ventaja general de un solver ni una mejora estadística entre revisiones.

## Qué se compara

Ambos solvers usan los mismos mecanismos, entradas, presión, ancho y modelo de
transporte, y los mismos umbrales de refinamiento. Las mallas, discretizaciones,
estimaciones iniciales y criterios de convergencia son propios de cada solver.
Cantera empieza con seis nodos de llama en este flujo; el nativo, con 31.
Por tanto, son costes de dos flujos completos para el mismo problema físico,
no una comparación de kernels con matrices y perfiles idénticos.

El nativo exige un máximo absoluto del residual escalado menor que `1e-7`.
Cantera 3.2 emplea una norma ponderada de la corrección de Newton; en la
instalación consultada sus tolerancias estacionarias predeterminadas son
`rtol=1e-4`, `atol=1e-9`. Esos números no son intercambiables: poner el mismo
valor numérico no igualaría las pruebas de convergencia. Véase el
[criterio de Cantera](https://cantera.org/3.2/reference/onedim/nonlinear-solver.html).

`runtime_s` del informe nativo incluye resolución, adaptación y diagnóstico
final, y excluye carga del mecanismo, preparación del problema y exportación.
`native_api_s` incluye además la preparación del problema. El tiempo de Cantera
abarca `solve(auto=True)` y excluye preparación y exportación. La comparación
debe conservar esta distinción y no confundirla con costes de consulta FGM.

## Instrumentación añadida

`examples/benchmark_diffusion_solver.py` admite `--repeats N`, conserva cada
muestra y calcula mediana, mínimo y máximo de los tiempos y cocientes. Rota
el orden de los casos entre repeticiones; dentro de cada pareja siempre ejecuta
el nativo antes de Cantera, sesgo de orden que queda explícito en el informe.
Cada repetición resuelve desde cero; no comparte un perfil convergido.

El JSON conserva versiones de Python, NumPy, SciPy, Numba y Cantera, plataforma,
hilos configurados, hash del código, mallas y validación física por muestra.
`cases` conserva la última muestra para compatibilidad; `samples` contiene
todas y `aggregates` es la vista que debe utilizarse al comparar repeticiones.
Una muestra rechazada impide que el benchmark completo se considere aprobado.

La referencia registra además mallas por etapa, número y tiempo de Jacobianos,
evaluaciones fuera del Jacobiano, pasos temporales, tolerancias por componente
y tipo de solver lineal. Las estadísticas de Cantera son las de su
[API oficial](https://cantera.org/3.2/python/onedim.html#cantera.Sim1D.jacobian_count_stats),
con sus propias definiciones; no se deben sumar sus tiempos CPU a tiempos de
pared ni comparar directamente sus contadores con los lotes del nativo.

En modo perfilado, `newton_calls` registra cada intento nativo: malla, etapa
estacionaria/temporal/temperatura fija, paso temporal, duración, Jacobianos,
resultado y motivo. La prueba H2 registró 67 Jacobianos en intentos
estacionarios, 66 en continuación temporal y 10 con temperatura fija. Así
se puede distinguir el coste de cada matriz del coste de construir demasiadas
matrices antes de alcanzar la primera solución.

La prueba funcional de dos repeticiones de H2 pasó ambas validaciones físicas
y ejercitó la serialización de las estadísticas. Se ejecutó mientras trabajaban
los revisores; sus tiempos sirven para comprobar el instrumento, no para
sustituir la medición de velocidad anterior. Cantera registró 63 Jacobianos,
con uno por etapa en las últimas siete mallas: una pista para revisar
reutilización y continuación, no una demostración causal por sí sola.

```powershell
$env:PYTHONPATH='src'
python examples/benchmark_diffusion_solver.py --output runs/review_repeated --no-profile --repeats 3
python examples/benchmark_diffusion_solver.py --output runs/review_profiled
```

## Criterios para aceptar una optimización

1. Contrastar variantes en la misma máquina sin otros benchmarks activos,
   con calentamiento de compilación y varias muestras sin instrumentación.
   Conservar casos nuevos y condiciones que no participaron en el ajuste.
2. Mantener la puerta de aceptación del residual y comparar perfiles físicos,
   conservación, composición mínima y sumas de especies. Una tolerancia más
   laxa requiere una decisión de precisión explícita, no cuenta por sí sola
   como optimización del algoritmo actual.
3. Para kernels exactos, comparar salidas y derivadas con el evaluador original
   en estados perturbados, incluidos terceros cuerpos, falloff, cambio de
   donante y ambas configuraciones de Soret/transporte.
4. Para discretizaciones o estrategias no lineales nuevas, medir error y coste
   frente a mallas progresivamente refinadas; pasar límites de perfiles no
   equivale a demostrar orden de convergencia o cierre integral de entalpía.
5. Activar rutas por descriptores verificables (especies, reacciones, memoria,
   transporte, rigidez observada y fallos), y mantener una ruta general de
   recuperación. No cambiar de física por el nombre del combustible.

Las mejoras propuestas por los cuatro informes requieren estos controles
antes de presentarlas como aceleraciones reales frente a Cantera.
