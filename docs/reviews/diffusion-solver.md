# Revisión del algoritmo no lineal y lineal de difusión

7 de octubre de 2026 · checkpoint `70708b4` · revisión independiente.

Las trazas y hallazgos se refieren a ese checkpoint. La corrección de transporte en curso pertenece a otra revisión; no invalida la evidencia original ni se evalúa aquí. El núcleo difusión/Newton/Jacobiano auditado permaneció sin cambios durante esta revisión.

El arranque frío y la continuación concentran las construcciones del Jacobiano nativo. Hay dos fallos de control reproducibles y tres intervenciones concretas que conviene evaluar en orden. El complemento de Schur es algebraicamente viable; su ventaja temporal no está medida. Se conserva el criterio final completo `Rmax < 1e-7`. No se modificaron código, referencias, benchmarks ni manuscritos; la integración y las mediciones pertenecen al agente principal.

## Evidencia y comparación correcta con Cantera

Se leyeron `docs/diffusion-flames.md`, `docs/assets/diffusion/timing-review.json`, los benchmarks baseline/premixed_reuse y las nuevas trazas `runs/diffusion_review_newton_smoke/benchmark.json` y `runs/diffusion_review_ch4_trace/benchmark.json`. Para reproducir los totales: agrupar `cases[caso].jacobian_matrices` por `size`, sumar `newton_calls[].jacobians` por `mode` y sumar `cantera_solver_statistics.jacobian_count_stats`. La dimensión inicial CH4 es `31*(53+3)+1=1737`; H2, `31*(10+3)+1=404`.

| Caso nativo | Jacobianos totales / en 31 nodos | Estacionario: llamadas / Jacobianos | T fijada: llamadas / Jacobianos | Temporal: llamadas / Jacobianos |
|---|---:|---:|---:|---:|
| CH4 anterior | 287 / 272 | Sin traza de fase | Sin traza de fase | Sin traza de fase |
| CH4 con reutilización y traza nueva | 288 / 273 | 25 / 128 | 1 / 11 | 70 / 149 |
| H2 con reutilización y traza nueva | 143 / 130 | 16 / 67 | 1 / 10 | 30 / 66 |
| H2 multicomponente/Soret, perfil guardado | 146 / 132 | Sin traza de fase | Sin traza de fase | Sin traza de fase |

CH4: **96 llamadas Newton totales, 87 en 31 nodos**; los 15 Jacobianos restantes corresponden al refinamiento. Las dos primeras llamadas estacionarias consumen 10 y 7 Jacobianos y terminan por límite de iteraciones; T fijada consume 11 y falla por `line_search`; 61 de 70 pasos temporales convergen y 9 fallan por `bounds`. H2: 47 llamadas totales, 39 en 31 nodos; primeras estacionarias 8 y 7 Jacobianos, ambas `iteration_limit`; T fijada 10, también `iteration_limit`; 29 de 30 pasos temporales convergen. Esta evidencia explica la tubería real, pero no identifica la ecuación o especie responsable ni permite atribuir los límites de iteración al bug del hallazgo 1.

Cantera CH4 registra **285**, suma exacta de `[69,36,100,53,10,7,2,1,1,1,1,1,1,1,1]`. Sus tres primeras etapas usan 6 nodos de llama, repetidos. Las últimas ocho etapas concentran aproximadamente el 72.5% del tiempo de Jacobiano registrado y solo ocho construcciones. No concluir que Cantera gana en CH4 por tener muchos menos Jacobianos: los totales son casi iguales y la distribución espacial/coste son diferentes. Cantera H2 sí registra 63 frente a 143 nativos; inicia en 6 nodos, hace 10 pasos temporales en 11 nodos y solo un Jacobiano por etapa fina desde la quinta etapa registrada. `grid_size_stats` incluye dos puntos de dominios frontera. Ambas trazas pasan la comparación física; sus tiempos concurrentes **no son un baseline de velocidad**.

Las muestras anteriores sin instrumentación son CH4 33.37 s frente a Cantera 27.63; H2 1.27 frente a .57; H2 multicomponente/Soret 4.00 frente a 2.22. Son muestras individuales. El perfil posterior CH4 registra 29.214 s inclusivos de Jacobiano, 6.166 de LU y 1.786 de sustituciones sobre 44.975 s. La reutilización química redujo coste por construcción, no su número. Categorías anidadas no se suman. Aun eliminar hipotéticamente toda la factorización, sin cambiar la trayectoria, daría una relación máxima de unos 1.16 en ese perfil; no es una aceleración realizable ni una predicción para el tiempo sin instrumentación.

Cantera 3.2 acepta por corrección RMS ponderada, tolerancias estacionarias relativas `1e-4` y absolutas `1e-9`; el nativo acepta por máximo residual escalado `<1e-7`. No son criterios ni unidades equivalentes. La norma RMS exacta está en [OneDim.cpp v3.2.0](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/OneDim.cpp#L94-L119); la política híbrida se describe en [la documentación 3.2](https://cantera.org/3.2/reference/onedim/nonlinear-solver.html). No se atribuye a 3.2 un Jacobiano analítico de 4.0-dev: `OneDim.cpp:268–309` construye perturbaciones numéricas locales.

## 1. Corregir rechazo de convergencia final y recuperación de ensayos

**Prioridad P1 · bugs de control/robustez confirmados mediante pruebas pequeñas.** No se demostró que causen resultados aceptados incorrectos ni cuánto tiempo consumen en las llamas guardadas.

**Última iteración:** `src/kflame/flame/diffusion.py:247–255,304–314` comprueba convergencia al inicio del bucle, pero devuelve siempre `False,'iteration_limit'` tras agotar sus iteraciones, incluso si el último ensayo aceptado es raíz exacta. En temporal, `diffusion.py:344–351` descarta el candidato clasificado como fallido.

Reproducción ejecutada en memoria: `n=1,nv=4,Tref=300`, `target=[0,0,1,.5,0]`, `x0=[1,1,1,.5,1]`, residual `x-target`, Jacobiano identidad CSC, plazo futuro y `max_iterations=1`. Resultado: `ok=False,reason='iteration_limit',maximum_residual=0.0`, estado exactamente igual a target. Corregir verificando el residual completo del ensayo aceptado antes del retorno por límite; reutilizar el `candidate` ya calculado. Validar éxito en la última iteración estacionaria y temporal y rechazo cuando el residual permanezca por encima de `1e-7`.

**Excepción del ensayo:** `diffusion.py:305` evalúa `fun(trial)` sin recuperación. NaN/Inf se amortigua, pero una excepción numérica sale de Newton. Reproducción ejecutada: mismo problema lineal, con residual que lanza `ValueError('synthetic inadmissible trial')` cuando `x[0]<.75`. El ensayo completo falla y `alpha=.25` es admisible y contrae el mérito; la llamada propagó `ValueError` sin probarlo. Esto demuestra la ruta de control, **no que una propiedad nativa haya fallado así en los benchmarks**.

Capturar solo excepciones numéricas recuperables identificadas; amortiguar y conservar contexto de fase/nodo/variable/dt. Fallo del estado base, fallo de ensayo y LU singular requieren diagnósticos diferentes. Actualmente cualquier `RuntimeError` durante construcción/factorización se etiqueta como `singular_jacobian` (`267–273`); la sustitución tampoco tiene recuperación contextual. No ocultar errores de programación con una captura indiscriminada. [MultiNewton.cpp 3.2](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/MultiNewton.cpp#L128-L167) reduce el ensayo tras errores de evaluación. Validar recuperación de ensayo admisible, diagnóstico de base inválida y diagnóstico de LU singular por separado.

## 2. Conservar modelo lineal y eliminar evaluaciones exactamente repetidas

**Prioridad P1/P2 · limitación de reutilización y optimizaciones exactas; ahorro sin medir.**

`factor=None;age=0` reinicia cada `_newton` (`diffusion.py:246`); el modelo se reconstruye tras cuatro usos (`256`). Cada paso temporal, prueba estacionaria y etapa térmica empieza sin modelo. CH4 hace 70 llamadas temporales con 149 Jacobianos; H2, 30 con 66. La política garantiza trabajo de construcción al inicio de casi cada llamada, aunque el estado anterior sea cercano. Las premezcladas ya conservan `JacobianState`, edades y residual del ensayo (`src/kflame/flame/solver.py:666–677,899–917,943–1001`); adaptar estas piezas a difusión, no trasladar todo su solver con otras dimensiones/residuales.

Primero introducir carry del residual aceptado como en `solver.py:952–1001`. Guardar la corrección **original** de `factor.solve(-base)` antes de bloquear trazas para calcular `oldnorm`, evitando otro solve (`diffusion.py:275,283–285,302`). Compartir propiedades base del residual y la construcción cuando el estado sea idéntico: actualmente `fun(x)` y después `problem.properties(x)` repiten su evaluación (`250,262`). Estas medidas mantienen matemáticamente el residual y el Jacobiano; aún requieren verificación numérica y temporal.

Después ensayar persistencia entre llamadas. Reutilizar solo con misma malla, escalas, mecanismo/transporte y modo térmico. Invalidar tras interpolación, normalización de Y, cambio de T fijada, cambio del patrón de donantes de m, grandes cambios T/Y, defecto lineal alto o fallo con modelo viejo. Un cambio de dt modifica la matriz temporal y normalmente exige refactorizar; conservar LU sin actualizar es una aproximación distinta. No anunciar mejora por cambiar edad 4→20 sin ensayo.

El mérito actual es `||D^-1 J_old^-1 F||2`, con pesos de media espacial fijos durante cada búsqueda (`300–306`). Que no divida por `sqrt(n*nv+1)` no explica por sí solo iteraciones extra: ese factor se cancela en la contracción y no se acepta por norma `<1`. El bloqueo de trazas cambia la dirección Newton, pero se comprueba **todo** el residual; no hay aceptación por residual proyectado. Mantenerlo inicialmente y registrar especie limitante/bloqueada: los 9 fallos `bounds` de CH4 temporal merecen esa evidencia antes de rediseñar positividad.

Validación: igualdad de residuales/Jacobianos en snapshots, trazas positivas que deben disminuir, especies en cota, ambas direcciones de flujo, todos los transportes y invalidaciones. Añadir dentro de la instrumentación existente causa de refresco, edad, alpha, ecuación del máximo residual y especie limitante; no hace falta una reescritura del solver.

## 3. Separar criterio temporal y actualizar correctamente su Jacobiano

**Prioridad P1 · política de convergencia; propuestas sin medir. La actualización temporal exacta requiere más que una diagonal constante.**

`diffusion.py:339–355` inicia dt en `1e-7`, multiplica por 1.5 cada éxito hasta .05 s, reduce por .25 ante fracaso y prueba estacionario cada cinco **intentos**, no éxitos. Exige el mismo máximo residual `<1e-7` para cada paso implícito y para la solución final. Las trazas muestran 15 llamadas estacionarias CH4 que agotan iteraciones antes de lograr la primera solución y siete H2; es razonable estudiar cuándo vale la pena repetir ese intento.

Si una fila de especie está dominada por el temporal, el umbral equivale aproximadamente a `errorY < 1e-7*dt*species_scale/rho`; para H2, dt=`1e-7` y escala=7.5, es `7.5e-14/rho`. Es una estimación dimensional local, no error medido. Puede sobrerresolver pasos iniciales. Probar criterio de corrección temporal con guardia de residual apropiada, crecimiento según dificultad/refrescos y prueba estacionaria según descenso de su residual. **Conservar el criterio estacionario final completo.** Cantera aumenta dt cuando el paso no requirió nuevos Jacobianos, una política distinta documentada en [SteadyStateSystem.cpp 3.2](https://github.com/Cantera/cantera/blob/v3.2.0/src/numerics/SteadyStateSystem.cpp#L113-L185). [Kelley–Keyes, 1998](https://doi.org/10.1137/S0036142996304796) fundamenta continuación pseudotemporal bajo hipótesis; no certifica aquí un calendario ni speedup.

En variables normalizadas, el residual actual puede escribirse:

```text
R(x;xold,dt) = F(x) + B(x)*(x-xold)/dt
Jtrans = Jsteady + [B(x)+G(x,xold)]/dt
G[:,k] = (dB/dx[k]) @ (x-xold)
```

B es diagonal en filas diferenciales, depende de rho y, en energía, cp; incluye las escalas de variables/filas (`src/kflame/flame/diffusion_kernels.py:46–47,58–61`). G contiene acoplamientos T/Y locales; continuidad, fronteras y Lambda son algebraicas. El signo es positivo con la convención nativa. Las diferencias finitas actuales incluyen estos términos correctamente.

Las premezcladas actualizan una máscara constante (`solver.py:704–729`), como [MultiJac.cpp 3.2](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/MultiJac.cpp#L44-L49). Copiarla omitiría G y los coeficientes de B: sería Jacobiano aproximado, no reutilización exacta. Ruta pequeña: almacenar `Jsteady`, B y las derivadas locales necesarias en el estado de anclaje; actualizar bloques temporales al cambiar dt/xold y refactorizar. Entre anclajes sigue siendo chord hasta refrescar. Comprobar diferencias finitas con `x!=xold`: en `x=xold`, G se anula y oculta una implementación incompleta. Una escalera dt o SER de las premezcladas es opción posterior, no requisito del primer cambio.

Validar por separado tolerancia temporal, calendario y reutilización. Registrar fallos, pasos exitosos, residual estacionario durante continuación y frecuencia de refresco. No mezclar los tres cambios en una única medición.

## 4. Reducir el coste del arranque según dificultad y mecanismo

**Prioridad P2 · limitación de estimación inicial; bootstrap propuesto sin medir.**

`diffusion.py:98–119` usa una erfc con media de difusividades de todas las especies; esa media depende del mecanismo, incluidas trazas. m procede de velocidad lineal y rho térmica, mientras V es triangular: no se obtienen imponiendo conjuntamente continuidad discreta. Es una estimación válida, pero puede empezar lejos de la solución de flujo. El fallback HP parcelario (`126–146`) cuesta 4.786 s en el perfil CH4 frente a .002 s H2. El arranque uniforme nativo de 31 nodos difiere del de 6 nodos de Cantera.

Ensayar una única intervención cada vez: resolver flujo a T/Y fijados antes de gastar etapas fallidas; o bootstrap espacial 6–11–21–31/concentración de nodos en la zona de mezcla. Medir defectos iniciales por ecuación y verificar que alcanza la misma rama encendida. Decidir según tamaños, escalas y fallos observados; no codificar estrategias por nombre CH4/H2. Una semilla mejor para un mecanismo puede perjudicar otro. Mantener malla explícita del usuario y fallback al arranque vigente.

El refinamiento fino ya necesita 1–2 Jacobianos por etapa. Relajar slope/curve no aborda el arranque observado y cambiaría la precisión espacial. La adaptación interpola y reconstruye colores (`365–378`) y correctamente debe invalidar el modelo. La reducción automática a 201 puntos para continuación desde llama aceptada ya existe (`src/kflame/diffusion.py:70–87`); no atribuirle beneficio nuevo en arranque frío.

Validación: frío y continuación; H2 y GRI30; mezcla y multicomponente/Soret; malla explícita/no uniforme; caudal/presión y casos próximos a extinción. Mantener rama, entradas, aceptación final y resolución espacial. No inferir rapidez del número inicial de nodos solamente.

## 5. Probar una resolución por bloques con borde global antes de Krylov

**Prioridad P2 · formulación algebraica comprobada en snapshot pequeño; tiempo y robustez general sin medir.**

El orden nodal contiene b=`ns+3` variables y un Lambda global (`diffusion.py:121–124`). El soporte es vecino; la última fila es continuidad del intervalo final (`diffusion.py:85–96`; `diffusion_kernels.py:62`). Así:

```text
J = [ A c; rT d ], rhs = [f;g] = -F
u=solve(A,f); v=solve(A,c); s=d-rT*v
deltaLambda=(g-rT*u)/s; deltaq=u-v*deltaLambda
```

A es tridiagonal por bloques; c aparece en momento interior; r en los dos últimos nodos. Aquí d=0, pero s puede ser distinto de cero. No tratar Lambda como dato fijo. Prueba ejecutada en memoria: H2/aire mezcla sin Soret, estimación inicial de 7 nodos, 92 incógnitas. A tiene cero entradas fuera de bloques vecinos. Schur=`-2.4507129359458983`; corrección frente a LU completo difiere `1.6135e-12` relativo; error hacia atrás `||J*delta+F||inf/(||J||inf*||delta||inf+||F||inf)=7.1035e-19`. Se construyó un solo Jacobiano y no se resolvió una llama completa.

Empezar con A dispersa + Schur como referencia; después adaptar los backends ya existentes de banda/bloques (`src/kflame/flame/equations.py:1922–1989,2715–2816`). Cachear v/s mientras la matriz sea idéntica. El semiancho seguro de A es `2*b-1`; en CH4 final implica unos 189 MiB de almacenamiento LAPACK para factorización, sin copias. J completo no cabe en esa banda porque Lambda es global; ensancharla destruye la ventaja. Thomas por bloques requiere O(n*b²) memoria y O(n*b³) factorización, igual orden de trabajo que banda con ancho O(b), pero constantes distintas.

El Thomas existente pivota dentro de bloques, no entre ellos (`equations.py:2738–2760`). A puede ser singular o sus pivotes problemáticos aunque J completo sea invertible. Evaluar pivotes, Schur y error hacia atrás; volver a LU completo ante fallo. No regularizar silenciosamente un Schur pequeño cerca de extinción. Validar snapshots temporales/estacionarios, finos/gruesos, no uniformes y todos los transportes; objetivo de error hacia atrás `<=1e-10` en sistemas bien condicionados, con comparación de correcciones interpretada según condicionamiento.

H2 pequeño favorece primero reducir reconstrucciones/overhead; GRI30 tiene 169 lotes perturbados por Jacobiano frente a 40 H2 (`diffusion.py:206–240`), cada lote recorriendo la malla. En H2 multicomponente/Soret, LU suma .259 s sobre 5.650 s perfilados frente a 3.517 s inclusivos de transporte: un backend lineal distinto aborda una fracción limitada. Cantera 3.2 congela coeficientes durante el Jacobiano local, pero recalcula flujos ([Flow1D.cpp](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/Flow1D.cpp#L368-L388)); el ensayo global congelado nativo empeoró CH4, aproximadamente 80 frente a 76 s perfilados. No recomendar repetirlo como primer cambio.

Newton–Krylov queda como opción posterior para mecanismos/mallas donde J/memoria siga dominando: precondicionador local con borde, residual lineal verdadero y tolerancia de Newton inexacto ([Eisenstat–Walker, 1996](https://epubs.siam.org/doi/abs/10.1137/0917003?journalCode=sjoce3)), respaldo directo y medición de productos residuales/iteraciones. No se ensayó y no justifica una reescritura ahora.

## Orden de ejecución y aceptación

1. Corregir los dos bugs del hallazgo 1 en cambios independientes y añadir sus reproducciones como pruebas funcionales.
2. Trasladar carry y reutilización exacta del hallazgo 2; verificar equivalencia antes de modificar política.
3. Ensayar persistencia/temporal del hallazgo 3 con traza existente enriquecida; separar construcción, refactorización y sustitución, y cambiar una política cada vez.
4. Evaluar arranque del hallazgo 4 con fallback, por mecanismo/dificultad.
5. Prototipar backend con borde del hallazgo 5 en snapshots y solo promoverlo si mejora tiempo completo y robustez.

Todos mantienen residual estacionario completo `<1e-7`, suma Y y mínimo Y con límites existentes, llama encendida y criterio de malla. Mantener límites de comparación guardados: T 20 K, Y .005, velocidad 3%, Lambda 2%, pico de calor 5%; comprobar que no empeora materialmente el defecto integral de entalpía. La suite de 158 pruebas y tres comparaciones corresponde al checkpoint suministrado; esta revisión no la volvió a ejecutar. Medir secuencialmente en máquina sin otros benchmarks, kernels compilados, repeticiones independientes y dispersión; separar API/solver y perfilado/no perfilado. Los artículos sustentan métodos, no promesas de velocidad.

Se ejecutaron solo dos reproducciones sintéticas y una comprobación algebraica H2 de 7 nodos, sin artefactos temporales ni benchmarks completos. Archivo cambiado por esta revisión: **`docs/reviews/diffusion-solver.md`**.
