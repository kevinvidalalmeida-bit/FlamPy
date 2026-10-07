# Revisión independiente: termoquímica y derivadas de difusión

7 de octubre de 2026; código inspeccionado: `70708b4`. Alcance: `NativeSpeciesBackend`, `grouped_core`, química analítica de premezcladas y perturbaciones termoquímicas de difusión. Se leyeron `docs/diffusion-flames.md` y `docs/assets/diffusion/timing-review.json`. No se editaron núcleo ni metadatos de medición; no se ejecutaron llamas completas. Integración, pruebas generales y repeticiones corresponden al agente principal.

**Conclusión:** la reutilización térmica es correcta para los mecanismos incluidos. Se confirmaron bugs al interpretar otros YAML válidos y una subestimación del pico de memoria. Reutilizar las derivadas químicas analíticas existentes merece un ensayo acotado; todavía no hay una mejora de tiempo demostrada en difusión.

## 1. P1 — Bug de formulación: campos químicos YAML ignorados

**Gravedad alta para mecanismos que los declaren.** El parser utiliza estequiometría como orden cinético, inicializa todas las eficiencias a 1, elimina M sin inferir `three-body`, y reconoce solo Troe entre funciones falloff. Evidencia: `src/kflame/chemistry/mechanism.py:359`, `:361`, `:363`, `:395`, `:401`, `:421`. Aceptar estas entradas silenciosamente cambia las ecuaciones físicas.

**Reproducción estrecha:** Cantera 3.2.0, Python 3.11 indicado por el usuario y `PYTHONPATH=src`. Copiar el YAML local `h2o2.yaml`, conservar especies/NASA7/unidades y reemplazar sus reacciones por una sola. Evaluar T=1100 K, P=101325 Pa, X=`H2:.4,O2:.2,N2:.39,H:.005,O:.005`, b=0, Ea=0. Comparar `NativeKinetics.net_production_rates(T,C,g_RT)` con `gas.net_production_rates`, calculando C mediante densidad nativa y las mismas Y. Los archivos temporales `diffchem_audit_*` se eliminaron al terminar.

| Reacción y parámetros | Fuente nativa/Cantera, igual para cada especie con fuente no nula |
|---|---:|
| `H2 + O2 => 2 OH`, A=1e8, `orders: {H2: 0.5}` | 0.00210511 |
| `H + O + M => OH + M`, `type: three-body`, A=1e8, `default-efficiency: 0`, `efficiencies: {H2: 2}` | 1.75 |
| `H + O + M => OH + M`, A=1e8, sin `type` | 90263.102689 |
| `H + O (+M) => OH (+M)`, `type: falloff`, Ahi=1e8, Alo=1e12, `Troe: {A: .5, T3: 100, T1: 1000, T2: 0}` | 4.63750544 |
| Misma reacción falloff/A, `SRI: {A: .5, B: 100, C: 1000}` | 1.13156799 |

Cantera define [órdenes y eficiencias](https://cantera.org/3.2/yaml/reactions.html#efficiencies) y permite [inferir terceros cuerpos desde la ecuación](https://cantera.org/3.2/reference/kinetics/reaction-rates.html#three-body-reactions). Para T2 exactamente cero, [Falloff.cpp v3.2.0](https://github.com/Cantera/cantera/blob/v3.2.0/src/kinetics/Falloff.cpp#L216) omite el cuarto término: el nativo suma exp(0)=1 (`backend.py:162`, `jacobian.py:75`). **T2 omitido sí funciona** mediante el valor muy grande del parser. [SRI](https://cantera.org/3.2/reference/kinetics/rate-constants.html#the-sri-falloff-function) requiere otra función; ignorarlo convierte la reacción en Lindemann.

**Cambio concreto:** separar órdenes y estequiometría, usar el orden al convertir A, interpretar `default-efficiency`, inferir el tipo desde M, y representar si el cuarto término Troe está activo. Rechazar SRI/Tsang y otros tipos no implementados antes de evaluar. No hace falta implementar todos los tipos para evitar resultados silenciosamente incorrectos.

**Aceptación:** cada caso debe concordar con Cantera en tasas directas/inversas o rechazarse con un diagnóstico de capacidad no soportada. Añadir conservación elemental. Estos campos problemáticos no aparecen en los dos YAML incluidos inspeccionados; no explican por sí solos los tiempos de CH4/H2 ni invalidan las comparaciones guardadas.

## 2. P1 — Bug de formulación: unidades y presión de referencia

**Gravedad alta para entradas afectadas.** `mechanism.py:269`, `:272` usa por defecto cm/mol/cal; Cantera usa m/kmol/J y Ea en J/kmol ([UnitSystem::defaults v3.2.0](https://github.com/Cantera/cantera/blob/v3.2.0/src/base/Units.cpp#L372)). El tiempo no se convierte (`mechanism.py:270`, `:147`). La presión de referencia no se almacena: los tres evaluadores usan 101325 fijo (`backend.py:111`, `jacobian.py:55`, `analytic.py:71`).

**Reproducción:** mismo procedimiento anterior, reacción única `H + O <=> OH`, A=1e8, b=0, Ea=1000, T=1100 K, P=101325 Pa, X=`H:.005,O:.005,N2:.99`.

| Modificación del YAML | Resultado |
|---|---|
| Quitar `units` | Fuente nativa/Cantera = 0.00063295 |
| Conservar unidades originales y añadir `time: min` | Fuente nativa/Cantera = 60 |
| Conservar unidades originales; todas las especies con `reference-pressure: 1 bar` | Kc nativo/Cantera = 0.98692327 |

En la última fila OH inicialmente ausente oculta el problema en la fuente directa; por eso se comparó Kc. El campo está definido en [Species de Cantera 3.2](https://cantera.org/3.2/yaml/species.html#species-thermo-models).

**Cambio concreto y aceptación:** defaults oficiales, conversión temporal, interpretación o rechazo de unidades anidadas y presión de referencia distinta. Comparar Kc además de fuentes, incluyendo composiciones con productos. Validar explícitamente las regiones NASA7: el parser exige `coefs[1]` (`mechanism.py:339`) aunque una región es válida; sustituir `assert` de `:332` por validación que también opere con Python optimizado. La ausencia de NASA9 es una limitación de admisión, no un defecto del NASA7 implementado.

## 3. P1 — Bug de recursos: la cota de 96 MiB omite bancos simultáneos

**Gravedad media; no altera las ecuaciones.** `diffusion.py:212` estima `8*n_color*(ns+2)*(3*ns+3)`: composición, omega, hk, T, rho y cp del lote nuevo. Al evaluar el lado derecho de `bank=...` en `:213`, todavía se retiene el banco anterior, cuyas salidas ocupan aproximadamente `8*n_color*(ns+2)*(2*ns+2)`. También pueden seguir vivas las copias `thermal` de `:225` y existen buffers por hilo (`jacobian.py:33`).

**Evidencia reproducible por cuenta de asignaciones, no medición de RSS:** ns=53/n=1322 da 29.98 MiB para el nuevo lote, aproximadamente 49.96 MiB con las salidas anteriores. ns=70/n=2200 da 85.88 MiB, admitidos por el filtro, pero aproximadamente **143.14 MiB** con las salidas anteriores. No incluye los otros arrays del residual o ensamblaje. La estimación sirve para un lote aislado, no para el pico simultáneo anunciado.

**Cambio concreto:** liberar banco y copias antes del siguiente color; estimar todas las asignaciones simultáneas; dividir nodos en lotes acotados. El fallback actual evalúa el residual ordinario (`diffusion.py:233`), conservando derivadas pero con posible salto de coste al superar el umbral. La división conservaría la reutilización. Si se añade fallback por fallo de asignación, liberar buffers antes de reintentar y registrar la ruta.

**Aceptación:** verificar pico de temporales del evaluador, equivalencia matricial a ambos lados del umbral y tamaños no divisibles por tres. No confundir esta cota con la memoria total del proceso.

## 4. P2 — Propuesta no medida: reutilizar las derivadas químicas existentes

**No se encontró un bug del banco térmico para los mecanismos incluidos.** `jacobian.py:41` calcula NASA y factores de tasas en dos temperaturas; `:88`, `:119`, `:128` recalcula densidad, C, terceros cuerpos y falloff para cada composición. `backend.py:395`, `:402` comprueba tamaño y temperaturas antes de agrupar; no redondea T ni conserva factores entre estados. Los polinomios corresponden a [NASA7 oficial](https://cantera.org/3.2/reference/thermo/species-thermo.html#the-nasa-7-coefficient-polynomial-parameterization).

El banco `[base,T,Y...]` tiene ns+2 columnas (`diffusion.py:192`), frente a ns+3 incógnitas `[m,V,T,Y...]`. Los pasos de `:197`, `:200` coinciden con `:219`, y `column=component-1` en `:226` es correcto. Y son independientes, sin renormalización (`backend.py:283`); cambiar eso cambiaría las columnas del sistema.

**Comprobación puntual:** dos nodos a 900/1600 K y 1 atm, Y positiva normalizada: empezar con `.001/ns` por especie, añadir `H2:.04,O2:.18,N2:.7,H2O:.05,OH:.001,H:.00001,O:.00001` y normalizar. Para ambos mecanismos, rho/cp/omega/hk agrupados y ordinarios resultaron idénticos elemento a elemento, usando los pasos de difusión. En otro estado a 1600 K, X=`H2:.2,O2:.2,N2:.5,H2O:.09,OH:.01`, el error máximo de fuentes frente a Cantera, dividido por su máximo absoluto, fue menor que `7e-16`; el defecto elemental normalizado fue menor que `4e-16`. Son verificaciones estrechas, no una validación global.

`chemistry/analytic.py:37` ya incluye masa de acción, terceros cuerpos, Lindemann/Troe y la cadena a presión fija. `thermochemical_partials` (`:164`) añade rho, cp y T. Con s=sum(Y/W), C=rho*Y/W y Jc=d(wdot)/dC:

```text
dC/dY = rho*diag(1/W) - C*(1/W)^T/s
JY_mass = diag(W)*Jc*dC/dY
JT_mass|P,Y = diag(W)*[d(wdot)/dT|P,C - Jc*C/T]
```

Estas expresiones coinciden con `analytic.py:133`, `:137`. Frente a las derivadas de Cantera, aplicando esa cadena y `skip-falloff=False`, `skip-third-bodies=False`, el error máximo relativo global fue <`5e-13` en JY y <`1.1e-7` en JT para los cuatro estados mecanismo/temperatura anteriores. La [API 3.2](https://cantera.org/3.2/python/kinetics.html#cantera.Kinetics.derivative_settings) utiliza otras variables independientes y puede omitir falloff; sus valores no pueden insertarse directamente como JY del solver.

**Cambio concreto:** adaptar solo la contribución química nodal, sin trasladar todo `flame/analytic_jacobian.py:193`, que tiene geometría/fronteras propias y coeficientes congelados. Incluir la derivada energética `sum(h_mass*omega_mass)`, cp y `dh_molar/dT=cp_molar`; mantener signos (`diffusion_kernels.py:45`, `:51`, `:55`) y multiplicar la columna T por Tref al derivar respecto a x=T/Tref. Una integración que linealiza omega durante las perturbaciones es aproximada; insertar directamente las derivadas evita ese error de truncamiento.

**Aceptación:** selección por capacidad y fallback: los planes analíticos exigen molecularidad entera <=3 y s no saturado (`analytic.py:144`, `:150`). Comprobar columnas/direcciones independientes a varios pasos, ceros y negativos con diferencias unilaterales donde cambie una rama, lados de Tmid y extremos de Pr. Después, suite vigente y tres comparaciones de llama con residuo completo `1e-7`. El 1D de Cantera **3.2** utiliza perturbaciones en [OneDim::evalJacobian](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/OneDim.cpp#L268); no atribuirle características analíticas de 4.0dev. [pyJac (2017)](https://arxiv.org/abs/1605.03262) respalda investigar esta estrategia, pero sus aceleraciones no predicen las de este código.

## 5. P2 — Limitación de rendimiento y elección genérica por mecanismo

Descriptores extraídos de arrays nativos:

| Mecanismo | ns/nr | Tres cuerpos/falloff | Columnas del banco | Soporte Jc directo | Con colliders / ns² |
|---|---:|---:|---:|---:|---:|
| h2o2.yaml | 10/29 | 5/1 | 12 | 70 | 80/100 |
| gri30.yaml | 53/325 | 12/29 | 55 | 1130 | 2453/2809 |

El soporte une especies con fuente estequiométrica con reactantes, productos de reacciones reversibles y especies de eficiencia no nula. Es soporte estructural en C, no nnz medido en la llama. El núcleo agrupado aún recorre todas las reacciones por columna (`jacobian.py:84`, `:111`): `(ns+2)*nr` pasa de 348 a 17875, unas 51 veces por nodo, **sin que eso sea una razón de tiempos**. NASA/Arrhenius solo se evalúa dos veces. El kernel analítico también recorre ns columnas por reacción y guarda JY denso (`analytic.py:42`, `:127`), por lo que no es automáticamente más rápido.

**Propuesta no medida:** acumular derivadas por soportes de reacción y productos exteriores de collider/densidad. La transformación a Y añade un término de rango uno. Con eficiencia uniforme, M=sum(C)=P/(R*T) y su derivada de Y a P,T fijos es cero; con eficiencias distintas aparecen otros acoplamientos. No usar el soporte Jc para colorear directamente JY ni prometer un Jacobiano espacial muy disperso.

Elegir evaluador ordinario/agrupado/analítico por ns, nr, tipos/órdenes admitidos, soportes, reversibilidad, patrones de eficiencia, presupuesto de memoria y coste de una muestra corta ya compilada. Cachear por hash de mecanismo/entorno, sin `fuel=='H2'` o `fuel=='CH4'`. La rigidez depende del estado y no se deduce de ns/nr; si se emplea, estimarla en estados representativos con un operador físico y sus modos de conservación interpretados.

**Margen sustentado por los perfiles existentes:**

| Caso | Química ordinaria + agrupada | Total perfilado | Fracción química |
|---|---:|---:|---:|
| CH4 | 1.052 + 3.864 = 4.916 s | 44.975 s | 10.9% |
| H2 | .0808 + .0625 = .1432 s | 1.747 s | 8.2% |
| H2-MS | .0824 + .0670 = .1494 s | 5.650 s | 2.6% |

Evidencia: `docs/assets/diffusion/timing-review.json:65`, `:72`, `:84`, `:168`, `:175`, `:187`, `:275`, `:282`, `:298`. No sumar a Jacobiano/residual: son categorías inclusivas. Si desapareciera exclusivamente la química agrupada, sin cambiar otros costes, el techo aritmético del total perfilado sería 1.094×/1.037×/1.012×. Son techos hipotéticos, no mediciones ni predicciones. Una reorganización que reduzca además residual/copias exige medir esas categorías.

Las muestras sin perfil siguen siendo CH4 33.37/Cantera 27.63 s, H2 1.27/.57 s, H2-MS 4.00/2.22 s (`timing-review.json:8`, `:111`, `:214`). El mayor margen químico absoluto está en GRI; el peor cociente relativo H2 no prueba que su química sea el cuello de botella.

## Pasos siguientes, en orden

1. Corregir o rechazar explícitamente las entradas de §1–2; comprobar las reproducciones de una reacción, Kc y conservación.
2. Corregir convivencia de buffers y procesar bancos acotados; verificar memoria y equivalencia del fallback.
3. Ensayar únicamente la contribución nodal analítica de §4 en mallas pequeñas, con los casos de derivadas indicados; mantener fallback por capacidades.
4. Medir química por mecanismo y elegir estrategia genérica. El agente principal repetirá después las soluciones completas secuencialmente, conservando límites físicos y residuo completo.

**Distinción de política:** cambiar tolerancias, pasos de perturbación, aceptación o congelar dependencias es una decisión de convergencia, no una corrección de los bugs químicos. Esta revisión no propone esos cambios ni una reescritura del solver.

Archivo creado/editado por esta revisión: `docs/reviews/diffusion-chemistry.md`. Ningún archivo de núcleo, metadatos de benchmark/referencia ni manuscrito fue editado.
