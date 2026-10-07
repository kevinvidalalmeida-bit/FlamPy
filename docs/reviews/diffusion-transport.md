# Revisión independiente del transporte nativo

7 de octubre de 2026 · checkpoint `70708b4` (posterior a `94baec9`). Alcance: transporte, viscosidad, multicomponente/Soret y `DiffusionFlameProblem.properties`. Sólo se escribieron este informe y una caché temporal aislada de compilación; no se modificó el núcleo ni se ejecutaron llamas completas. Los benchmarks globales y su metadata pertenecen al agente principal.

**Resultado:** se confirmó un bug de consistencia en difusión alternativa y Soret de mezcla para Y no normalizada. La ruta rápida de difusión ordinaria y la formulación multicomponente pasan las comprobaciones puntuales descritas abajo. Hay oportunidades exactas de reutilización y fusión, cuyo beneficio global todavía no está medido.

## Cinco hallazgos accionables

| Orden | Clasificación | Acción | Evidencia principal |
|---|---|---|---|
| 1 | **P1, bug confirmado** | Unificar el peso molecular del estado no normalizado en difusión alternativa y Soret de mezcla | `transport.py:100`, `:112`, `:738`, `:754`; reproducción abajo |
| 2 | P1, limitación observada; propuesta sin medir | Reutilizar datos térmicos de cara entre perturbaciones de Y, con presupuesto de memoria | `diffusion.py:161`, `:175`, `:231`; Wilke acumula 8.87 s inclusivos en CH4 |
| 3 | P2, redundancia demostrable; propuesta sin medir | Fusionar evaluación de difusión, conductividad, viscosidad y Soret opcional | `diffusion.py:170`, `:174`, `:175`; polinomios y composición repetidos |
| 4 | P2, redundancia demostrable; propuesta sin medir | Ensamblar sólo bloques necesarios para Schur y evitar tensores de difusión redundantes | `multicomponent_kernels.py:99`, `:233`, `diffusion.py:165` |
| 5 | P2, limitación observada; propuesta sin medir | Recalcular sólo caras adyacentes a nodos perturbados; medir planificación por tamaño | `diffusion.py:209`, `:221`, `:231` |

`transport.py`, `backend.py` y los kernels corresponden a `src/kflame/chemistry/`; `diffusion.py` corresponde a `src/kflame/flame/`. Las líneas locales son del checkpoint indicado. Congelar transporte durante perturbaciones T/Y es una **política de convergencia**, distinta de estos cambios exactos: el ensayo previo documentado empeoró el coste global. Esta auditoría no recomienda repetirlo ni revisa la política no lineal.

## 1. Bug: peso molecular incorrecto fuera de sum(Y)=1

El backend conserva Y sin renormalizar (`backend.py:283`, `:291`); las perturbaciones de una especie alteran su suma (`diffusion.py:219`, `:221`). La conversión utiliza:

\[
\overline W=(\sum_k Y_k/W_k)^{-1},\quad X_k=Y_k\overline W/W_k,
\quad \sum_k X_kW_k=\overline W\sum_kY_k.
\]

Por tanto, `sum(X*W)` no sustituye a `Wbar` en estados de Newton no normalizados. La ruta rápida conserva el peso correcto (`transport.py:155`, `:233`); la alternativa lo reconstruye como `sum(X*mw)` (`transport.py:738`, `:746`). Soret fusionado hace lo mismo pese a disponer ya de `wmix` (`transport.py:84`, `:100`, `:112`); su alternativa llama a la difusión afectada (`transport.py:420`).

Cantera 3.2 conserva `Wbar=1/sum(Y/W)` con [Phase::setMassFractions_NoNorm](https://github.com/Cantera/cantera/blob/v3.2.0/src/thermo/Phase.cpp#L413), y [GasTransport::getMixDiffCoeffs](https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/GasTransport.cpp#L167) emplea ese peso en el numerador `(Wbar-Xk*Wk)` y denominador. [MixTransport::getThermalDiffCoeffs](https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/MixTransport.cpp#L51) utiliza esos mismos coeficientes y el peso del estado.

**Reproducción ejecutada:** Python 3.11, Cantera **3.2.0**, `h2o2.yaml`, 1000 K, 101325 Pa, `Y_k=scale*(k+1)/55`, k=0…9. Referencia con `set_unnormalized_mass_fractions` antes de fijar TP. Máximos absolutos de D en m²/s; el relativo Soret se divide por el máximo absoluto del vector de referencia.

| scale=sum(Y) | D rápida–Cantera, absoluto | D alternativa–Cantera, absoluto | rápida–alternativa, relativo máximo | Soret–Cantera, relativo al máximo |
|---:|---:|---:|---:|---:|
| 1 | 3.47e-18 | 3.47e-18 | 0 | 2.13e-14 |
| 1.00001 | 3.69e-18 | 7.82e-10 | 2.22e-6 | 8.91e-7 |
| 1.01 | 3.90e-18 | 7.82e-7 | 2.23e-3 | 8.77e-4 |

Para repetir desde el workspace con `PYTHONPATH=src`:

```python
import numpy as np, cantera as ct
from kflame.chemistry.mechanism import load_mechanism, resolve_mechanism
from kflame.chemistry.transport import NativeTransport
from kflame.chemistry.thermo import NativeThermo
m = load_mechanism(resolve_mechanism('h2o2.yaml'))
tr, th = NativeTransport(m), NativeThermo(m)
g = ct.Solution('h2o2.yaml')
T, P = np.array([1000.]), 101325.
for scale in (1., 1.00001, 1.01):
    Y = (scale*np.arange(1., 11.)/55.)[:, None]
    fast = tr.eval_faces_poly_fast(T, P, Y, m.inv_molecular_weights)[1][:, 0]
    slow = tr.mix_diff_coeffs(T, P, th.Y_to_X(Y))[:, 0]
    g.set_unnormalized_mass_fractions(Y[:, 0]); g.TP = T[0], P
    dt = tr.thermal_diff_coeffs(T, P, Y)[:, 0]
    print(scale, max(abs(fast-g.mix_diff_coeffs)),
          max(abs(slow-g.mix_diff_coeffs)), max(abs(fast-slow)/abs(fast)),
          max(abs(dt-g.thermal_diff_coeffs))/max(abs(g.thermal_diff_coeffs)))
```

**Cambio concreto:** transmitir Wmix del estado Y a la reducción alternativa; revisar el contrato de la API que sólo recibe X, porque X no conserva la escala original de Y. Soret debe usar su `wmix` existente. No solucionar esto renormalizando estados de Newton.

**Alcance:** bug en la extensión a estados no normalizados, relevante para las derivadas y evaluaciones intermedias. No demuestra que los perfiles finales aceptados tengan un error físico significativo. Verificación exigida: estados normalizados/no normalizados, perturbaciones individuales, ambas rutas backend y Soret; acuerdo puntual aproximado `rtol=1e-10` para magnitudes significativas, más columnas afectadas del residual.

## 2. Banco térmico exacto por caras y lote

Las viscosidades puras, Phi de Wilke, conductividades puras, binarios a presión unidad, A*, B*, C*, relajación rotacional y cp/R dependen de T y mecanismo, no de Y. Hoy se recalculan las raíces y pares Wilke (`viscosity_kernels.py:12`, `:21`), los binarios de mezcla (`transport.py:191`) y todos los ajustes multicomponentes (`multicomponent_kernels.py:59`) para las columnas de especies.

En un lote de perturbaciones de Y, T de cada cara es idéntica. Guardar datos térmicos base permite compartirlos sin congelar las reducciones que sí dependen de Y: X, Wmix, rho, sumas de difusión, viscosidad de mezcla y matrices multicomponentes. La columna T necesita datos nuevos en las caras afectadas; las columnas de especies vuelven al banco térmico base. Invalidar por cambios de T, mecanismo o malla y por P en cantidades dependientes de presión. La identidad del lote y los nodos perturbados proporcionan invalidación exacta; no usar redondeo de T.

**Memoria:** un tensor float64 de pares ocupa `8*F*S²` bytes. CH4, 53 especies y 1321 caras: **28.3 MiB por tensor**; dos son 56.6 MiB, además del banco termoquímico actual. A*, B*, C* y binarios solos ocuparían 113 MiB. Compartir un límite global de memoria o procesar caras por bloques. No multiplicar esos tensores por las especies perturbadas.

**Factorizaciones:** L00 y Schur dependen de X. A T fija, una perturbación de Y exige nuevas matrices y resoluciones. Reutilizar sus factores sólo cuando el estado completo de esa cara no cambia. Validar propiedades y columnas frente al evaluador completo, cambios de malla y ausencia de datos obsoletos.

## 3. Fusión de propiedades de cara

Mezcla con Soret evalúa X, log(T), binarios y D en la ruta ordinaria; Soret los vuelve a evaluar (`transport.py:79`, `:94`, `:103`); viscosidad convierte de nuevo Y a X y evalúa ajustes (`diffusion.py:175`). Una API que devuelva rho, Wmix, D, lambda, mu y D térmico opcional puede compartir estos intermediarios y las sumas Wilke utilizadas por Soret. Sin Soret no requiere C*. Su fallback también evalúa binarios dos veces: llamada directa en `transport.py:420` e interna en `transport.py:730`.

Multicomponente ya dispone de viscosidades puras, X y Wmix (`multicomponent_kernels.py:48`). Incorporar la reducción Wilke evita repetir sus ajustes en `diffusion.py:175`. Desactivar Soret **no permite omitir el sistema térmico de conductividad**: `backend.py:488` sólo anula el vector D térmico, y lambda conserva el cierre multicomponente. Ahorrar únicamente su postprocesado tendría un beneficio pequeño sin medir.

Preservar/documentar los suelos de composición: mezcla ordinaria usa `1e-300` (`transport.py:215`, `:361`), Soret/multicomponente `1e-20`, y [Cantera Tiny=1e-20](https://github.com/Cantera/cantera/blob/v3.2.0/include/cantera/base/ct_defs.h). No cambiar estas regularizaciones como supuesto atajo de velocidad. Verificar especies nulas/traza y sus derivadas al fusionar.

Como ajuste posterior, los binarios y A*, B*, C* permiten evaluar un triángulo y reflejarlo; Wilke permite la identidad `Phi_jk=Phi_kj*(mu_j/mu_k)*(W_k/W_j)`, utilizada en [GasTransport::updateViscosity_T 3.2](https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/GasTransport.cpp#L84). Cambia el orden aritmético: comparar derivadas y propiedades antes de adoptarlo. No se midió su ganancia.

## 4. Schur ya reutiliza L00; sobran temporales

La ruta compilada resuelve L00 contra la identidad una sola vez (`multicomponent_kernels.py:81`) y entrega esa inversa a Schur (`:83`, `:247`). **No hay una segunda factorización de L00 que eliminar.** La resolución restante es la de Schur (`:251`).

Sin embargo, `thermal_system` reserva y llena una matriz 3N×3N (`:99`) y un bloque l22 denso (`:193`, `:233`), aunque el modo Schur utiliza directamente l01, l11, l12 y el vector diagonal. Reservar la matriz completa sólo para `dense=True` y evitar l22 denso en Schur conserva el cierre y reduce asignaciones. Mantener el modo denso para verificar residuos de sistema y propiedades, especialmente en estados mal condicionados.

Además, `evaluate_faces` devuelve un tensor multi de F×S×S (`:30`), y `properties` crea otro tensor coeff del mismo tamaño (`diffusion.py:165`). Una ruta del solver puede aplicar pesos y gradientes dentro del kernel por cara y devolver directamente J, lambda y mu; conservar la API matricial para consumidores que la necesiten. No requiere reformular las ecuaciones ni cambiar la inversa inicialmente.

Una sustitución posterior de la inversa por LU y múltiples RHS es exploratoria y de menor prioridad. El producto de difusión contiene `L00^{-1}g-diag(L00^{-1})*sum(g)` para `g=grad(X)`; no descartar el segundo término por asumir una suma exactamente nula en coma flotante. Schur sigue requiriendo `L00^{-1}L01`, con múltiples RHS: no desaparece automáticamente la escala cúbica.

## 5. Localidad y planificación por tamaño

Un nodo perturbado modifica sólo sus dos caras adyacentes. El coloreado cambia aproximadamente un tercio de los nodos por lote (`diffusion.py:209`), pero `properties` recalcula todas las caras (`:161`, `:231`). Conservar las caras no afectadas evita cerca de un tercio de evaluaciones de transporte por lote interior; **no implica reducir un tercio del tiempo global**. Construir la máscara a partir de los nodos conocidos, no de comparaciones aproximadas de estados.

Los kernels usan prange; el paquete establece por defecto cuatro hilos Numba y un hilo OpenBLAS (`src/kflame/__init__.py:5`). El comentario de `multicomponent_kernels.py:24` no impone el número efectivo de hilos BLAS si NumPy se importó antes o el usuario configuró otra cosa. Medir esa configuración, y seleccionar serial/paralelo por F, S y modelo respetando opciones explícitas. El coste de lanzamiento puede pesar más en H2; CH4 tiene muchos más pares. No seleccionar por nombre de combustible.

## Evidencia independiente y siguiente secuencia

`diffusion.py:171` usa `rho*Wk/Wbar` y el flujo promediado tiene signo negativo y corrección de masa (`flame/equations.py:932`, `:940`). Multicomponente usa `rho*Wk*Wi/Wbar²`, signo positivo y `-Dthermal*grad(T)/T` (`diffusion.py:165`, `:167`, `:168`). Coinciden con las [ecuaciones oficiales 3.2](https://cantera.org/3.2/reference/onedim/governing-equations.html#diffusive-fluxes). La ausencia de otra corrección sobre el multicomponente no es por sí misma un bug.

Comprobación pequeña ejecutada: `h2o2.yaml`, `Y_k=(k+1)/55`, 101325 Pa, T=[300,1000,2000] K; comparar `NativeMulticomponentTransport.eval_faces` con las propiedades de `ct.Solution(..., transport_model='multicomponent')` en esos estados. Los máximos relativos, normalizados por el máximo de cada propiedad de referencia, fueron:

| T (K) | D multicomponente | D térmico | Conductividad |
|---:|---:|---:|---:|
| 300 | 1.40e-14 | 2.29e-14 | 1.68e-14 |
| 1000 | 3.44e-15 | 1.48e-14 | 1.85e-15 |
| 2000 | 2.66e-15 | 1.17e-14 | 3.40e-15 |

Al repetir con `linear_solver='dense'`, Schur y denso difirieron como máximo `1.22e-16` en lambda y `4.09e-16` en D térmico; D ordinario fue idéntico. La conductividad incluye ambos bloques energéticos (`multicomponent_kernels.py:261`), como el [código primario MultiTransport 3.2](https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/MultiTransport.cpp). Son verificaciones puntuales, sin prueba de velocidad ni garantía para todo mecanismo.

Se buscaron también los artículos originales de Wilke, DOI [10.1063/1.1747673](https://doi.org/10.1063/1.1747673), y Dixon–Lewis, DOI [10.1098/rspa.1968.0178](https://doi.org/10.1098/rspa.1968.0178); el texto editorial no fue accesible. No se presentan como artículos leídos: la verificación operativa se basa en código y documentación primarios de Cantera **3.2**, cuya [referencia de multicomponente](https://cantera.org/3.2/cxx/df/d7c/classCantera_1_1MultiTransport.html) atribuye el cierre a Dixon–Lewis. No se usó el Jacobiano analítico 4.0-dev.

La evidencia temporal ya guardada da 49 478 llamadas de viscosidad CH4 y 8.87 s, frente a 4.95 s de mezcla (`docs/assets/diffusion/timing-review.json:68`, `:76`). H2 mezcla: 0.138 s de viscosidad y 0.116 s de mezcla (`:170`, `:179`). H2 multicomponente/Soret: 3.52 de 5.65 s instrumentados son transporte, y 0.168 s viscosidad (`:275`, `:286`, `:290`). Categorías inclusivas: no sumarlas ni convertirlas en promesas. Las muestras sin instrumentación son 33.37/27.63 s CH4, 1.27/0.57 s H2 y 4.00/2.22 s H2-MS, nativo/Cantera; no se repitieron aquí.

Mezcla y Wilke escalan aproximadamente F·S² por evaluación; repetir por especies aproxima F·S³. Multicomponente añade resoluciones de orden S y coste cúbico por cara, repetidas cuando cambia Y. Diez especies de H2 no representan el coste de 53 o centenares. El banco térmico elimina trabajo repetido pero conserva las reducciones de composición.

**Pasos ordenados para integración:**

1. Corregir el bug y validar el contrato no normalizado antes de optimizar.
2. Medir fusión y banco térmico acotado, comparando propiedades y columnas T/Y con el evaluador completo; incluir ambos sentidos de flujo, malla/P/T cambiantes y especies traza.
3. Eliminar temporales Schur y tensores redundantes; comprobar sistema denso y residual del sistema de transporte. Incorporar después localidad y planificación por tamaño.
4. Microbenchmarks secuenciales, kernels calientes, tamaños pequeños/medios/finales y varias repeticiones; registrar memoria e hilos efectivos. Exigir precisión aproximada `rtol=1e-10` en propiedades significativas y tolerancias absolutas justificadas cerca de cero; contrastar columnas con los límites existentes `tests/test_diffusion_flame.py:41`.
5. El agente principal ejecuta posteriormente las llamas/referencias y conserva residuo completo `1e-7`, límites físicos y controles de masa/especies. Registrar evaluaciones e iteraciones para detectar mejoras locales que empeoren la convergencia. [Flow1D 3.2](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/Flow1D.cpp#L400) congela transporte en el Jacobiano, pero el ensayo nativo previo no avala esa política.

## Archivos modificados

- `docs/reviews/diffusion-transport.md` — único archivo modificado por esta revisión en el workspace.
- Caché de compilación fuera del workspace: `%TEMP%/difftransport_audit_70708b4_numba`.
