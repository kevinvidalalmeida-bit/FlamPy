# Transporte multicomponente con mecanismos grandes

Revisión del 11 de octubre de 2026. [Guía de lectura](README.md) · [Mecanismos grandes](hpc-large-mechanisms.md) · [Transporte multicomponente (núcleo exacto)](hpc-multicomponent.md) · [Validación](../examples/validate_multicomponent_trace.py)

**Resultado:** `transport='multicomponent'` es práctico con JetSurF 2.0 y AramcoMech, en premezcladas y en contraflujo. El núcleo exacto de Dixon–Lewis cuesta O(n³) por cara (Aramco 3.0: 51–66 ms por cara en un hilo) y el Jacobiano multicomponente denso no cabe en memoria. Ahora, desde 200 especies, cada cara omite las colisiones entre dos especies traza, que son las que suman como mucho 10⁻⁸ en fracción molar. Así cuesta 0,8–10 ms en Aramco 3.0 y 0,2–1,7 ms en JetSurF. Newton se resuelve con GMRES precondicionado por el Jacobiano estructurado de mezcla. Los mecanismos de menos de 200 especies conservan el núcleo exacto sin cambios.

## El modelo: Dixon–Lewis sin colisiones traza–traza

Todo acoplamiento entre dos especies i ≠ j de los sistemas de Dixon–Lewis (L00, L01, L11, L12, L22) procede de colisiones i–j y es proporcional a x_i·x_j. En cada cara, y en cada evaluación, las especies se ordenan por su fracción molar local:

- **traza:** las más pequeñas cuya suma no supera `tail` = 10⁻⁸;
- **mayoritarias:** todas las demás.

El modelo conserva toda colisión en la que participa una especie mayoritaria y las autocolisiones de todas las especies. Omite las colisiones entre dos especies traza distintas, incluida su parte de las sumas diagonales. Lo omitido es O(x_traza) relativo.

El criterio es local y no depende del combustible: con H₂, CH₄, C₂H₄, n-heptano o n-dodecano cambia qué especies son mayoritarias, no la regla.

Dos propiedades exactas lo hacen barato:

1. **D y la solución térmica no dependen de la regularización de rango uno u vᵀ del bloque singular de Stefan–Maxwell L00**, para cualquier u con Σu ≠ 0. D se toma con D_kk = 0.
   - Los flujos de un D con Σg = 0 están fijados por Stefan–Maxwell y la restricción de masa. Con D_kk = 0 la representación es única.
   - En el sistema térmico, la fila de restricción da vᵀa₀ = 0 porque las columnas de L01 suman cero.

   Por eso u se coloca solo en la especie más abundante. La elección de Cantera (u en todas las especies) deja las filas de todas las especies pequeñas paralelas hasta O(x). Su condición es del orden de 1/x_min, y al podar se vuelve numéricamente singular. Con u en una sola especie, cada fila es una fila de Stefan–Maxwell proporcional a su propia x, y las filas se equilibran antes de cada LU.
2. **Sin colisiones traza–traza, todo bloque traza del sistema 3N es diagonal**: un bloque 3×3 por especie traza.
   - El complemento de Schur de L00 sobre las m especies mayoritarias se forma con BLAS.
   - El sistema térmico se resuelve con refinamiento iterativo sobre el bloque mayor. Sus correcciones de traza son O(tail) por entrada, así que bastan dos o tres barridos para llegar al redondeo.
   - Solo se evalúan los pares de colisión del modelo: O(m·n) ajustes polinómicos en lugar de O(n²).

El coste por cara pasa de O(n³) a O(m²·t + t²·m + m³), con m especies mayoritarias y t traza. Con composiciones de llamas reales de Aramco 3.0, m vale de media 93 y como máximo 166.

| Mecanismo | Núcleo exacto, ms/cara (1 hilo) | Modelo podado, ms/cara | Mayoritarias por cara |
|---|---:|---:|---:|
| JetSurF 2.0 (348) | 13–21 | 0,2–1,7 | 3–75 |
| AramcoMech 3.0 (581) | 51–66 | 0,8–10,4 | 3–162 |

## Precisión del modelo

Medida cara a cara, con composiciones de llamas premezcladas CH₄/aire convergidas, frente al núcleo exacto:

| Mecanismo | Flujo de especie, peor relativo | Flujo, ponderado por masa | D_T y λ |
|---|---:|---:|---:|
| GRI-Mech 3.0 (podado forzado) | 1,2·10⁻⁷ | 8·10⁻¹⁰ | 10⁻¹⁴ |
| USC Mech II (podado forzado) | 2,6·10⁻⁷ | 1,9·10⁻⁹ | 10⁻¹⁴ |
| JetSurF 2.0 | — | 2,4·10⁻⁹ | 10⁻¹⁴ |
| AramcoMech 3.0 | 2·10⁻⁶ (especies de flujo casi nulo) | 2,6·10⁻⁹ | 10⁻¹⁴ |

El núcleo compilado resuelve el modelo podado a 10⁻¹⁵ (D) y 10⁻¹¹ (flujos) frente a su sistema denso. Con tail → 0, es decir, sin especies traza, reproduce el núcleo exacto a 10⁻¹¹. D conserva la masa (Σ_k j_k = 0) igual que el núcleo exacto, hasta el redondeo del sistema de cara. Si una cara no converge, se calcula con el núcleo exacto.

## Newton con el modelo podado

El acoplamiento de composición del transporte multicomponente es denso (D_kl acopla todas las especies), así que no tiene la forma `diag + P Qᵀ` del Jacobiano estructurado. Por eso:

- **Contraflujo.** El precondicionador es el Jacobiano estructurado con coeficientes de cara de mezcla alrededor de los flujos, la conductividad y la viscosidad multicomponentes de la base. La dirección de Newton sale de GMRES sobre el residual multicomponente completo (Jv sin matriz).
- **Premezclada.** Desde 400 especies (AramcoMech), la factorización estructurada del Jacobiano de mezcla en x precondiciona GMRES (`flame/krylov.py`), anclado en cada iterado de Newton. El amortiguamiento de Cantera no cambia: los pasos de prueba reutilizan el mismo operador.
  - Por debajo de 400 especies (JetSurF) se conservan los bloques multicomponentes densos con el núcleo podado: 38 s y misma Su que el núcleo exacto. Ahí los bloques sí caben en memoria.
  - `problem.mixture_preconditioner = True/False` fuerza la elección.
- **Arranque de mezcla.** Las llamas multicomponentes o con Soret parten de la llama de mezcla aceptada, como ya hacía la premezclada (`multicomponent_bootstrap`). El pseudo-transitorio frío, la parte más cara, se hace con el transporte barato. La llama pedida se resuelve, se refina y se acepta con su propio residual completo.

Cada paso se acepta con el residual multicomponente completo y los criterios de malla y de convergencia no cambian.

## Uso

```python
import kflame

flame = kflame.solve_flame(mechanism='jetsurf2.yaml', fuel='NC7H16', phi=0.8, temperature=400.,
                           transport='multicomponent')             # podado automático (>= 200 especies)
exact = kflame.solve_flame(mechanism='jetsurf2.yaml', fuel='NC7H16', phi=0.8, temperature=400.,
                           transport='multicomponent', multicomponent_trace=None)   # núcleo exacto
```

`multicomponent_trace` acepta `'auto'` (por defecto), `None` (núcleo exacto) o un valor de tail en (0, 10⁻²). `metadata.json` registra el modelo usado.

## Llamas completas

| Caso | Exacto | Podado | Diferencia |
|---|---:|---:|---|
| JetSurF 2.0, premezclada CH₄/aire φ = 1, MC | 45,4 s | 37,7 s | Su idéntica a 10 cifras (0,3647264429 m/s) |
| GRI-Mech 3.0, contraflujo CH₄/aire, MC (podado forzado) | — | — | ΔT máx 5,7·10⁻⁷ K, misma malla de 1288 nodos |
| GRI-Mech 3.0, premezclada, MC con GMRES forzado | 21,5 s | 42,5 s | Su 8·10⁻⁶ relativo (tolerancia de Newton); por eso el camino GMRES solo es automático desde 400 especies |

Abierto ([estado por mecanismo](estado-mecanismos.md#pendiente)):

- **Contraflujo MC con JetSurF y Aramco.** GMRES precondicionado por el Jacobiano de mezcla converge en 2 iteraciones en mallas de 61–121 nodos, pero no en 12 sobre la malla fina de 1510 nodos ([`audit_multicomponent_krylov.py`](../examples/audit_multicomponent_krylov.py)).
- **Premezclada MC con Aramco.** Falta la llama completa.

[`validate_multicomponent_trace.py`](../examples/validate_multicomponent_trace.py) recorre H₂, CH₄, C₂H₄, n-heptano (pobre y rico) y n-dodecano a 10 atm con JetSurF, en premezcladas y contraflujo, frente al núcleo exacto. Las premezcladas ya se pueden ejecutar; los casos de contraflujo dependen del punto abierto.
