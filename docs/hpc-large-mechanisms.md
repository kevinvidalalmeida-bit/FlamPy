# Mecanismos grandes: Jacobiano estructurado, Newton–Krylov y transporte vectorizado

Revisión del 11 de octubre de 2026. [Guía de lectura](README.md) · [Mecanismos](mechanisms.md) · [Estado de estrategias](hpc-strategy-status.md) · [Datos](assets/diffusion/hpc-large-mechanisms-2026-10-10.json) · [Datos del 11 de octubre](assets/diffusion/hpc-large-mechanisms-2026-10-11.json)

**Resultado:** la llama de difusión completa CH₄/aire con AramcoMech 3.0 (581 especies, 3037 reacciones, ratio 3, slope 0,01, curve 0,02) pasa de **no terminar** (más de 640 s y agotamiento de los 32 GB de RAM a 1377 nodos) a **103–115 s** con 1438 nodos y solución aceptada (91 s con refinamiento multipunto), con un pico de memoria de 5,2 GB. El mismo camino acelera JetSurF, AramcoMech 2.0 y, en menor medida, los mecanismos pequeños. La física (residual, propiedades, criterios de malla y de convergencia) no cambia: sólo cambia cómo se calculan las direcciones de Newton.

En la **llama premezclada** del artículo de AramcoMech 3.0 (1,3-butadieno/aire, φ = 1, 295 K, 1 atm) el tiempo baja de 580 s a **178 s**, con la misma velocidad de llama (0,5435 m/s) y de 9,6 a 2,6 GB de memoria ([sección propia](#llama-premezclada-el-caso-de-validación-de-aramcomech-30)).

Llama completa CH₄/aire (H₂/aire para H2-O2), 300 K, 1 atm, mezcla promediada, 31 nodos iniciales, ratio 3, slope 0,01, curve 0,02. Tiempo del solver sin JIT, un proceso por caso, misma máquina (Ryzen 9 7845HX, 32 GiB). «Antes» es el camino anterior (`structured_jacobian=False`, `fast_transport=False`, `newton_krylov=False`).

| Mecanismo | Especies | Antes (s) | Ahora (s) | Aceleración | Nodos finales |
|---|---:|---:|---:|---:|---:|
| H2-O2 | 10 | 0,43 | 0,40 | mismo camino | 1032 |
| DRM19 | 21 | 1,07 | 0,99 | 1,1× | 1183 |
| DRM22 | 24 | 1,45 | 1,24 | 1,2× | 1196 |
| FFCM-1 | 38 | 2,29 | 1,89 | 1,2× | 1297 |
| GRI-Mech 3.0 | 53 | 3,04 | 2,08 | 1,5× | 1322 |
| USC Mech II | 111 | 11,9 | 6,6 | 1,8× | 1432 |
| JetSurF 2.0 | 348 | 166 | 45,9 | 3,6× | 1510 |
| AramcoMech 2.0 | 493 | no medido | 89,5 (87,2 multipunto) | — | 1411 |
| AramcoMech 3.0 | 581 | **no termina** (>640 s, 32 GB agotados a 1377 nodos) | **115** (91 multipunto) | de no terminar a 2 min | 1438 |

En todos los mecanismos el camino nuevo produce la **misma malla final** y la misma solución que el anterior: ΔTmax ≤ 2·10⁻⁷ K, limitado por la tolerancia de Newton. Por debajo de 20 especies se conserva el camino anterior: H2-O2 tardaba 1,6 s con el nuevo frente a 0,42 s.

## Por qué fallaba Aramco 3.0

- **Bloques densos.** El Jacobiano por diferencias finitas almacenaba bloques densos de 584×584 (2,7 MB) por nodo: tres por nodo más la inversa y el acoplamiento de la factorización. Con 1377 nodos son unos 19 GB y el sistema empezaba a paginar.
- **Coste O(n·nv³).** Cada factorización invertía un bloque y hacía dos productos densos por nodo (≈ 30 ms por nodo). La malla de 1191 nodos costaba más de 90 s por factorización.
- **Malla inicial.** El pseudo-transitorio necesitaba unas 80 evaluaciones del Jacobiano de 31 nodos (≈ 2 s cada una): 160 s antes del primer refinamiento.
- **Equilibrio inicial.** Catorce especies de Aramco declaran T_max = 2000 K y el intervalo NASA común termina ahí. La búsqueda del equilibrio HP se limitaba a [300, 2000] K, pero la llama estequiométrica supera 2200 K. CH₄/aire no podía ni empezar; ocurría lo mismo con USC Mech II y JetSurF. Ahora el intervalo se amplía a 2·T_max si hace falta, la misma cota que ya usaba el resolvedor.

## Qué se implementó

### 1. Jacobiano analítico estructurado (`flame/structured_jacobian.py`)

El residual de mezcla promediada se deriva analíticamente con los coeficientes de transporte de cara congelados en composición (ρW_kD_k/W̄, D^T, λ y μ). Su dependencia de la temperatura se conserva con una evaluación extra de la cara a T_f(1+10⁻⁶). Con coeficientes congelados, el flujo de una cara depende de la composición de cada nodo como **diagonal + dos términos de rango uno** (gradiente molar con W̄ y velocidad de corrección). Por tanto cada bloque fuera de la diagonal es `diag(d) + P Qᵀ` con como mucho **cinco columnas**: dos de especies, columna de temperatura, fila de energía y fila de continuidad. Los bloques diagonales siguen densos porque contienen el Jacobiano químico analítico.

Ensamblar un nodo cuesta O(ns²), frente a O(ns²) **por columna** (O(ns³) por nodo) de las diferencias finitas. Para Aramco 3.0 son ≈ 0,9 ms por nodo frente a ≈ 13 ms.

El residual físico sigue siendo el completo: el Jacobiano sólo da forma a la dirección y cada paso se acepta con el residual exacto. Las soluciones convergidas coinciden con las del camino anterior en la misma malla (GRI: 2·10⁻⁷ K; Aramco 3.0: 9·10⁻¹³ K entre dos versiones del nuevo camino).

### 2. Factorización estructurada de dos extremos (`flame/structured_linear.py`)

El complemento de Schur de un pivote sólo necesita su inversa G y los productos de rango r `G P` y `Qᵀ G`:

S_{i+1} = D_{i+1} − (d_l + P_l Q_lᵀ) G_i (d_u + P_u Q_uᵀ)

Así desaparecen los dos productos densos por nodo y su almacenamiento: un nodo cuesta una inversión LAPACK más O(r·nv²). En la malla de 31 nodos de Aramco 3.0 la factorización pasa de 0,88 s a 0,27 s, con la misma solución (diferencia relativa 5·10⁻¹²). En mallas grandes la factorización estacionaria se hace **in situ**: cada inversa sobrescribe su bloque diagonal y la memoria por nodo baja de 13,6 MB a 2,7 MB. Si después hiciera falta otro Δt, el Jacobiano analítico se recalcula; cuesta ≈ 1 s.

**Ensamblaje fusionado por tramos (nv ≥ 200, `LAZY_MIN_VARIABLES`).** En los mecanismos grandes los bloques diagonales ya no se guardan. Se calculan por tramos de 48 MiB (`CHUNK_BYTES`) dentro de la propia factorización:

- un hilo auxiliar ensambla el tramo siguiente mientras el barrido elimina el actual;
- el barrido descendente y el ascendente avanzan a la vez y se encuentran en el nodo central;
- cada pivote se invierte en float64 y se guarda en float64 (`STORAGE`); el complemento de Schur siguiente usa esa misma inversa.

Las operaciones y su orden son los de la factorización completa: en la malla inicial de Aramco 3.0 da los mismos 39 Jacobianos y la misma Tmax bit a bit, en 35,0 s frente a 39,3 s (4 hilos). La memoria es la de las inversas (n·nv²·8 bytes) más cuatro tramos.

Guardar las inversas en float32 reduciría esa memoria a la mitad, pero el redondeo perturba la norma de amortiguamiento: la malla inicial pasó a 55 Jacobianos y 47 s. Por eso se descartó (ver la tabla de estrategias descartadas).

**Pivotaje.** LAPACK lee el bloque en orden C como su traspuesta. Por eso se traspone in situ antes y después de invertir: así la LU pivota sobre las filas del propio bloque, igual que `numpy.linalg.inv`. Una versión intermedia invertía la traspuesta sin trasponer, lo que equivale a pivotar por columnas. Con bloques químicos de número de condición 10¹⁸–10²¹, la malla inicial de Aramco 3.0 pasó de 39 a 90 Jacobianos. Además, en la malla de 61 nodos el Newton estacionario fallaba en la búsqueda lineal y la llama completa superaba los 9 minutos. Una prueba fija ahora la secuencia de pivotes.

### 3. Memoria acotada (`flame/buffer_pool.py`)

Windows rellena con ceros cada página nueva en el primer acceso (≈ 0,8 s por GB). Por eso los arreglos grandes de una misma resolución salen de un depósito de búferes reutilizables: Jacobianos, inversas, tramos y teselas de química. El depósito nunca cambia memoria por velocidad:

- cada búfer tiene el tamaño exacto pedido;
- sólo se presta de nuevo cuando no queda ninguna vista viva de él;
- los búferes grandes ociosos se liberan antes de reservar otro, así que la malla anterior nunca convive con la nueva;
- lo ocioso retenido no supera 512 MiB (`MAX_FREE_BYTES`).

Además, el modelo lineal anterior se suelta antes de construir uno nuevo, y en mecanismos grandes sólo existe una factorización a la vez. Al terminar la llama, el depósito se libera.

Una primera versión del depósito contaba mal las referencias de un búfer ocioso, así que nunca reutilizaba ni liberaba nada: la llama de Aramco 3.0 llegaba a 16–21 GB y el equipo dejaba de responder. Con la versión actual, la llama completa de Aramco 3.0 tiene un pico de 5,2 GB, de los cuales ≈ 3,9 GB son las inversas de la malla final.

### 4. Newton–Krylov con factorización envejecida (`_gmres` en `flame/diffusion.py`)

Para mecanismos de al menos 400 especies, la dirección de Newton se obtiene con GMRES precondicionado por la derecha. El producto Jv se calcula sin matriz con el residual completo: (F(x+εv) − F(x))/ε, con ε fijado por las escalas de la norma de Newton. El precondicionador es la factorización estructurada existente, aunque esté envejecida. La primera dirección de Krylov es el propio paso factorizado, así que GMRES nunca es peor que el paso anterior en norma del residual.

El Jacobiano sólo se reconstruye cuando GMRES no alcanza la tolerancia (10⁻² relativa en 12 iteraciones) o cuando el modelo supera su edad máxima. En el pseudo-transitorio, una factorización se reutiliza mientras 1/Δt no cambie más de un factor 8; GMRES corrige el desajuste (en CVODE el criterio equivalente es |γ/γ̄ − 1| > 0,3). En la malla inicial de Aramco 3.0 las evaluaciones del Jacobiano bajan de 79–153 a 35–44.

Por debajo de 400 especies una factorización nueva sale más barata que los residuos adicionales de Krylov: GRI 2,2 s frente a 3,2 s, USC 5,7 s frente a 7,9 s, y JetSurF empata (42,8 s frente a 43,5 s). Por eso esos mecanismos conservan el Newton directo.

### 5. Transporte de caras vectorizado y por bloques (`flame/fast_transport.py`)

Mismo cierre de mezcla promediada (difusividades de mezcla, viscosidad de Wilke y conductividad combinada), con otra organización:

- no se escriben temporales O(ns²) por cara: 1/D_kj y la matriz de Wilke se acumulan directamente;
- un bloque de 8 caras comparte una pasada sobre cada fila del mecanismo, así que las tablas (cinco coeficientes de ajuste y dos constantes de Wilke) se leen una vez por bloque;
- se permite reasociar las sumas (`fastmath` limitado a `reassoc` y `contract`), lo que habilita AVX-512. Un único bucle enmascarado sustituye a los dos rangos alrededor de k: los rangos partidos no se vectorizaban y costaban 3,7× más.

Para Aramco 3.0 se pasa de 1,29 ms a 0,24 ms por cara (5,3×), y de 4,3× para GRI. Las diferencias con el núcleo escalar son de redondeo: λ 10⁻¹³, μ 5·10⁻¹⁵ y flujos 10⁻¹⁵ relativos al máximo. La variante con Soret añade las difusividades con fracciones molares recortadas a 10⁻²⁰, los a_k de tipo Wilke y los ajustes C*_kj en log(T/ε_kj) precalculando log ε_kj. Es 3× más rápida (Aramco 3.0: 4,8 → 1,6 ms por cara). El transporte multicomponente conserva su núcleo.

### 6. Derivadas químicas analíticas dispersas (`chemistry/analytic.py`)

Cada monomio de reactivos y productos sólo toca sus propias especies. La eficiencia de tercer cuerpo por defecto (1 para todas las especies) se suma al final como un único término de rango uno u·1ᵀ, y sólo se dispersan las eficiencias no unitarias. Antes cada reacción recorría vectores de longitud ns (borrado, eficiencias, M y la columna dq). El cálculo es matemáticamente idéntico: 2,7× para Aramco 3.0 (4,3 → 1,6 ms por nodo), 2× para USC y diferencias de 10⁻¹⁶ relativas al máximo.

### 7. Hilos

Las caras y el ensamblaje analítico de mecanismos de al menos 100 especies usan por defecto la mitad de las CPU lógicas menos dos: 10 hilos en este portátil de 12 núcleos y 24 hilos lógicos. Así quedan dos núcleos físicos libres y el equipo sigue respondiendo. `KFLAME_THREADS` fija otro valor, y nunca se usan menos hilos que `NUMBA_NUM_THREADS`, que mantiene su valor por defecto (4).

### 8. Opcional: refinamiento multipunto

`solve_diffusion_flame(..., refine_split=4)` (`--refine-split 4`) divide cada intervalo marcado en min(4, ⌈r⌉) partes. Aquí r es el peor cociente de los criterios de pendiente y curvatura. La malla final debe volver a superar los criterios sin cambios. Con Aramco 3.0 se resuelven 6 niveles en lugar de 10 y la suma de nodos factorizados baja de 6221 a 3949. Se obtienen 1352 nodos finales y unos 91 s; Tmax difiere 0,008 K de la secuencia de puntos medios. Por defecto se mantiene el punto medio (`refine_split=2`).

## Llama premezclada: el caso de validación de AramcoMech 3.0

AramcoMech 3.0 se publicó validado con velocidades de llama laminar de 1,3-butadieno/aire (Zhou et al., Combust. Flame 197, 2018, 423–438): 295, 359 y 399 K, 1 atm, φ de 0,6 a 1,7, y 5 atm, φ de 0,6 a 1,4. Las velocidades se calcularon con CHEMKIN-PRO y el artículo no da tiempos de cálculo. Aquí se resuelve su punto central con `solve_flame`: C4H6/aire, φ = 1, 295 K, 1 atm, ajustes por defecto, de una sola vez y sin semilla.

| Camino | Tiempo del solver (s) | Pico de memoria | Su (m/s) | Nodos |
|---|---:|---:|---:|---:|
| Anterior: bloques analíticos densos | 580 | 9,6 GB | 0,543535494872 | 390 |
| Jacobiano estructurado | 236 | 2,6 GB | 0,543535494872 | 390 |
| Estructurado + escalera de Δt (por defecto) | **178** | 2,6 GB | 0,543535494872 | 390 |

Su coincide en las tres variantes hasta 10⁻¹⁴. Con el mismo caso, USC Mech II (111 especies) baja de 9,0 a 4,1 s y GRI 3.0 con CH₄ de 3,05 a 1,81 s, con la misma Su y la misma malla.

**Qué cambia.** `flame/premixed_structured.py` ensambla la misma derivada que `analytic_jacobian.assemble_blocks` (transporte de cara congelado, química analítica, mismas ramas de convección y de contorno), pero en forma estructurada. Los bloques diagonales son densos. Cada bloque fuera de la diagonal es `diag(D) + P Qᵀ` de rango 5: fila de continuidad, fila de energía, los dos términos de rango uno de la cara (gradiente molar y velocidad de corrección) y la columna de temperatura (Soret). La química se deriva por teselas, sin guardar el tensor nodos × ns × nv. La factorización es la de dos extremos de `structured_linear`: una inversa LAPACK por nodo y ningún producto denso. Se activa con transporte de mezcla promediada (con o sin Soret) a partir de 20 especies y sin umbral de poda (`jacobian_threshold = 0`). Para volver al camino anterior: `problem.structured_jacobian = False`.

**Escalera de Δt.** El pseudo-transitorio PTC-SER cambia Δt en cada paso, y cada cambio obligaba a refactorizar. `time_step_ladder='auto'` (valor por defecto de `solve_flame`) redondea Δt hacia abajo a potencias de 4 cuando el mecanismo tiene al menos 100 especies. Mientras Δt sigue en el mismo peldaño, la factorización se reutiliza tal cual. Con USC Mech II las factorizaciones bajan de 458 a 240. Cambia la trayectoria del pseudo-transitorio, no la solución: la llama final pasa el mismo criterio de Newton estacionario. `time_step_ladder=None` conserva los pasos continuos.

**Referencias de la literatura (no medidas aquí; otro hardware y otros casos).**

- En el foro de usuarios de Cantera, una llama premezclada con AramcoMech 2.0 (493 especies) tardaba 1,5 h con Cantera 2.5.1, frente a unos 10 s con GRI 3.0 y 90 s con USC Mech II.
- Lapointe et al. (2019) dan 30 000–70 000 s con Chemkin para un barrido de 8 dosados con un mecanismo de 2878 especies en 16 núcleos, y 500–1000 s con su resolvedor de Jacobiano aproximado.

Sólo sirven como orden de magnitud: aquí la llama completa de Aramco 3.0, con 581 especies, tarda unos 3 minutos en un portátil.

### Caso de la Tabla 3.2 de de Boer (2024)

La tesis de máster de Y. de Boer (TU Eindhoven, 2024, *Flame speed based 2-zone combustion model for methanol-diesel dual fuel engines*, p. 28, Tabla 3.2) da el tiempo de una llama libre de Cantera a ϕ = 0,5, T = 600 K, p = 40 atm, con «criterios de refinamiento medios». Por el contexto de la tesis el combustible es metanol, aunque el pie de la tabla no lo dice. La tesis tampoco da los valores de refinamiento, el modelo de transporte ni el ordenador.

Aquí se resolvió el mismo caso: CH₃OH/aire, ϕ = 0,5, 600 K, 40 atm, mezcla promediada. Se usaron dos juegos de refinamiento: los de `solve_flame` (ratio 2,5, slope 0,04, curve 0,08) y los del ejemplo habitual de Cantera (3; 0,06; 0,12).

| Mecanismo | de Boer, Cantera (s) | Aquí, refinamiento por defecto (s) | Aquí, refinamiento del ejemplo de Cantera (s) |
|---|---:|---:|---:|
| GRI-Mech 3.0 | 15 | 4,8 (Su 0,0956 m/s, 240 nodos) | 2,8 (Su 0,0965 m/s, 165 nodos) |
| AramcoMech 3.0 | 9195 | **117** (Su 0,1037 m/s, 241 nodos, 1,75 GB) | **116** (Su 0,1046 m/s, 178 nodos, 1,4 GB) |

El hardware y la versión de Cantera son distintos, así que el cociente (≈ 80× con Aramco 3.0) es indicativo y no una comparación controlada. Con Aramco 3.0 el tiempo apenas depende del refinamiento: domina el pseudo-transitorio en las mallas gruesas iniciales.

## Qué se probó y se descartó

| Estrategia | Resultado medido | Decisión |
|---|---|---|
| Precondicionador disperso por umbral (bloques químicos podados, LU de SuperLU) | GMRES sin converger en 800 iteraciones incluso con umbral 10⁻⁸; los bloques tienen número de condición 10¹⁸–10²¹ | Descartado |
| Factorización aproximada de Zero-RK (química por nodo × transporte tridiagonal) | Acoplamientos de rango uno (W̄, tercer cuerpo, corrección) imprescindibles | Descartado |
| Subespacio activo (especies traza desacopladas) | Radio espectral de I − P⁻¹J entre 0,27 y 0,96; 60–200 iteraciones de GMRES | Descartado |
| Bloques fuera de la diagonal comprimidos por SVD | Rango 16: contracción 2,6·10⁻³; rango 32: 1,8·10⁻⁵ | Motivó la forma estructurada analítica |
| Inversión de pivotes en aritmética float32 | 2,4·10¹⁰ de error ponderado en las componentes traza | Descartado |
| Inversas calculadas en float64 y guardadas en float32 | Malla inicial de Aramco 3.0: 55 Jacobianos y 47 s frente a 39 y 39 s | Descartado |
| LAPACK multihilo para la inversión | 1 hilo 11,8 ms; 4 hilos 13,3 ms; 6 hilos 16,8 ms | Descartado |
| Factorización particionada exacta (separadores, 8–12 segmentos) | Correcta (10⁻¹⁰), pero sin ganancia: el portátil da 5,7× de rendimiento agregado con 12 hilos y la partición cuesta 2,5× flops | Retirada del código |
| Malla inicial más gruesa o continuación desde GRI | 35–39 s frente a ≈ 25 s directos; el pseudo-transitorio no escala con n | Descartado |
| Δt inicial 10⁻⁶–10⁻⁵ o crecimiento 2× | Sin cambio en GRI (100–119 Jacobianos) | Sin cambio |
| Tolerancia de GMRES de Eisenstat–Walker (η entre 10⁻³ y 0,1) | Malla inicial de Aramco 3.0: 99 Jacobianos frente a 90 (medido con el pivotaje por columnas, ver §2) | Descartado |
| Premezclada: reutilizar la factorización de un Δt cercano como precondicionador de GMRES (matriz desplazada exacta) | USC: factorizaciones de 458 a 157, pero 8,7 iteraciones por resolución y 4,9 → 6,3 s; para Aramco 3.0 cada resolución costaría ≈ 2,6 s frente a ≈ 2,3 s de refactorizar | Descartado; se usa la escalera de Δt |
| No reconstruir el Jacobiano cuando GMRES no converge | 104 Jacobianos con 12 iteraciones y 109 con 25, frente a 90 (mismo pivotaje) | Descartado |

## Referencias

- S. Lapointe, R. A. Whitesides, M. J. McNenly, *Sparse, iterative simulation methods for one-dimensional laminar flames*, Combust. Flame 204 (2019) — Newton–Krylov con Jacobiano aproximado para llamas 1D (código [Zero-RK](https://github.com/LLNL/zero-rk)).
- M. J. McNenly, R. A. Whitesides, D. L. Flowers, *Faster solvers for large kinetic mechanisms using adaptive preconditioners*, Proc. Combust. Inst. 35 (2015) 581–587.
- D. A. Knoll, D. E. Keyes, *Jacobian-free Newton–Krylov methods: a survey of approaches and applications*, J. Comput. Phys. 193 (2004) 357–397.
- Y. Saad, M. H. Schultz, *GMRES*, SIAM J. Sci. Stat. Comput. 7 (1986); A. C. Hindmarsh et al., *SUNDIALS/CVODE* (criterio de reutilización de la matriz de Newton).

## Uso

Los cambios son el camino por defecto para transporte de mezcla promediada (con o sin Soret) a partir de 20 especies (`STRUCTURED_MIN_SPECIES`); Newton–Krylov se activa a partir de 400. Para reproducir el camino anterior en un problema:

```python
problem.structured_jacobian = False   # Jacobiano por columnas y bloques densos
problem.fast_transport = False        # núcleos escalares de cara
problem.newton_krylov = False         # Newton directo
```

`True` en cualquiera de las tres opciones las fuerza en cualquier tamaño. `kflame.flame.diffusion.KRYLOV_MIN_SPECIES` fija el umbral automático (400).

```sh
python examples/benchmark_diffusion_mechanisms.py --output runs/diffusion_mechanisms.json
python -m pytest -q tests/test_structured_newton.py
```

## Validación

- **Batería completa:** 1031 de 1043 pruebas pasaron con el camino nuevo por defecto. Las 12 restantes comparaban byte a byte las derivadas químicas con el núcleo denso anterior. Se cambiaron a tolerancia de redondeo, 10⁻¹³ del máximo, porque la acumulación dispersa reordena sumas. Tras el cambio pasan las 82 de ese archivo, y las 170 pruebas afectadas por los últimos cambios (Soret vectorizado y `refine_split`) volvieron a pasar.
- **`tests/test_premixed_structured.py` (18 pruebas):** los bloques estructurados de la premezclada coinciden con el ensamblaje denso hasta 10⁻¹³ en todas las ramas: libre y quemador, upwind y exponencial, con y sin Soret, base molar y másica, ambas condiciones de salida, con y sin energía. Las resoluciones con la diagonal transitoria coinciden hasta 10⁻⁹. Además, la llama de GRI estructurada reproduce la densa: misma malla, Su hasta 10⁻⁹ y T hasta 10⁻⁶ K.
- **`tests/test_structured_newton.py` (16 pruebas):**
  - el Jacobiano analítico coincide con diferencias centradas del residual con transporte congelado (H₂, con y sin Soret y radiación) dentro de 2·10⁻⁶ de la escala de cada columna;
  - la factorización estructurada, también in situ, resuelve igual que la densa;
  - GMRES alcanza la tolerancia;
  - las caras vectorizadas, con y sin Soret, coinciden con las escalares;
  - las derivadas dispersas coinciden con la referencia densa;
  - Newton estructurado y Newton por columnas llegan a la misma raíz;
  - el equilibrio HP de USC supera el límite NASA de 2000 K;
  - los cocientes de refinamiento marcan los mismos intervalos que el refinador;
  - la factorización fusionada por tramos resuelve igual que la completa (tramos forzados a pocos nodos);
  - la inversión de pivotes sigue la secuencia de pivotes por filas de LAPACK sobre el propio bloque;
  - el depósito de búferes reutiliza memoria ociosa y respeta su cota.
- **GRI con transporte congelado:** sólo 53 de 54 965 entradas difieren más de 10⁻⁴ de las diferencias congeladas, y todas en filas de contorno al nivel del ruido de redondeo de las diferencias finitas.
- **Llamas completas:** misma malla y ΔTmax ≤ 2·10⁻⁷ K entre camino nuevo y anterior en los siete mecanismos donde ambos terminan.
- **11 de octubre, tras la factorización por tramos, el depósito acotado y la corrección del pivotaje.** GRI, USC Mech II y JetSurF dan la misma malla, el mismo número de Jacobianos y la misma Tmax bit a bit que la campaña de la tabla:

  | Mecanismo | Tabla (s) | 11 oct (s) |
  |---|---:|---:|
  | GRI | 2,08 | 1,94 |
  | USC | 6,63 | 4,31 |
  | JetSurF | 45,9 | 36,1 |

  Aramco 3.0 tarda 102,8 s con perfilado: 1438 nodos, 50 Jacobianos y 5,2 GB de pico. Estas ejecuciones usaron prioridad alta.

## Límites

- El transporte **multicomponente** sigue con el Jacobiano por columnas y bloques densos. Con mecanismos grandes su residual es O(ns³) por cara (≈ 40 ms para Aramco 3.0) y una llama completa no es práctica. La reducción en el límite de especies traza, exacta a O(X_traza), es el siguiente paso natural.
- En la llama premezclada de Aramco 3.0 la factorización sigue siendo el coste principal: se refactoriza en cada cambio de peldaño de Δt y en cada Jacobiano nuevo (≈ 400 factorizaciones en total).
- Los tiempos de este portátil varían hasta un 30 % entre procesos (temperatura y frecuencia). Las cifras de la tabla son ejecuciones individuales en procesos separados; Con la versión final, Aramco 3.0 midió 102 y 115 s en dos ejecuciones independientes.
- La factorización es ahora ≈ 60 % del tiempo de Aramco 3.0. En esta máquina doce hilos sólo dan 5,7× de rendimiento agregado en inversiones y GEMM de 584×584, así que la eliminación en cadena está cerca de su límite práctico. Lo que queda es reducir nodos factorizados (`refine_split`) o el tamaño del sistema.
