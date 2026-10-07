# Llamas no premezcladas en contraflujo

`solve_diffusion_flame` resuelve una llama de difusión axisimétrica entre dos
entradas independientes. Combustible y oxidante se mezclan dentro del dominio.
El mecanismo, las composiciones molares y las condiciones de entrada son
parámetros; las ecuaciones y la elección automática del progreso no contienen
coeficientes particulares de CH4 o H2.

```python
from kflame import solve_diffusion_flame, generate_diffusion_fgm, DiffusionFGM

settings = dict(mechanism='gri30.yaml', fuel='CH4', oxidizer='O2:1,N2:3.76',
                fuel_temperature=300., oxidizer_temperature=300.,
                pressure=101325., width=.02,
                fuel_mass_flux=.12, oxidizer_mass_flux=.24)
flame = solve_diffusion_flame(**settings, output='runs/ch4_diffusion')
family = generate_diffusion_fgm(**settings, flow_scales=(.75, 1., 1.25),
                               initial_solution=flame['output'],
                               output='runs/ch4_diffusion_fgm')
table = DiffusionFGM(family['output'])
# Consultar Z,C dentro de las celdas reales de la familia.
```

Ambos caudales son magnitudes positivas en kg/(m² s). El flujo axial sale de
la entrada de combustible con signo positivo y de la entrada de oxidante con
signo negativo. Para H2/aire se puede usar `h2o2.yaml`, `fuel='H2'` y los
caudales `.05` y `.15`. `transport='multicomponent'` y `soret=True` son opciones
independientes. No se asigna una velocidad laminar `Su` a esta configuración.

La API guarda `flame.npz`, todas las especies en `profiles.csv` y
`metadata.json`, incluyendo el mecanismo, su hash, el hash del código,
las condiciones físicas, las tolerancias y el historial de refinamiento.
`plots=True` añade perfiles PNG/PDF. Una solución rechazada conserva sus
diagnósticos y lanza `RuntimeError`. La continuación solo admite resultados
nativos aceptados con los mismos flujos de entrada químicos y transporte;
permite cambiar caudales y separación. Un directorio existente se rechaza.

```sh
flampy diffusion --fuel CH4 --output runs/ch4_diffusion
flampy diffusion --mechanism h2o2.yaml --fuel H2 --fuel-mass-flux .05 --oxidizer-mass-flux .15 --soret --output runs/h2_diffusion
flampy diffusion-fgm --initial-solution runs/ch4_diffusion --output runs/ch4_diffusion_fgm
flampy reference-diffusion --native runs/ch4_diffusion --output runs/ch4_cantera
python examples/example_diffusion.py --output runs/diffusion_example --reference
```

## Ecuaciones y convergencia

Las incógnitas son el caudal axial firmado `m=ρu`, la tasa de expansión radial
`V`, `T`, todas las fracciones másicas y el autovalor constante de presión
radial `Lambda`. Se resuelven continuidad `m' + 2ρV = 0`, momento radial,
especies y energía con química y transporte nativos. El gradiente convectivo
cambia de lado al cambiar el signo de `m`. Las dos entradas satisfacen
`m(Y-Y_feed)+J=0`; la normalización sustituye una ecuación de especie en cada
entrada, mientras el interior conserva todas las ecuaciones diferenciales.

Newton usa un Jacobiano disperso por colores, factorizaciones reutilizadas y
amortiguación basada en el error. Si la estimación inicial no converge, usa
equilibrio HP nativo y continuación temporal implícita. Ninguna referencia
de Cantera inicializa una solución nativa. Se exige residuo escalado completo
menor que `1e-7`, conservación de la suma de especies y fracciones no menores
que `-1e-7`. La rama de llama debe alcanzar una temperatura al menos 100 K
superior a ambas entradas; configuraciones de entrada muy calientes pueden
necesitar otro criterio físico de identificación de combustión.

Los valores predeterminados son `ratio=3`, `slope=.01`, `curve=.02`,
`max_points=2400` y `max_time=600` s. La cota temporal corresponde a la fase de
resolución y puede sobrepasarse durante una operación individual. Una malla
fija (`refine=False`) acepta convergencia algebraica y declara
`grid_converged=False`; requiere un estudio espacial independiente.

La viscosidad de Wilke se calcula en un kernel Numba sin el tensor temporal
especies × especies × caras. La construcción del Jacobiano agrupa los valores
en arrays, evitando millones de objetos escalares Python en mecanismos grandes.
La continuación desde una llama aceptada evita repetir la inicialización fría.
Al cambiar la deformación, una malla inicial automática muy fina se reduce a
un subconjunto de hasta 201 nodos para la estimación inicial y después se refina
de nuevo con los mismos criterios. Una malla explícita del usuario se conserva.

## Entalpía y tabulación

`h` incluye entalpía sensible y de formación. También se guardan los flujos
conductivo, difusivo y axial total, y `2ρVh`, la exportación radial de entalpía.
El balance continuo es `(mh + q_conducción + Σh_k J_k)' + 2ρVh = 0`.
El informe calcula un defecto integrado entre la primera y la última cara.
Es un diagnóstico de discretización: la convergencia algebraica y una buena
comparación de temperatura no garantizan por sí solas un balance integral
preciso. No se aplica la identidad planar de caudal constante a esta geometría.

La familia varía ambos caudales mediante `flow_scales`. Cada llama aporta su
trayectoria espacial en **Z local de Bilger**. Se construyen triángulos entre
trayectorias adyacentes en `(Z,C)`, con `C=Σα_kY_k` común a toda la familia y
su fuente `Σα_kω_k`. Los pesos se pueden proporcionar; la selección automática
busca una combinación lineal que separe las filas de deformación y admite
pesos con signo. Las trayectorias no se ordenan artificialmente por progreso.
Los pliegues se rechazan y las celdas degeneradas se excluyen con diagnóstico.
El dominio de mezcla se restringe a `0 ≤ Z ≤ 1`. Los excesos de Z causados
por difusión diferencial quedan fuera de la tabla y se registran como puntos
excluidos; sus composiciones no se recortan ni se proyectan sobre Z=0 o Z=1.
Las consultas fuera de las celdas disponibles lanzan `OutsideManifoldError`.

La interpolación recupera T a partir de la **entalpía total interpolada y Y**,
sin recomputar química detallada. Esta tabla de dos controles contiene las
historias de entalpía de las llamas de entrenamiento. No ofrece todavía un
control independiente de pérdidas térmicas `(Z,C,h)`, radiación ni un solver
reducido de contraflujo. Esas capacidades requieren una familia física adicional
y cierres de transporte propios de esta geometría.

`validate_diffusion_fgm(table, native, output)` compara una llama nativa
reservada. Declara todos los puntos no cubiertos, comprueba por separado la
cobertura de la zona reactiva y mide T, Y, fuentes y consistencia de entalpía.
Los errores se calculan exclusivamente en estados cubiertos y no se extrapola.

## Referencia

La comparación opcional crea `CounterflowDiffusionFlame` desde una estimación
propia de Cantera con las mismas condiciones físicas, sin desplazar perfiles ni
usar la malla nativa como estimación inicial. Guarda la solución completa y un
informe con límites explícitos: 20 K para T, `.005` para Y, 3% para velocidad,
2% para el autovalor y 5% para el pico de liberación de calor.

Se consultaron las [ecuaciones oficiales de Cantera](https://cantera.org/stable/reference/onedim/governing-equations.html),
el [ejemplo de llama de difusión](https://cantera.org/stable/examples/python/onedim/diffusion_flame.html)
y las implementaciones de [Flow1D en 3.2.0](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/Flow1D.cpp)
y [las fronteras de entrada](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/Boundary1D.cpp).
La referencia y el solver nativo desactivan radiación en estas comparaciones.

## Resultados verificados el 7 de octubre de 2026

Con ambas entradas a 300 K, 101325 Pa, separación de 0.02 m y transporte
promediado sin Soret, pasan todos los límites de comparación independiente.

| Caso | Caudales combustible/aire, kg/(m² s) | Nodos nativos | Error máximo T frente a Cantera | Error máximo Y |
|---|---:|---:|---:|---:|
| CH4/aire | .12 / .24 | 1322 | 0.640 K | 0.000432 |
| H2/aire | .05 / .15 | 1071 | 4.502 K | 0.000523 |

H2 con transporte multicomponente y Soret también pasa: 4.284 K y 0.000575
de diferencia máxima. Son condiciones ensayadas, no garantías para todos los
combustibles, presiones o proximidad a extinción.

Las dos familias entrenan escalas de caudal `.75, 1., 1.25`; se reserva `1.1`.
Ambas cubren el 100% de los puntos de la zona reactiva definida en la validación.

| Tabla | Error máximo T reservado | Error máximo Y reservado | Error de fuente de especies, relativo al máximo global | Puntos cubiertos / totales |
|---|---:|---:|---:|---:|
| CH4/aire | 0.089 K | 0.0000214 | 0.493% | 1254 / 1390 |
| H2/aire | 1.776 K | 0.0000258 | 0.242% | 934 / 1070 |

El error máximo de entalpía total de interpolación es 164 J/kg para CH4 y
2642 J/kg para H2; dividido por el cp del estado reservado equivale, como
máximo, a 0.077 K y 0.708 K. La inversión termodinámica reproduce la entalpía
interpolada con error inferior a `1e-6` J/kg. Se declaran los estados fríos no
cubiertos, las celdas degeneradas y las exclusiones por Z fuera del intervalo
de mezcla. No se usan estos puntos para calcular errores de interpolación.

El defecto integral de energía de las llamas nativas base es 0.10% para CH4
y **4.25% para H2**, normalizado por el máximo flujo axial/exportación radial.
Este último necesita más refinamiento antes de interpretar pérdidas pequeñas;
el informe conserva ese defecto aunque la comparación de perfiles pase.

En esta máquina, con kernels Numba ya compilados y sin solución inicial,
el solver tomó aproximadamente 58 s para CH4 y 4.4 s para H2. La condición
reservada de CH4 necesitó 13.5 s con continuación. Estas son medidas locales
con otras validaciones concurrentes. El microbenchmark del ensamblaje del
Jacobiano de 377 nodos y 53 especies midió una mejora de 2.53×, con igualdad
exacta de las matrices. La suite completa pasa 148 pruebas.

Los [resultados, límites y hashes guardados](assets/diffusion/validation.json)
permiten auditar estas afirmaciones. `examples/example_diffusion.py` ejecuta
el flujo completo de generación y validación en un directorio nuevo.
