# API Python de FlamPy

[Inicio](../README.md) · [Archivos y consulta](outputs.md) · [Ejemplos](../examples/README.md)

La interfaz pública expone dos funciones, con argumentos exclusivamente
por nombre:

~~~python
from kflame import solve_flame, generate_fgm
~~~

Ambas utilizan el núcleo nativo de CPU. La configuración interna del
Jacobiano, el amortiguamiento, PTC y el álgebra lineal permanece dentro del
procedimiento de resolución.

## `solve_flame`

Resuelve una llama libre premezclada adiabática a presión constante, guarda
sus resultados y devuelve un diccionario.

~~~python
result = solve_flame(
    mechanism="h2o2.yaml",
    fuel="H2",
    oxidizer={"O2": 1.0, "N2": 3.76},
    phi=1.0,
    temperature=300.0,
    pressure=101325.0,
    transport="multicomponent",
    soret=True,
    species=("H2", "O2", "H2O", "OH"),
)
~~~

### Composición de una llama

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `phi` | `None` | Relación de equivalencia positiva. Si no se proporciona composición, se utiliza φ = 1 |
| `X` | `None` | Cantidades molares mediante diccionario o cadena, normalizadas internamente |
| `Y` | `None` | Cantidades másicas mediante diccionario o cadena, normalizadas internamente |
| `diluent` | `None` | Composición del diluyente, por ejemplo `"N2"` |
| `dilution` | `0.0` | Fracción molar final del diluyente especificado; debe cumplir `0 <= dilution < 1` |

`phi`, `X` y `Y` son alternativas: proporciona como máximo una.
Con `phi`, la mezcla se construye a partir de `fuel` y `oxidizer`.
`dilution` requiere `diluent` cuando es distinta de cero.

~~~python
# Entrada directa en base molar; no se proporciona phi.
result = solve_flame(
    mechanism="h2o2.yaml",
    X={"H2": 2.0, "O2": 1.0, "N2": 3.76},
    species=("H2", "O2", "H2O", "OH"),
)
~~~

### Malla inicial y tolerancias

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `grid` | `None` | Nodos iniciales en metros, estrictamente crecientes desde 0 hasta `width` |
| `rtol` | `1e-4` | Tolerancia relativa de la corrección estacionaria ponderada |
| `atol` | `1e-9` | Tolerancia absoluta de la corrección estacionaria ponderada |

Una malla inicial explícita no desactiva la adaptación posterior. Las
tolerancias deben ser positivas y finitas; los umbrales internos de las
etapas transitorias y de aceptación final se gestionan por separado.

### Resultado

El diccionario contiene `z`, `T`, `u`, `Y`, `rho`, `cp_mass`,
`conductivity`, `qdot` y `species_names`, además de:

| Clave | Significado |
|---|---|
| `Su` | Velocidad laminar respecto de la mezcla no quemada, m/s |
| `accepted` | `True` cuando la resolución y sus criterios finales se han satisfecho |
| `report` | Diagnóstico de convergencia, aceptación y control espacial |
| `runtime_s` | Tiempo de construcción del problema y resolución, en segundos; excluye el posproceso y la escritura |
| `output` | `pathlib.Path` de la carpeta de salida |

`Y` tiene forma `(especies, nodos)`. Seleccionar `species` no modifica
el mecanismo químico ni las especies almacenadas en `flame.npz`.

## `generate_fgm`

Construye una tabla FGM sobre `(Z, c)` y devuelve su carpeta como
`pathlib.Path`, tras comprobar la aceptación de las llamas y la tabla.

~~~python
folder = generate_fgm(
    mechanism="gri30.yaml",
    fuel="CH4",
    phis=(0.7, 0.9, 1.0, 1.1, 1.4),
    progress_species="CO2:1.0,H2O:1.0,CO:1.0,H2:0.5",
    target_defect=0.01,
    export=True,
    plots=False,
)
~~~

### Familia y coordenada de progreso

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `phis` | `(0.7, 0.9, 1.0, 1.1, 1.4)` | Composiciones iniciales, positivas, finitas y estrictamente crecientes |
| `progress_species` | `"CO2:1.0,H2O:1.0,CO:1.0,H2:0.5"` | Pesos de especies para definir el progreso de reacción |
| `progress_points` | `241` | Número solicitado de puntos sobre el eje adaptativo de progreso |
| `adaptive_phi` | `True` | Resolver nuevos flamelets para refinar la composición |
| `target_defect` | `0.01` | Umbral interno de defecto por exclusión; `0.01` equivale al 1 % |
| `max_bridges_per_round` | `10` | Máximo de composiciones nuevas por ronda |
| `max_adaptive_rounds` | `64` | Límite de rondas de refinamiento |
| `max_flamelets` | `256` | Límite de llamas en la familia adaptativa |
| `export` | `True` | Exportar todas las especies a archivos `.fla` y CSV |

El modo adaptativo requiere **al menos tres** valores iniciales de `phis`.
Con `adaptive_phi=False` se permite una familia fija de al menos dos valores.
Una familia fija sigue utilizando la adaptación espacial de cada llama y
la redistribución del eje de progreso.

Para iniciar una familia adaptativa de hidrógeno con el progreso usado
en la galería:

~~~python
folder = generate_fgm(
    mechanism="h2o2.yaml",
    fuel="H2",
    phis=(0.7, 0.9, 1.0, 1.1, 1.4),
    progress_species="H2O:1.0,HO2:10.0,OH:-1.0",
    species=("H2O", "OH"),
    plots=True,
)
~~~

Este ejemplo refina el intervalo φ = 0,7–1,4. Las figuras publicadas
proceden de 71 composiciones fijas entre 0,5 y 5; para reproducirlas,
utiliza los [datos y el generador incluidos](results.md#reproducir-las-imágenes).

Los pesos se comprueban sobre las trayectorias calculadas. El defecto por
exclusión dirige el refinamiento; no es una cota rigurosa del error en
todos los estados ni sustituye la comprobación con llamas independientes.

`X`, `Y`, `diluent`, `dilution`, `grid`, `rtol` y `atol` pertenecen
a `solve_flame`. Para diluir una familia FGM, incluye el diluyente en
las corrientes `fuel` u `oxidizer`.

## Parámetros comunes

| Parámetro | Valor por defecto | Descripción y unidades |
|---|---|---|
| `mechanism` | `"gri30.yaml"` | Mecanismo incluido o ruta local a un YAML compatible |
| `temperature` | `300.0` | Temperatura de la mezcla de entrada, K |
| `pressure` | `101325.0` | Presión termodinámica constante, Pa |
| `fuel` | `"CH4"` | Composición molar del combustible, cadena o diccionario |
| `oxidizer` | `"O2:1, N2:3.76"` | Composición molar del oxidante, cadena o diccionario |
| `width` | `0.03` | Longitud inicial del dominio, m; puede ampliarse |
| `initial_points` | `8` | Número de nodos iniciales, al menos 3 |
| `transport` | `"mixture-averaged"` | `"mixture-averaged"` o `"multicomponent"` |
| `soret` | `False` | Termodifusión; disponible con los dos modelos de transporte |
| `ratio` | `2.5` | Criterio de relación entre tamaños de intervalos, al menos 2 |
| `slope` | `0.04` | Umbral normalizado de variación, en `(0, 1]` |
| `curve` | `0.08` | Umbral normalizado de cambio de pendiente, en `(0, 1]` |
| `prune` | `0.003` | Umbral de eliminación; `0 <= prune < min(slope, curve)` |
| `max_points` | `1600` | Máximo de nodos por llama, no menor que `initial_points` |
| `max_time` | `180.0` | Presupuesto interno por llama, s; no incluye todos los costes externos ni garantiza una duración total exacta |
| `output` | `None` | Carpeta nueva; por defecto se crea una ruta fechada dentro de `runs/` |
| `plots` | `False` | Generar figuras; requiere el extra `plots` |
| `verbose` | `False` | Mostrar la traza del solver; FGM mantiene sus resúmenes por llama |
| `species` | Véase abajo | Especies de salida para las gráficas y, en una llama, el CSV |

`species` es `("CH4", "O2", "CO2", "H2O", "OH")` en `solve_flame`
y `("CO2", "H2O")` en `generate_fgm`. Las figuras FGM requieren
exactamente dos especies presentes en el mecanismo. La exportación FGM
activada con `export=True` conserva todas las especies.

La API FGM utiliza un proceso y desactiva la caché persistente de semillas.
La línea de comandos expone opciones adicionales para campañas; no todos
sus ajustes internos forman parte de estas dos funciones.

## Errores y diagnóstico

| Situación | Comportamiento |
|---|---|
| Parámetro desconocido | `TypeError` por la firma de la función |
| Entrada o combinación inválida | `ValueError` en las comprobaciones de la API |
| Ruta de mecanismo inexistente | `FileNotFoundError`; no se sustituye silenciosamente |
| Carpeta `output` ya existente | `FileExistsError`; utiliza una ruta nueva |
| Llama rechazada | `RuntimeError` después de guardar los campos y el diagnóstico |
| Límite adaptativo sin alcanzar el objetivo | Error con los diagnósticos de la construcción |

Consulta `metadata.json` y las trazas de la ejecución. Alcanzar el límite
de nodos o de tiempo no permite omitir los criterios de aceptación.
