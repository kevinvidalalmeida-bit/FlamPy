# Archivos de salida y consulta de tablas

[Inicio](../README.md) · [API](api.md) · [Script de consulta](../examples/query_fgm.py)

## Llamas individuales

| Archivo | Contenido |
|---|---|
| `flame.npz` | Perfiles completos y nombres de todas las especies |
| `profiles.csv` | Una fila por nodo; propiedades y especies seleccionadas |
| `metadata.json` | Condiciones, malla, criterios, tiempo y diagnóstico de aceptación |
| `profiles.png` / `profiles.pdf` | Figuras, si se solicita `plots=True` |

Sea `N` el número final de nodos y `K` el número de especies:

| Clave NPZ | Forma | Unidad |
|---|---|---|
| `z` | `(N,)` | m |
| `T` | `(N,)` | K |
| `u` | `(N,)` | m/s |
| `rho` | `(N,)` | kg/m³ |
| `cp_mass` | `(N,)` | J/(kg·K) |
| `conductivity` | `(N,)` | W/(m·K) |
| `qdot` | `(N,)` | W/m³ |
| `Y` | `(K, N)` | Fracción másica, adimensional |
| `species_names` | `(K,)` | Identificadores de especie |

~~~python
from pathlib import Path
import numpy as np

folder = Path("runs/flame/MI_EJECUCION")
with np.load(folder / "flame.npz", allow_pickle=False) as data:
    z = data["z"]
    temperature = data["T"]
    names = data["species_names"].astype(str).tolist()
    water = data["Y"][names.index("H2O")]
~~~

## Tablas FGM

| Archivo o carpeta | Contenido |
|---|---|
| `fgm_table.npz` | Ejes, propiedades, especies y estado de cada flamelet |
| `metadata.json` | Parámetros, aceptación global y comprobación estructural |
| `summary_phi.csv` | Resumen de las llamas calculadas |
| `raw_profiles/` | Perfiles espaciales antes de proyectarlos sobre el progreso |
| `continuation_schedule.json` | Composiciones iniciales y resueltas; filas solicitadas y añadidas |
| `continuation_trace.json` | Trayectoria e inicialización de las llamas |
| `adaptive_rounds.json` | Diagnóstico de las rondas, en el modo adaptativo |
| `flamelet_export/` | Exportación `.fla` y CSV, si `export=True` |

Con `plots=True` también se generan `fig8_ZC.pdf`, `Su_Z.pdf`,
`c_grid_distribution.pdf/.png` y `C_vs_Z_domain.pdf/.png`.

Sea `N_Z` el número de flamelets, `N_c` el número de puntos de progreso
y `K` el número de especies:

| Clave NPZ | Forma | Significado |
|---|---|---|
| `phi_grid` | `(N_Z,)` | Relaciones de equivalencia de las llamas |
| `Z_grid` | `(N_Z,)` | Fracción de mezcla evaluada en la entrada |
| `c_grid` | `(N_c,)` | Progreso normalizado entre 0 y 1 |
| `T`, `u`, `rho`, `cp_mass`, `conductivity`, `qdot` | `(N_Z, N_c)` | Propiedades tabuladas, con las unidades indicadas arriba |
| `omega_c` | `(N_Z, N_c)` | Fuente volumétrica normalizada de progreso, kg/(m³·s) |
| `Y` | `(N_Z, K, N_c)` | Fracciones másicas |
| `Su` | `(N_Z,)` | Velocidad laminar de cada flamelet, m/s |
| `final_accepted` | `(N_Z,)` | Aceptación numérica de cada llama |
| `requested` / `bridge` | `(N_Z,)` | Fila inicial o añadida por continuación/refinamiento |
| `predictor_kind` | `(N_Z,)` | Tipo de inicialización utilizado |

`Z_grid` identifica la composición de **entrada** de cada llama. Con
difusión diferencial, no equivale necesariamente al escalar de Bilger local
a lo largo del perfil. Si un gráfico utiliza `Z_star`, se trata de una
normalización para dibujar y no sustituye `Z_grid` en la consulta.
La [galería FGM reproducible](results.md) presenta `Z_in` sin reescalar
su intervalo, con la referencia de las corrientes en base molar.

## Consulta escalar y por lotes

Para un campo escalar, SciPy interpola directamente los dos ejes:

~~~python
import numpy as np
from scipy.interpolate import RegularGridInterpolator

with np.load("runs/fgm/MI_EJECUCION/fgm_table.npz", allow_pickle=False) as data:
    z_grid = data["Z_grid"]
    c_grid = data["c_grid"]
    temperature = data["T"]

interpolate_T = RegularGridInterpolator(
    (z_grid, c_grid),
    temperature,
    method="linear",
    bounds_error=True,
)

z_mid = 0.5 * (z_grid[0] + z_grid[-1])
points = np.array([[z_mid, 0.25], [z_mid, 0.50], [z_mid, 0.75]])
print(interpolate_T(points))  # K
~~~

Para consultar composición, mueve el eje de especies detrás de los dos ejes
de interpolación:

~~~python
with np.load("runs/fgm/MI_EJECUCION/fgm_table.npz", allow_pickle=False) as data:
    names = data["species_names"].astype(str).tolist()
    values = np.moveaxis(data["Y"], 1, -1)

interpolate_Y = RegularGridInterpolator(
    (z_grid, c_grid), values, bounds_error=True
)
water = interpolate_Y(points)[:, names.index("H2O")]
~~~

`bounds_error=True` rechaza consultas fuera de la tabla. La llamada recibe
puntos con forma `(n_consultas, 2)` y devuelve todos los resultados del
lote. [`examples/query_fgm.py`](../examples/query_fgm.py) integra esta carga
y consulta sin volver a resolver las llamas.

## Exportación

~~~sh
flampy export --npz runs/fgm/MI_EJECUCION/fgm_table.npz --all-species --single-flamelets
~~~

La carpeta `flamelet_export/` contiene `fgm_table_full.fla`,
`summary_ZC.csv` y los archivos individuales en `single_flamelets/`.
Los parámetros `--mech`, `--fuel`, `--oxidizer`, `--pressure` y `--T-in`
describen la cabecera: deben coincidir con las condiciones de generación.
La API los transmite automáticamente.

El exportador produce el formato de texto implementado en este repositorio.
El uso en un solver CFD concreto requiere adaptar y comprobar su lector y
sus convenciones de coordenadas, fuentes y unidades.
