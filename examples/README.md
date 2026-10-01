# Ejemplos de FlamPy

[Inicio](../README.md) · [API](../docs/api.md) · [Archivos y consulta](../docs/outputs.md)

Desde la raíz del repositorio:

~~~sh
python -m pip install -e ".[plots]"
~~~

| Script | Qué hace |
|---|---|
| [`example.py`](example.py) | Resuelve CH₄–aire, guarda una matriz de perfiles y crea figuras modificables |
| [`example_fgm.py`](example_fgm.py) | Construye una familia adaptativa y dibuja perfiles y mapas FGM |
| [`query_fgm.py`](query_fgm.py) | Carga una tabla y consulta temperatura y composición por lotes |

## Llama individual

~~~sh
python examples/example.py
~~~

Edita `CASE` para cambiar las condiciones. `PLOT_SPECIES` selecciona las
especies del CSV y de las figuras; no reduce el mecanismo químico.
Además de las salidas de la API, el ejemplo escribe:

- `solution_matrix.csv`: una fila por nodo con posición, temperatura,
  velocidad, liberación de calor y fracciones másicas seleccionadas.
- `custom_flame_plots.pdf`: figuras creadas desde los arrays de `flame.npz`.

Para H₂–aire, utiliza `mechanism="h2o2.yaml"`, `fuel="H2"` y
`PLOT_SPECIES = ("H2", "O2", "H2O", "OH")`.
`soret=True` se admite con `"mixture-averaged"` y `"multicomponent"`.

## Tabla adaptativa

~~~sh
python examples/example_fgm.py
~~~

Los cinco valores de `FGM_CASE["phis"]` son la familia inicial, no el
número final de llamas. Se añaden nuevas composiciones hasta alcanzar el
objetivo de defecto o un límite de ejecución.

El ejemplo escribe:

- `fgm_matrix.csv`: una fila por estado `(Z, Z_star, c)`, con temperatura,
  densidad, liberación de calor y las especies seleccionadas.
- `custom_fgm_plots.pdf`: perfiles y mapas construidos explícitamente
  con Matplotlib.

Para H₂ cambia el mecanismo, combustible, pesos de progreso y especies
de las figuras:

~~~python
FGM_CASE.update(
    mechanism="h2o2.yaml",
    fuel="H2",
    progress_species="H2O:1.0,HO2:10.0",
)
PLOT_SPECIES = ("H2", "O2")
~~~

Los títulos y etiquetas de los paneles de especie se actualizan a partir
de `PLOT_SPECIES`.

## Consultar una tabla

~~~sh
python examples/query_fgm.py runs/fgm/MI_EJECUCION/fgm_table.npz
~~~

El script acepta también `--Z`, `--c` y `--species`:

~~~sh
python examples/query_fgm.py runs/fgm/MI_EJECUCION/fgm_table.npz --c 0.25 0.5 0.75 --species H2O
~~~

Si no proporcionas `--Z`, utiliza el punto medio del intervalo almacenado.
Las consultas fuera de la tabla se rechazan.

## Salidas y repetición

Sin una ruta explícita, cada llamada crea una carpeta fechada en `runs/`,
ignorada por Git. Las rutas proporcionadas a `output` deben ser nuevas.
La primera ejecución incluye compilación JIT y puede tardar más.

Las funciones de dibujo de los ejemplos utilizan arrays guardados: puedes
modificar colores, campos y límites sin cambiar el solver.
