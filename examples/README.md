# Ejemplos de FlamPy

[Inicio](../README.md) · [API](../docs/api.md) · [Archivos y consulta](../docs/outputs.md)

Desde la raíz del repositorio:

~~~sh
python -m pip install -e ".[plots]"
~~~

| Script | Qué hace |
|---|---|
| [`example.py`](example.py) | Resuelve CH₄–aire, guarda una matriz de perfiles y crea figuras modificables |
| [`example_fgm.py`](example_fgm.py) | Construye una familia o lee una tabla guardada y dibuja los mapas FGM de la tesis |
| [`query_fgm.py`](query_fgm.py) | Carga una tabla y consulta temperatura y composición por lotes |
| [`example_heat_loss.py`](example_heat_loss.py) | Resuelve una familia con pérdidas hacia un quemador y construye un FGM `(c,h)` |
| [`validate_burner.py`](validate_burner.py) | Compara llamas de quemador con Cantera y comprueba una llama retenida |

Para la nueva familia no adiabática, consulta [el modelo y sus referencias](../docs/heat-loss.md).

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

- `fgm_matrix.csv`: una fila por estado `(Z_in, c)`, con temperatura,
  fuente de progreso y dos fracciones másicas. Añade densidad y liberación
  de calor cuando esos campos están presentes en la tabla.
- `custom_fgm_plots.pdf` y `.png`: cuatro mapas con `Z_in` físico horizontal,
  `c` vertical, escala Viridis e isolíneas, como en la tesis.

Para construir una familia de H₂–aire:

~~~sh
python examples/example_fgm.py --fuel H2
~~~

Se seleccionan `h2o2.yaml` y los pesos `H2O:1,HO2:10,OH:-1`.
El intervalo inicial sigue siendo el de `FGM_CASE["phis"]`; edítalo
para estudiar otras composiciones.

## Reproducir los mapas publicados

Los siguientes comandos usan las tablas incluidas y omiten la simulación:

~~~sh
python examples/example_fgm.py --table docs/assets/data/fgm-ch4.npz
python examples/example_fgm.py --table docs/assets/data/fgm-h2.npz --fuel H2
~~~

Reproducen el mapa de CH₄ con 40 flamelets y el de H₂ con 71 flamelets,
respectivamente. Este último cubre `phi = 0.5–5` y utiliza 1001 puntos
de progreso. `--output` permite indicar una carpeta nueva.

El ejemplo llama al [mismo generador](../docs/assets/generate_fgm_figures.py)
que produce las figuras publicadas. Al leer una tabla antigua con coordenadas
Bilger calculadas con corrientes másicas, convierte únicamente las etiquetas
de dibujo a la base molar de la alimentación; conserva el NPZ original.

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
