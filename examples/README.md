# Ejemplos de FlamPy

[Inicio](../README.md) · [Guía de lectura](../docs/README.md) · [API](../docs/api.md) · [Archivos y consulta](../docs/outputs.md)

Desde la raíz del repositorio:

~~~sh
python -m pip install -e ".[plots]"
~~~

Los ejemplos que comparan con Cantera requieren el extra `reference`:

~~~sh
python -m pip install -e ".[reference]"
~~~

| Script | Qué hace |
|---|---|
| [`example.py`](example.py) | Resuelve CH₄–aire, guarda una matriz de perfiles y crea figuras modificables |
| [`example_fgm.py`](example_fgm.py) | Construye una familia o lee una tabla guardada y dibuja los mapas FGM de la tesis |
| [`query_fgm.py`](query_fgm.py) | Carga una tabla y consulta temperatura y composición por lotes |
| [`example_heat_loss.py`](example_heat_loss.py) | Resuelve una familia con pérdidas hacia un quemador y construye un FGM `(c,h)` |
| [`validate_burner.py`](validate_burner.py) | Compara llamas de quemador con Cantera y comprueba una llama retenida |
| [`example_diffusion.py`](example_diffusion.py) | Resuelve contraflujo y construye un FGM `(Z,C)`; la comparación opcional con `--reference` requiere el extra `reference` |
| [`example_diffusion_heat_loss.py`](example_diffusion_heat_loss.py) | Genera una familia radiativa `(Z,C,h)` y valida un flamelet reservado |
| [`validate_diffusion_pollutants.py`](validate_diffusion_pollutants.py) | Resuelve y audita CO, NO y NO₂ frente a Cantera y FGM; requiere `reference` |
| [`benchmark_diffusion_solver.py`](benchmark_diffusion_solver.py) | Mide resolución nativa/Cantera con repeticiones; requiere `reference` y una carpeta nueva indicada con `--output` |
| [`benchmark_diffusion_mechanisms.py`](benchmark_diffusion_mechanisms.py) | Llamas de difusión completas para cada mecanismo instalado, camino nuevo frente al anterior (`--modes new legacy split4`), un proceso por caso; requiere `--output` nuevo |
| [`benchmark_memory_layout.py`](benchmark_memory_layout.py) | Compara perfiles y tiempos entre dos árboles de código; requiere `--baseline-src` y una carpeta nueva con `--output` |
| [`example_mechanisms.py`](example_mechanisms.py) | Resuelve la misma llama premezclada (o de contraflujo con `--flame diffusion`) con cada mecanismo instalado y dibuja perfiles, velocidad de llama o temperatura máxima y tiempo del solver; `--plot-only` redibuja la figura publicada |
| [`validate_mechanisms.py`](validate_mechanisms.py) | Comprueba el catálogo instalado, transporte nativo, cinética fusionada y entrada analítica contra Cantera; requiere `reference` y `--output` |

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

Reproducen el mapa de CH₄ con 40 flamelets y el de H₂ con 86 flamelets,
respectivamente. Este último cubre `phi = 0.5–5` y utiliza 2001 puntos
de progreso. `--output` permite indicar una carpeta nueva.
Las condiciones se conservan en la [procedencia de los datos](../docs/assets/data/provenance.json).

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

Para generar y validar un FGM no adiabático `(Z,C,h)` de CH₄-aire:

~~~sh
python examples/example_nonadiabatic_3d.py --output runs/nonadiabatic
python examples/validate_nonadiabatic_3d.py runs/nonadiabatic --output runs/validation3d --fine-native --mesh-check
python examples/plot_nonadiabatic_3d.py runs/nonadiabatic --validation runs/validation3d --output output/figures/nonadiabatic3d
~~~

Sin `--validation`, `plot_nonadiabatic_3d.py` dibuja la familia y los cortes
de la tabla (figuras 1 y 2) de una tabla recién generada; las figuras 3 y 4
necesitan la salida de `validate_nonadiabatic_3d.py`. Una prueba rápida de toda
la cadena, con 12 llamas en menos de un minuto:

~~~sh
python examples/example_nonadiabatic_3d.py --output runs/na3d_small --phis 0.9 1.0 1.1 --mass-flux-fractions 0.45 0.2 0.1
python examples/plot_nonadiabatic_3d.py runs/na3d_small --output output/figures/na3d_small
~~~

Para dibujar los datos publicados sin simular:

~~~sh
python examples/plot_nonadiabatic_3d.py docs/assets/nonadiabatic3d --output output/figures/nonadiabatic3d
python examples/plot_source_audit.py --output output/figures/source_accuracy --pdf output/pdf/Revision_fuentes_FGM.pdf
python examples/audit_nonadiabatic_sources.py --output runs/source_audit
~~~

El [modelo y las métricas](../docs/nonadiabatic-fgm.md) distinguen validación
de la tabulación, comparación numérica independiente y cobertura del dominio.
El ejemplo genera por defecto 450 llamas; el análisis de fuentes informa
errores respecto al pico, errores integrados y balances de progreso.
`check_burner_accuracy.py` comprueba tolerancias y malla en llamas aceptadas,
y `check_training_balance.py` revisa el balance de todos los perfiles de quemador.

Sin una ruta explícita, cada llamada crea una carpeta fechada en `runs/`,
ignorada por Git. Las rutas proporcionadas a `output` deben ser nuevas.
La primera ejecución incluye compilación JIT y puede tardar más.

Las funciones de dibujo de los ejemplos utilizan arrays guardados: puedes
modificar colores, campos y límites sin cambiar el solver.

## Transporte y HPC por mecanismo

`benchmark_mechanisms.py` mide transporte multicomponente y Soret de los
mecanismos instalados. Separa carga/JIT de las muestras en caliente y verifica
también tasas químicas contra Cantera. Intersecta el intervalo solicitado con
el intervalo común NASA de cada archivo; la salida debe ser nueva.

```sh
python examples/benchmark_mechanisms.py --faces 32 --threads 4 --repeats 7 --validate-cantera --output runs/mechanism_timings.json
```

[Antes/después, alcance y decisión CPU/GPU](../docs/hpc-mechanisms.md).

## Tolerancias y refinamiento no adiabático

Edita `fgm_tolerances.json` para elegir errores por magnitud y
`adaptive_fgm_settings.json` para candidatos y condiciones físicas.
`adaptive_nonadiabatic_3d.py` genera la familia, `benchmark_fgm_lookup.py`
compara tiempos con el commit anterior y `reconstruct_adaptive_table.py`
reproduce la selección publicada sin repetir simulaciones.
`plot_adaptive_fgm.py` redibuja las tres figuras desde los datos guardados.
[Unidades, criterios y resultado de validación](../docs/adaptive-fgm.md).

`audit_burner_enthalpy.py` compara el balance global, los defectos de entrada
y el presupuesto de entalpía de las referencias adiabáticas de perfiles
guardados, sin resolver nuevas llamas. `benchmark_reduced_burner.py`
incluye una comparación intercalada de la inversión termodinámica.
[Hallazgos y reproducción](../docs/heat-loss-review.md).

`example_reduced_burner.py` resuelve un caso publicado con el modelo reducido
y dibuja T, C y h frente a la llama detallada publicada. Con la biblioteca
regenerada se le indica su biblioteca de entrenamiento:
`--table runs/regen_table --profiles runs/regen_exponential`.

`validate_against_native_references.py` valida el modelo reducido sobre la
biblioteca regenerada: T, especies y fuentes a igual progreso, calor y
distancia del frente. `plot_reduced_validation.py` dibuja sus figuras desde el
JSON y los perfiles guardados, también los publicados en
`docs/assets/reduced-burner-regen`
([resultados](../docs/reduced-burner.md#validación-con-la-biblioteca-regenerada)).

`refine_library_enthalpy.py` continúa referencias y reconstruye sus caudales
contra el presupuesto de pérdida. `select_reduced_progress.py` selecciona
pesos comunes usando solo entrenamiento y comprueba después la geometría.
`extend_reduced_fgm_tail.py --jobs 2 --endpoint-policy monotone_branch`
continúa ramas químicas nativas. `validate_frozen_reduced_cases.py` compara una
tabla reconstruida con referencias independientes guardadas, sin volver a
resolverlas; congela entradas y código y permite reanudar casos calculados.
`audit_burner_enthalpy.py --species-balances` añade todos los balances de
especies y elementos. `--thermal-rates` distingue las tasas materiales de
temperatura y entalpía de la disipación escalar de h, reutilizando el transporte
de los perfiles guardados. Estos controles adicionales no amplían por sí solos
el rango físico validado.
