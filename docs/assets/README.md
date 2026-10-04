# Datos y figuras de la galería

[Resultados y condiciones](../results.md)

| Archivo | Contenido |
|---|---|
| `data/flame-ch4.npz` | `z`, `T`, `u`, `qdot`, cuatro filas de `Y` y sus nombres de especie |
| `data/fgm-ch4.npz` / `data/fgm-h2.npz` | Ejes físicos `Z_grid`, `c_grid`; `T`, dos especies de `Y`, `omega_c`, `phi_grid`, `Su` y aceptación |
| `data/fgm-*-fidelity.npz` | Perfiles retenidos, predicciones, errores y coordenadas para la prueba independiente |
| `data/fgm-*-family.json` | Tiempo por llama y tipo de predictor, en el orden real de resolución de ocho construcciones |
| `data/provenance.json` | Condiciones, identificadores de ejecución y SHA-256 de los archivos de datos y sus fuentes |
| `generate_fgm_figures.py` | Código exacto de las seis figuras FGM, en PNG y PDF |
| `generate_gallery.py` | Regenera la llama individual y las seis figuras FGM |

Carga los NPZ con `numpy.load(..., allow_pickle=False)`. Los campos
mantienen las unidades de las [salidas del solver](../outputs.md).
Se han seleccionado arrays de ejecuciones guardadas, sin cambiar sus
valores. La galería desplaza y recorta el eje de posición para visualizar
el frente. Los mapas FGM presentan el Bilger de entrada sin normalizar
su intervalo, con `Z_in` horizontal y progreso `c` vertical.

Desde la raíz del repositorio:

~~~sh
python docs/assets/generate_fgm_figures.py
python docs/assets/generate_gallery.py
~~~

Requiere el extra `plots`. No necesita Cantera ni una nueva simulación.
El generador comprueba los SHA-256 antes de dibujar. `--output CARPETA`
permite escribir las figuras FGM en otra ubicación. Las fracciones másicas
seleccionadas y los tiempos se conservan exactamente; estos archivos son
entradas de dibujo, con dos especies por mapa, y no tablas completas de CFD.
