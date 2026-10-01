# Datos y figuras de la galería

[Resultados y condiciones](../results.md)

| Archivo | Contenido |
|---|---|
| `data/flame-ch4.npz` | `z`, `T`, `u`, `qdot`, cuatro filas de `Y` y sus nombres de especie |
| `data/fgm-ch4.npz` / `data/fgm-h2.npz` | `Z_grid`, `c_grid`, `T`, `phi_grid`, `Su` y `final_accepted` |
| `data/provenance.json` | Condiciones, identificadores de ejecución y SHA-256 de los NPZ |
| `generate_gallery.py` | Generador de las dos figuras PNG a partir de estos archivos |

Carga los NPZ con `numpy.load(..., allow_pickle=False)`. Los campos
mantienen las unidades de las [salidas del solver](../outputs.md).
Se han seleccionado arrays de ejecuciones guardadas, sin cambiar sus
valores. La galería desplaza y recorta el eje de posición para visualizar
el frente, y normaliza el eje de composición para comparar las familias.

Desde la raíz del repositorio:

~~~sh
python docs/assets/generate_gallery.py
~~~

Requiere el extra `plots`. No necesita Cantera ni una nueva simulación.
