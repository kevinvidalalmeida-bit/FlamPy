# Instalación y compatibilidad

[Inicio](../README.md) · [API](api.md)

## Entorno

FlamPy declara `Python >= 3.11`. Utiliza un entorno virtual y una
instalación desde el repositorio; las instrucciones no requieren publicar
ni descargar un paquete homónimo de PyPI.

~~~sh
python -m venv .venv
~~~

~~~powershell
# Windows / PowerShell
.\.venv\Scripts\Activate.ps1
~~~

~~~sh
# Linux / macOS
source .venv/bin/activate
~~~

~~~sh
python -m pip install -e ".[plots]"
python -c "import kflame; print(kflame.solve_flame.__name__)"
flampy --help
~~~

Si la política local de PowerShell impide activar el entorno, puedes
utilizar su ejecutable directamente:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e ".[plots]"
.\.venv\Scripts\python.exe -m kflame --help
~~~

## Dependencias declaradas

Las restricciones de instalación proceden de [`pyproject.toml`](../pyproject.toml):

| Dependencia | Restricción |
|---|---|
| NumPy | `>=2.0,<2.5` |
| SciPy | `>=1.14,<1.18` |
| PyYAML | `>=6.0,<7.0` |
| Numba, Python anterior a 3.12 | `>=0.61,<0.66` |
| Numba, Python 3.12 o posterior | `>=0.67,<0.68` |
| Matplotlib, extra `plots` | `>=3.8,<4` |
| Cantera, extra `reference` | `==3.2.0` |

Estos rangos son restricciones del proyecto, no una afirmación de que se
hayan probado todas sus combinaciones. `llvmlite` se instala como
dependencia de Numba.

| Instalación | Incluye |
|---|---|
| `python -m pip install -e .` | Núcleo de química, transporte, llamas y FGM |
| `python -m pip install -e ".[plots]"` | Núcleo y Matplotlib |
| `python -m pip install -e ".[reference]"` | Núcleo, Matplotlib y Cantera |

La primera ejecución puede compilar funciones de Numba. Ese coste debe
separarse de los tiempos de ejecuciones posteriores cuando se comparan
prestaciones. El solver mantenido se ejecuta en CPU.

## Problemas frecuentes

| Mensaje o situación | Comprobación |
|---|---|
| `No module named kflame` | Instala el proyecto con el mismo Python que ejecutará el script |
| `flampy` no se encuentra | Activa el entorno o utiliza `python -m kflame` |
| Falta Matplotlib | Instala el extra `plots` |
| Falta Cantera en una comparación | Instala el extra `reference` |
| La carpeta de salida ya existe | Cambia `output` o deja `output=None` |
| Especies ausentes en H₂ | Cambia `species` y `progress_species` junto con el mecanismo y combustible |
| Error de monotonía del progreso | Revisa los pesos y las trayectorias en el dominio físico elegido |
| La llama no supera la aceptación | Examina `metadata.json`; revisa entradas y presupuestos de malla/tiempo |
| Consulta fuera del intervalo FGM | Utiliza los límites reales de `Z_grid` y `c_grid` |

Cambiar tolerancias, mecanismos, transporte o condiciones de entrada puede
modificar el coste y las necesidades de malla. Las comprobaciones numéricas
deben acompañar a cada nueva configuración.
