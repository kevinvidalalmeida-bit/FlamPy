# FlamPy

FlamPy resuelve llamas libres premezcladas unidimensionales y construye tablas
FGM en coordenadas de fracción de mezcla y progreso. El solver de producción es
nativo de CPU: Cantera solo se emplea en comparaciones de referencia explícitas.

## Instalación

```sh
python -m pip install .
flampy --help
```

Los ejemplos mínimos son [examples/example.py](examples/example.py) y
[examples/example_fgm.py](examples/example_fgm.py). Las salidas de nuevas
ejecuciones se crean fuera del paquete bajo `runs/` y están ignoradas por Git.

## Estructura

```text
src/kflame/                    solver, química, transporte y FGM
examples/                      entradas mínimas editables
docs/                          API, arquitectura y compatibilidad
TESIS_RESUL/CAPITULO_RESULTADOS/ manuscrito y PDF finales de la tesis
TESIS_RESUL/herramientas/       campañas y posprocesado exclusivos de tesis
```

`TESIS_RESUL/corridas/` conserva localmente las corridas que respaldan el
manuscrito; no forma parte de Git por su tamaño. Los experimentos históricos,
salidas temporales, cachés, pruebas de desarrollo y duplicados de código se
retiraron tras quedar incorporados al análisis de tesis.

## Alcance numérico

El solver conserva el orden de operaciones y las configuraciones numéricas de
las campañas: Jacobiano analítico por bloques con transporte congelado durante
la linealización, LU con pivoteo, Newton amortiguado, PTC y rescate Euler
implícito. Los mecanismos GRI-Mech 3.0 y `h2o2.yaml` están incluidos.

La API pública se documenta en [docs/api.md](docs/api.md).

El paquete conserva el módulo `kflame` por compatibilidad; el nombre público y el comando principal son `FlamPy` y `flampy`.
