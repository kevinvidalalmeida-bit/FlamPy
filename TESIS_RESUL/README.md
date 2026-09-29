# Resultados de tesis

Este directorio reúne los artefactos finales y los datos necesarios para
reproducirlos localmente.

## Capítulo

`CAPITULO_RESULTADOS/main.tex` es el capítulo de resultados. Sus figuras se
organizan en:

- `llamas_individuales/`: 13 PDF usados por la sección de llamas.
- `FGM/`: 10 PDF usados por la sección FGM, para CH4 y H2.

No hay referencias activas a las carpetas antiguas `revision_figures` ni
`FGM_FIGURAS`.

## Corridas de respaldo

Los datos pesados se conservan localmente en `corridas/` y se excluyen de Git:

- `corridas/llamas_individuales/`: campaña L3, sensibilidad, diagnóstico,
  acondicionamiento y recuperación.
- `corridas/FGM/`: referencia FGM de CH4, comparación CH4, comparación H2 y
  llamas retenidas H2.

Los scripts que antes usaban `runs/...` deben recibir estas rutas mediante sus
opciones `--input`, `--construction`, `--holdouts` o `--output` según el caso.
