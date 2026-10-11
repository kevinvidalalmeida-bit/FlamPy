# Guía de lectura de FlamPy

[Inicio](../README.md) · [Revisión del repositorio y HPC, 8 de octubre de 2026](reviews/repository-hpc-2026-10-08.md)

## Elegir una ruta

| Objetivo | Leer en este orden |
|---|---|
| Instalar y resolver la primera llama | [README](../README.md) → [Compatibilidad](compatibility.md) → [Ejemplos](../examples/README.md) |
| Elegir DRM, FFCM, USC, JetSurF o Aramco | [Mecanismos, instalación local y validación](mechanisms.md) |
| Resolver llamas de difusión y premezcladas con JetSurF, Aramco 2.0 o Aramco 3.0 en minutos | [Jacobiano estructurado, Newton–Krylov y transporte vectorizado](hpc-large-mechanisms.md) → [Datos](assets/diffusion/hpc-large-mechanisms-2026-10-10.json) → [Benchmark](../examples/benchmark_diffusion_mechanisms.py) |
| Entender el núcleo numérico | [Arquitectura](architecture.md) → [API](api.md) → [Archivos de salida](outputs.md) |
| Trabajar con difusión y radiación | [Llamas de difusión](diffusion-flames.md) → [Ejemplo básico](../examples/example_diffusion.py) → [Ejemplo con pérdidas](../examples/example_diffusion_heat_loss.py) |
| Validar CO, NO y NO₂ | [Contaminantes](pollutants.md) → [Script de validación](../examples/validate_diffusion_pollutants.py) |
| Trabajar con quemador y FGM no adiabático | [Quemador](heat-loss.md) → [FGM no adiabático](nonadiabatic-fgm.md) → [Adaptación](adaptive-fgm.md) |
| Entender el modelo reducido | [Transporte reducido y HPC](reduced-burner.md) → [Ejemplo](../examples/example_reduced_burner.py) |
| Optimizar transporte multicomponente y Soret | [Perfil, cambios y papers](hpc-multicomponent.md) → [Muestras y comparación exacta](assets/diffusion/hpc-multicomponent-2026-10-08.json) |
| Comparar HPC entre DRM, FFCM, GRI, USC, JetSurF y Aramco | [Antes/después y decisión CPU/GPU](hpc-mechanisms.md) → [Datos y comprobaciones](assets/chemistry/hpc-mechanisms-2026-10-09.json) |
| Ver el banco de transporte y su coste dentro de Newton | [Fuente final, tiempos y memoria](hpc-transport-bank.md) → [Datos completos](assets/chemistry/hpc-transport-bank-2026-10-09.json) |
| Ver la reutilización exacta entre perturbaciones de temperatura | [Mejora integrada, tiempo por fase y controles](hpc-temperature-reuse.md) → [Datos completos](assets/chemistry/hpc-temperature-reuse-2026-10-09.json) |
| Ver el nuevo reparto de trabajo para JetSurF y Aramco en mallas pequeñas | [Fuente final, tiempos y pruebas](hpc-small-grids.md) → [Datos auditables](assets/chemistry/hpc-small-grids-2026-10-09.json) |
| Ver el perfil actualizado y tiempos reales frente a Cantera | [Revisión por fases, cambios integrados y llamas completas](hpc-reprofile.md) → [Datos de la nueva referencia](assets/chemistry/hpc-reprofile-2026-10-09.json) |
| Ver derivadas y actualizaciones algebraicas del transporte | [Fuente integrada, tiempo de ensamblaje y límites en Newton](hpc-analytic-transport.md) → [Datos y controles](assets/chemistry/hpc-analytic-transport-2026-10-10.json) |
| Ver la mejora del Jacobiano térmico y su coste de preparación | [Derivadas, tiempos por fase y validación](hpc-thermal-jacobian.md) → [Muestras y pruebas](assets/chemistry/hpc-thermal-jacobian-2026-10-10.json) |
| Saber qué propuestas de los agentes se aplicaron o quedaron pendientes | [Estado de todas las estrategias HPC revisadas](hpc-strategy-status.md) |
| Seguir las pruebas de cada estrategia pendiente | [Campaña del 9 de octubre, en curso](hpc-pending-strategies.md) |
| Evaluar velocidad y próximos cambios HPC | [Transporte multicomponente](hpc-multicomponent.md) → [Caras y derivadas por bloques](hpc-next-strategies.md) → [Protocolo de medición](reviews/diffusion-benchmark.md) |
| Entender disposición de arrays y reutilización de buffers | [Localidad de memoria](hpc-memory-layout.md) → [Comparación entre versiones](../examples/benchmark_memory_layout.py) |
| Reproducir figuras | [Resultados](results.md) → [Datos y generación](assets/README.md) |

## Dónde buscar

| Carpeta | Contenido |
|---|---|
| `src/kflame/chemistry/` | Termodinámica, cinética y transporte |
| `src/kflame/flame/` | Balances, Jacobianos, solvers y adaptación espacial |
| `src/kflame/fgm/` | Construcción, consulta y validación de tablas |
| `src/kflame/reference/` | Comparaciones opcionales con Cantera |
| `tests/` | Pruebas automatizadas |
| `examples/` | Ejemplos, auditorías y benchmarks |
| `docs/assets/` | Figuras e informes de mediciones guardadas |
| `docs/reviews/` | Revisiones fechadas, hallazgos y criterios de aceptación |

## Interpretar los resultados

Los documentos temáticos explican el uso actual. Los informes de `reviews/`
describen el estado de una revisión concreta: sus recomendaciones pueden haber
sido implementadas después. La revisión del 8 de octubre distingue el trabajo
confirmado en Git, los cambios locales y las mejoras propuestas.

Una prueba aprobada verifica su caso y sus criterios. Los tiempos en
`assets/diffusion/hpc-*.json` corresponden al código y al entorno registrados
en esos archivos; para atribuir una mejora nueva hay que repetir la medición.
El coste de generar una tabla y el de consultarla son medidas diferentes.
