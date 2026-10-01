# Arquitectura numérica de FlamPy

[Inicio](../README.md) · [API](api.md) · [Resultados](results.md)

## Organización

| Módulo | Responsabilidad |
|---|---|
| `kflame.api` | Entradas públicas, comprobación de parámetros y escritura de resultados |
| `kflame.chemistry` | Lectura YAML, mezcla, equilibrio adiabático, termoquímica, cinética y transporte |
| `kflame.flame` | Problema de llama libre, residual, Jacobiano, resolución no lineal y control espacial |
| `kflame.fgm` | Familias, progreso, adaptación de tablas, exportación y comprobación |
| `kflame.reference` | Comparaciones opcionales con Cantera |
| `kflame.cli` | Enrutamiento de los comandos `flampy` |

## Resolución de una llama

~~~mermaid
flowchart TD
    A["YAML + estado fresco + transporte"] --> B["Equilibrio HP e inicialización"]
    B --> C["Balances discretos y Jacobiano por bloques"]
    C --> D["Newton amortiguado"]
    D -->|Necesita recuperación| E["PTC–SER o Euler implícito"]
    E --> D
    D --> F["Control del dominio y adaptación de malla"]
    F -->|Cambios espaciales| C
    F --> G["Aceptación final con transporte actualizado"]
    G --> H["Perfiles, metadatos y diagnóstico"]
~~~

1. **Entrada e inicialización.** El lector carga un mecanismo YAML local.
   El equilibrio adiabático a entalpía y presión prescritas proporciona un
   estado quemado para construir los perfiles iniciales y el anclaje térmico.
2. **Balances.** Continuidad, especies y energía se discretizan sobre una
   malla no uniforme. La convección emplea diferencias aguas arriba y la
   difusión se incorpora mediante flujos conservativos en las caras.
3. **Propiedades.** La termoquímica NASA-7, las fuentes químicas y el
   transporte se evalúan con el núcleo nativo. Los parámetros moleculares
   y las integrales de colisión se obtienen de los datos locales.
4. **Linealización.** El Jacobiano aproximado conserva el acoplamiento
   tridiagonal por bloques. Las derivadas de los balances se ensamblan con
   los coeficientes de transporte congelados durante esa linealización.
5. **Álgebra y globalización.** Se utilizan LU con pivoteo y sustituciones
   por bloques compiladas. Newton aplica amortiguamiento y puede reutilizar
   el Jacobiano; cuando necesita recuperación intervienen PTC–SER y Euler
   implícito. En la inicialización puede utilizarse un rescate con
   temperatura prescrita antes de reactivar el balance energético.
6. **Control espacial.** Se comprueban los extremos del dominio y los
   criterios `ratio`, `slope`, `curve` y `prune`. Una modificación del
   dominio o de la malla requiere repetir la resolución.
7. **Aceptación.** El transporte se reevalúa sobre el estado final.
   La finitud, la corrección ponderada y la guarda residual se combinan con
   el resultado del ciclo espacial. Una matriz válida o una pequeña
   corrección aislada no bastan para aceptar una llama.

La aproximación de transporte congelado y la reutilización del Jacobiano
son decisiones distintas: la primera define las derivadas de una
linealización; la segunda permite conservar esa matriz durante varias
correcciones.

## Química y transporte

Los mecanismos incluidos describen CH₄–aire y H₂–aire. El lector nativo
trabaja con una fase de gas ideal y termodinámica NASA-7; el código incluye
tratamiento de reacciones ordinarias, tercer cuerpo y *falloff*.
Un archivo YAML no es compatible solo por utilizar la extensión `.yaml`.

El transporte admite:

| Modelo | Difusión por composición | Soret |
|---|---|---|
| `mixture-averaged` | Coeficiente efectivo por especie y corrección de flujo neto | Cierre térmico promediado por mezcla |
| `multicomponent` | Acoplamiento entre especies | Cierre térmico multicomponente |

Los cierres Soret son distintos. La ruta nativa no utiliza Cantera como
evaluador de propiedades en tiempo de ejecución. Las atribuciones de las
formulaciones adaptadas se conservan en
[los datos químicos](../src/kflame/chemistry/data/README.md).

## Construcción FGM

Cada llama aceptada se identifica por su composición de entrada `Z` y
aporta perfiles a lo largo de una coordenada de progreso:

`βc = Σ aₖ Yₖ`, `c = (βc − βc,u) / (βc,b − βc,u)`.

La coordenada debe tener un salto no degenerado y una evolución admisible.
Ordenar datos no recupera una trayectoria físicamente ambigua.

La continuación reutiliza una solución cercana o un predictor secante en
`log(phi)`. El cambio paramétrico se limita; fuera de la vecindad admitida,
se reconstruye una inicialización física. En correctores de continuación
con transporte promediado por mezcla, un defecto localizado del modelo
lineal puede activar la actualización de bloques del Jacobiano.

Hay tres adaptaciones diferentes:

| Adaptación | Qué cambia |
|---|---|
| Malla espacial | Nodos y resolución de una llama individual |
| Eje de progreso `c` | Distribución de estados ya resueltos dentro de la tabla |
| Composición de la familia | Nuevas llamas detalladas que añaden filas a la tabla |

En `generate_fgm(adaptive_phi=True)`, el defecto por exclusión de filas
selecciona intervalos. Se resuelven nuevas composiciones en puntos medios
de `log(phi)` y se reconstruye la tabla hasta alcanzar el umbral o un
límite de ejecución. Las filas nuevas proceden de llamas resueltas, no de
interpolación.

## Referencias, cachés y ejecución

Los comandos nativos se importan de forma diferida. Los comandos `compare`
y `reference-fgm` cargan Cantera explícitamente.

La API pública FGM utiliza un proceso y desactiva la caché persistente de
semillas. El generador de línea de comandos ofrece políticas adicionales
de caché y paralelismo; su clave de semillas incluye el contenido del
mecanismo y las condiciones físicas y numéricas.

Los ajustes moleculares inmutables pueden reutilizarse entre llamas o
mallas. Esa reutilización no congela las propiedades dependientes del
estado durante la aceptación final.

## Alcance físico

La formulación actual considera una llama adiabática, plana, estacionaria,
unidimensional y de presión constante. No incorpora pérdidas térmicas hacia
paredes, radiación, química superficial ni un modelo de combustión
turbulenta. Las tablas `(Z, c)` mostradas fijan la presión y la temperatura
de entrada.

La comprobación numérica del sistema discreto, la sensibilidad espacial y
la concordancia con Cantera se distinguen de la validación experimental
de la química y los modelos de transporte.
