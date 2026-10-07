# Resultado integrado de las cuatro revisiones de difusión

7 de octubre de 2026. Checkpoint previo solicitado: `70708b4`. La optimización
térmica anterior está en `94baec9`. Se trabajó con **cuatro agentes independientes**;
el agente principal integró hallazgos, correcciones y validación.

| Revisor | Alcance | Informe |
|---|---|---|
| Beauvoir | Formulación discreta, conservación, entalpía y estancamiento | [Matemática](diffusion-mathematics.md) |
| Herschel | Termoquímica, derivadas, mecanismos y memoria | [Química](diffusion-chemistry.md) |
| Russell | Difusión, viscosidad, multicomponente y Soret | [Transporte](diffusion-transport.md) |
| Descartes | Newton, continuación, inicialización y álgebra lineal | [Solver](diffusion-solver.md) |

Los informes individuales auditan el checkpoint y conservan sus reproducciones;
sus líneas y cifras no describen necesariamente el código posterior corregido.
Esta síntesis distingue lo integrado de las propuestas aún pendientes.

## Dónde sigue el coste

El arranque frío concentra el trabajo nativo. La nueva traza previa a las
correcciones registra 273 de 288 Jacobianos CH4 en 31 nodos, antes del primer
refinamiento. Son 87 llamadas a Newton en esa malla, incluidos 70 intentos de
paso temporal a lo largo de la resolución completa. H2 registra 130 de 143
Jacobianos en la malla inicial. Un Jacobiano de malla fina cuesta más: estos
porcentajes de conteo no son porcentajes de tiempo.

**CH4 no pierde porque Cantera construya muchos menos Jacobianos:** en la
misma traza independiente Cantera registra 285, frente a 288 nativos, pero
distribuidos entre otras mallas y fases. Cantera inicia en seis nodos; el
nativo, en 31. H2 sí registra 63 Jacobianos Cantera frente a 143 nativos. Hay
que abordar el coste por evaluación y la trayectoria, no solo los totales.

Tras el banco térmico anterior, la química ordinaria + agrupada ocupa 10.9%
del perfil CH4, 8.2% H2 y 2.6% H2 multicomponente/Soret. Una derivada química
analítica podría mejorar ese componente, pero por sí sola tiene margen limitado.
En el perfil CH4, viscosidad ocupa 8.87 s inclusivos sobre 44.98 s; en H2-MS,
transporte multicomponente ocupa 3.52 s sobre 5.65 s. Estos perfiles instrumentados
son antecedentes, no deben sumarse con sus categorías padre ni mezclarse con
los nuevos tiempos sin instrumentación.

## Correcciones integradas

| Hallazgo | Cambio | Verificación |
|---|---|---|
| Difusión alternativa y Soret usan un peso molecular incorrecto cuando suma(Y)≠1 | Transmitir `1/sum(Y/W)` del estado; mantenerlo también en la ruta Soret fusionada y en evaluadores de punto/cara | Estados normalizados, escala uniforme y perturbación individual; dos presiones y tres temperaturas frente a Cantera; equivalencia de rutas |
| El diagnóstico entálpico no respeta un cambio del cero de entalpía aun con continuidad satisfecha | Reconstruir caudal de cara con la continuidad trapezoidal y usar h interpolada de cara; normalizar por `mref*cpref*Tref` | Mallas uniforme/irregular, h constante y variable, varios offsets; defectos absolutos compatibles dentro del redondeo |
| Newton rechaza una raíz alcanzada en la última iteración permitida | Comprobar el residual completo del candidato antes del retorno por límite | Problemas lineales estacionario y temporal que convergen en el último paso |
| El lote térmico siguiente convive con el anterior y con copias temporales | Liberar banco/copia antes de asignar el siguiente; mantener fallback por presupuesto estimado | Referencias débiles verifican liberación; Jacobiano agrupado frente al fallback |
| YAML sin unidades usa defaults distintos de Cantera; tiempo y eficiencias predeterminadas se ignoran | Defaults m/kmol/J, conversión de tiempo, `default-efficiency`, inferencia de M, omisión del cuarto término Troe con T2=0 | Reacciones de una sola etapa comparadas con Cantera a distintas T/P y conservación elemental |
| Opciones químicas no soportadas pueden cambiar silenciosamente la física | Rechazar órdenes explícitos, SRI/Tsang, tipos no admitidos, unidades anidadas y presión de referencia distinta de 101325 Pa | Rechazo explícito con `ValueError`; no se promete soporte de todo YAML Cantera |
| NASA7 de una región no se carga; comprobación del modelo depende de assert | Admitir una o dos regiones validadas y rechazar modelos no admitidos mediante excepción explícita | cp y entalpías de una región frente a Cantera |

Además se trasladó la reutilización exacta del residual aceptado de las
premezcladas y se usa la corrección Newton original para la norma de amortiguación,
evitando otra sustitución lineal. Se calcula la norma **antes** de modificar
direcciones por especies traza. No se congelan derivadas T/Y ni se cambian
calendario temporal, edad del Jacobiano o umbral final estacionario `1e-7`.

El presupuesto de 96 MiB sigue siendo una estimación del lote principal;
no es una cota RSS del proceso. Aún faltan partición adaptativa del lote y
contabilidad conjunta de buffers de hilos, transporte y matriz.

La recuperación de excepciones numéricas durante un ensayo Newton queda
pendiente de definir un protocolo de errores recuperables. Capturar todo
`ValueError` ocultaría posibles errores de programación; la reproducción del
revisor es sintética y no prueba que cause los tiempos de las llamas base.

## Entalpía: significado de la corrección

Con `s=2*rho*V`, la reconstrucción de masa coherente con continuidad es

```text
m_face = (m_left+m_right)/2 + dz*(s_right-s_left)/8
F_h_face = m_face*h_face - lambda*grad(T) + sum(h_k_face*J_k)
defecto = F_h_ultima_cara - F_h_primera_cara + integral(s*h dz)
```

Si h se desplaza por una constante y continuidad/sumas de masa se satisfacen,
el defecto absoluto permanece invariante hasta redondeo. La reconstrucción
anterior no tenía esa propiedad. La nueva escala relativa es independiente
del cero de entalpía: **los porcentajes antiguos y nuevos no son directamente
comparables**. La evidencia conserva la reconstrucción anterior y la corregida
sobre los mismos perfiles guardados para separar este efecto del cambio del
solver. No se sobrescriben los resultados históricos.

| Perfil guardado, sin volver a resolver | Defecto anterior | Reconstrucción corregida |
|---|---:|---:|
| CH4 | 556.530 W/m² | 581.241 W/m² |
| H2 | 923.806 W/m² | 923.792 W/m² |
| H2 multicomponente/Soret | 1217.355 W/m² | 1217.340 W/m² |

La corrección casi no altera el defecto absoluto de estos perfiles H2: **no
explica su defecto energético previo ni lo elimina**. Su mérito es reparar
la compatibilidad de la reconstrucción, demostrada mediante el caso fabricado;
la conservación espacial sigue siendo un trabajo distinto.

Corregir el diagnóstico no convierte la discretización de temperatura en un
balance entálpico integral exacto. El defecto remanente puede ser error espacial;
no debe presentarse automáticamente como pérdida física de calor o QGM. La
exportación radial ya forma parte del balance de contraflujo.

## Estrategias genéricas que conviene ensayar, en orden

1. **Inicialización y continuación con modelo persistente.** Medir causas de
   refresco, especie limitante, alpha y residual estacionario durante pasos
   temporales. Probar estimaciones de flujo coherentes, bootstrap espacial y
   calendario según dificultad observada. Conservar la malla explícita del usuario.
   Una tolerancia temporal diferente es una política nueva y requiere validación
   separada de la puerta estacionaria final.
2. **Reutilización exacta y fusión del transporte.** Una evaluación por cara
   debe compartir X, Wmix, ajustes térmicos, binarios y Wilke para D/lambda/mu
   y Soret opcional. Entre columnas Y, reutilizar solo intermediarios que dependen
   de T; recalcular todas las reducciones de composición y factores multicomponentes
   cuando cambie Y. Actualizar solo caras adyacentes a nodos perturbados y acotar
   memoria por bloques. No repetir como prioridad el congelado completo que
   empeoró el ensayo CH4 anterior.
3. **Derivadas químicas y elección por mecanismo.** Ensayar contribución nodal
   analítica a presión fija con todos los terceros cuerpos/falloff y energía,
   comprobando capacidad, ceros, trazas y ramas. Elegir evaluador por coste
   compilado observado, ns/nr, tipos admitidos, soporte de reacciones y memoria.
   La rigidez depende del estado; no se deduce solo del tamaño del mecanismo.
4. **Solver lineal con borde.** Prototipar el bloque local A y Lambda global
   mediante Schur; comparar correcciones y error hacia atrás antes de cambiar
   LU. Ensayar banda/bloques con retorno a LU dispersa ante pivotes o Schur
   problemáticos. La banda CH4 fina puede requerir aproximadamente 189 MiB;
   no se puede truncar el borde global para ahorrar almacenamiento.
5. **Precisión espacial y conservación.** Comparar una energía conservativa
   en entalpía total o gradiente h independiente del donante en estancamiento.
   No añadir otra fuente de reacción a la ecuación conservativa de h total.
   Evaluar error, robustez y coste sobre mallas progresivas antes de promover
   un esquema distinto. Central puro no garantiza segundo orden en la malla
   irregular actual ni positividad. Krylov queda después de las intervenciones
   anteriores, condicionado por coste y tamaño reales.

La actualización temporal correcta del modelo persistente necesita
`Jtrans=Jsteady+[B(x)+G(x,xold)]/dt`, con
`G[:,k]=(dB/dx[k])*(x-xold)`. B contiene rho y cp. Copiar la actualización
diagonal constante de premezcladas omitiría G; guardar una LU con otro dt
sería una aproximación adicional, no reutilización exacta.

| Descriptor del caso | Primer ensayo recomendado | Condición antes de activarlo |
|---|---|---|
| Pocas especies y lotes pequeños | Reducir llamadas, bootstrap y ejecución serial frente a paralela | Menor tiempo hasta aceptación, no solo por kernel |
| Muchas especies/reacciones con química aún costosa | Analítico nodal o lotes acotados según capacidades | Derivadas correctas y coste total menor |
| Multicomponente/Soret domina el perfil | Fusión, localidad y reutilización térmica de cara | Matrices/reducciones de composición actualizadas y memoria acotada |
| Muchas etapas temporales/refrescos | Persistencia y criterio temporal por dificultad | Actualización completa, invalidación y recuperación verificadas |
| Matriz grande y LU/memoria dominante | A local + borde Schur, luego Krylov precondicionado | Error hacia atrás, estabilidad y fallback medidos |

Los mecanismos de referencia (10 especies/29 reacciones y 53/325) sirven para
ensayar rutas distintas. No se añaden ramas por nombre de combustible ni se
cambia el cierre físico automáticamente para conseguir un tiempo menor.

## Validación y medición

La suite final pasa **184 pruebas**, incluidas las nuevas regresiones. Cantera
emite el aviso esperado para el caso sintético T2=0 de la prueba de admisión.
El benchmark ahora admite repeticiones y conserva entorno, tolerancias,
estadísticas Cantera y trazas por llamada Newton. Véase
[comparación reproducible](diffusion-benchmark.md).

Las nuevas mediciones secuenciales sin instrumentación, tres muestras por caso,
sus hashes, límites físicos y balances se conservan en
[four-agent-review.json](../assets/diffusion/four-agent-review.json).

| Caso | Nativo, mediana (mín–máx) | Cantera, mediana (mín–máx) | Mediana de cocientes por pareja |
|---|---:|---:|---:|
| CH4/aire | 43.06 s (41.74–48.30) | 36.66 s (36.49–41.67) | 1.14 |
| H2/aire | 1.79 s (1.73–1.82) | 0.82 s (0.78–0.84) | 2.17 |
| H2/aire, multicomponente + Soret | 5.81 s (5.53–6.04) | 3.38 s (3.36–3.54) | 1.71 |

**Pasan las nueve comparaciones físicas y el residual completo final `1e-7`.**
La mediana de cocientes de parejas no es necesariamente el cociente de las
dos medianas, especialmente con la variación CH4 observada. No se presentan
estas tres muestras como garantía estadística. Tampoco se atribuye una mejora
porcentual frente a la muestra individual anterior: falta una repetición
equivalente del código anterior en el mismo estado del equipo. Cantera sigue
siendo más rápido en estas mediciones. Las estrategias de la sección anterior
siguen siendo propuestas sin una aceleración medida atribuible a ellas.
