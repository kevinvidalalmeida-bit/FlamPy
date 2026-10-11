# Nueva revisión de tiempos y estrategias HPC

**Última mejora del 9 de octubre:** [banco de transporte compartido entre propiedades y Jacobiano](hpc-transport-bank.md), con tiempos de la fuente final y comprobaciones dentro de Newton.

**Nueva continuación del 9 de octubre:** [reutilización exacta de coeficientes entre columnas de temperatura](hpc-temperature-reuse.md), con tiempos de la fuente integrada y controles sin reutilización.

**Continuación del 9 de octubre:** [reparto por rangos en mecanismos grandes y mallas pequeñas](hpc-small-grids.md), con la fuente final validada y sus límites de medición.

[HPC por mecanismo](hpc-mechanisms.md) · [Estado de estrategias](hpc-strategy-status.md) · [Datos completos](assets/chemistry/hpc-reprofile-2026-10-09.json)

Se integran viscosidad con ocho sumas independientes, buffers privados que sobreviven al cambio de bloque químico en sistemas grandes y despacho con menos hilos en caras pequeñas de mezcla. Se mantienen expresiones, orden de cada suma, tolerancias, precisión y residual. La referencia es una copia nueva de 70 archivos, posterior a la campaña de 69 archivos.

## Antes y después del residual y Jacobiano

Nueve nodos, multicomponente con Soret y transporte congelado sólo en columnas de composición del Jacobiano, en ambas versiones. Cuatro hilos Numba y un hilo BLAS; siete pares alternados. El residual y las columnas de temperatura permanecen completos. Este timer llama al residual y luego al Jacobiano público; incluye dos evaluaciones de propiedades. Son estados fijos, no nueve llamas convergidas. H₂ usa este modo como control del perfil; su modo automático de producción conserva el Jacobiano multicomponente exacto.

| Mecanismo | Antes (ms) | Ahora (ms) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|
| H2-O2 | 0.790 | 0.796 | 1.032× | 4/7 |
| DRM19 | 1.641 | 1.584 | 1.049× | 4/7 |
| DRM22 | 1.713 | 1.673 | 1.018× | 4/7 |
| FFCM-1 | 3.143 | 3.471 | 0.980× | 3/7 |
| GRI-Mech 3.0 | 6.832 | 7.020 | 1.056× | 5/7 |
| USC Mech II | 30.974 | 28.423 | 1.111× | 7/7 |
| JetSurF 2.0 | 700.710 | 475.108 | 1.475× | 7/7 |
| AramcoMech 2.0 | 1898.114 | 1239.400 | 1.545× | 7/7 |
| AramcoMech 3.0 | 3151.123 | 1989.926 | 1.550× | 7/7 |

Se considera ganancia consistente una mediana emparejada ≥1,03× y al menos cinco victorias de siete. H₂, DRM19, DRM22 y FFCM-1 no cumplen este criterio en este ensayo; no se les atribuye una mejora uniforme. Las dos medianas separadas y la mediana de cocientes emparejados pueden diferir.

La prueba del ciclo utilizado por Newton —una pasada de propiedades, residual y Jacobiano con propiedades compartidas— está también en el JSON. El prototipo completo ganó 1,56× en JetSurF, 1,64× en Aramco 2 y 1,68× en Aramco 3; esa cifra no se traslada automáticamente a una llama completa.

Las ocho sumas de Wilke avanzan juntas pero cada fila acumula en el orden original. Esto permite solapar cadenas de operaciones independientes y leer bloques de filas contiguas. No se midieron contadores de fallos L1/L2: la evidencia es el tiempo del código completo y sus resultados exactos, no un porcentaje atribuido a una caché concreta.

## Transporte frente a Cantera

32 estados diferentes, dos calentamientos por motor, siete pares alternados. Se devuelven densidad, conductividad, peso molecular, difusión multicomponente y Soret. La asignación y exportación de los arrays están incluidas y los strides coinciden. Numba usa cuatro hilos; Cantera recorre los estados en su API escalar, con BLAS=1 en ambos. La ventaja mezcla algoritmo, batching y configuración de paralelismo.

| Mecanismo | Nativo (ms) | Cantera (ms) | Cantera / nativo |
|---|---:|---:|---:|
| H2-O2 | 0.091 | 0.496 | 5.39× |
| DRM19 | 0.308 | 1.650 | 5.29× |
| DRM22 | 0.277 | 2.110 | 7.36× |
| FFCM-1 | 0.677 | 5.458 | 8.28× |
| GRI-Mech 3.0 | 1.301 | 12.510 | 9.46× |
| USC Mech II | 6.703 | 77.731 | 11.67× |
| JetSurF 2.0 | 132.037 | 1797.490 | 13.60× |
| AramcoMech 2.0 | 326.421 | 4821.519 | 14.72× |
| AramcoMech 3.0 | 524.451 | 7741.518 | 14.79× |

El benchmark consulta D antes de conductividad y Soret: en [MultiTransport.cpp de Cantera 3.2](https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/MultiTransport.cpp), `getMultiDiffCoeffs` invalida la solución térmica. Este orden permite compartir una única solución térmica entre los dos getters energéticos. Las pruebas preliminares con salida intercalada y otro orden no sustentan esta tabla.

## Llamas completas frente a Cantera

Cada motor inicia su propia llama, sin semillas del otro; ratio=3, slope=0,01, curve=0,02 y máximo 2400 nodos. El nativo exige residual escalado <10⁻⁷ y certificado espacial; Cantera emplea su norma de corrección ponderada. Los criterios no son idénticos. Se comparan perfiles en z físico, sin desplazar la llama, con los límites de validación existentes.

La revisión NASA conserva los datos brutos, incluidos los rechazos iniciales: el margen fijo de 128εTmax marcaba temperaturas de Cantera 1,5×10⁻¹⁰ y 1,1×10⁻⁹ K inferiores a 300 K. Se revisan sólo esos efectos de redondeo con `max(128, 8·nodos)·ε·Tmax`, menor de 3×10⁻⁸ K en estos casos. Es un margen de revisión numérica en la frontera, no una extensión física del intervalo NASA. No se redondea ningún campo ni se amplían los límites de error de perfiles. La política y el hash del informe bruto se guardan en los datos.

| Caso | Antes nativo (s) | Ahora nativo (s) | Cantera (s) | Cantera / ahora |
|---|---:|---:|---:|---:|
| H2 | 0.533 | 0.398 | 0.547 | 1.37× |
| H2-MS | 0.826 | 0.890 | 2.261 | 2.54× |
| DRM19-MS | 1.888 | 1.927 | 10.482 | 5.38× |
| DRM22-MS | 2.020 | 2.018 | 15.799 | 7.78× |
| FFCM1-MS | 4.187 | 4.309 | 41.014 | 9.58× |
| GRI | 2.669 | 2.737 | 28.120 | 10.31× |
| GRI-MS | 8.566 | 8.538 | 84.230 | 9.93× |
| USC-diluted-MS | — | — | — | No elegible: incomplete-or-failed / incomplete-or-failed |

Tres pares nativo/Cantera por caso, además del calentamiento. El timer nativo incluye los diagnósticos finales; el de Cantera cubre `flame.solve`. Los bloques antes y después se ejecutaron secuencialmente: ganancias pequeñas entre versiones necesitan cautela. JetSurF y Aramco tienen prueba de transporte y estado fijo; no se les atribuye una ganancia de llama completa sin resolverla. USC usa explícitamente CH4:1,N2:4, dentro del intervalo NASA común de 300–2000 K; es otro caso físico respecto al metano puro.

En las siete configuraciones elegibles se compararon **28 pares de NPZ**: calentamiento y tres muestras por configuración, con los **21 campos exactos** y finitos. USC produjo resultados nativos aceptados, pero su referencia completa de Cantera agotó el límite de 300 s: queda sin comparación completa ni cociente de velocidad. Su calentamiento nativo conserva también los campos entre versiones.

### Coste visible de las APIs

Estas medianas incluyen preparación, carga de kernels desde disco y escritura/validación de salida; cada muestra empieza en un proceso nuevo con caché JIT disponible. Son interfaces con distinto trabajo de diagnóstico e I/O, por lo que no se interpreta su cociente como ganancia aislada del solver. La compilación y los calentamientos previos quedan fuera de esta tabla; no se compara un arranque sin caché. Mejorar el kernel caliente no implica ganar esa primera invocación.

| Caso | API nativa (s) | API de referencia Cantera (s) |
|---|---:|---:|
| H2 | 1.261 | 0.591 |
| H2-MS | 1.732 | 2.305 |
| DRM19-MS | 3.014 | 10.567 |
| DRM22-MS | 3.081 | 15.896 |
| FFCM1-MS | 5.558 | 41.194 |
| GRI | 4.056 | 28.330 |
| GRI-MS | 9.889 | 84.479 |

## PTC/SER y selección por tamaño

En difusión sigue backward Euler. SER ya se descartó porque el residual estacionario inicial podía crecer y colapsar dt. En premezcladas/burner sigue `ptc-ser` con respaldo backward Euler. Son rutas diferentes. La [documentación oficial del solver de Cantera](https://www.cantera.org/3.2/reference/onedim/nonlinear-solver.html) describe la recuperación temporal y el equilibrio entre refrescar el Jacobiano y gastar correcciones.

| Ensayo de política, diagnóstico instrumentado | Pares | Campos exactos | Decisión |
|---|---:|---:|---|
| diffusion-H2-patient-newton, P | 1 | No | Mantener configuración actual; no hay selección por tiempo instrumentado |
| diffusion-H2-steady-age-long, P | 1 | No | Mantener configuración actual; no hay selección por tiempo instrumentado |
| diffusion-H2-transient-age-short, P | 1 | No | Mantener configuración actual; no hay selección por tiempo instrumentado |
| diffusion-CH4-patient-newton, P | 1 | No | Mantener configuración actual; no hay selección por tiempo instrumentado |
| diffusion-CH4-steady-age-long, P | 1 | No | Mantener configuración actual; no hay selección por tiempo instrumentado |
| diffusion-CH4-transient-age-short, P | 1 | No | Mantener configuración actual; no hay selección por tiempo instrumentado |
| diffusion-H2-patient-newton, MS | 1 | No | Mantener configuración actual; no hay selección por tiempo instrumentado |
| diffusion-H2-steady-age-long, MS | 1 | No | Mantener configuración actual; no hay selección por tiempo instrumentado |
| diffusion-H2-transient-age-short, MS | 1 | No | Mantener configuración actual; no hay selección por tiempo instrumentado |

La candidata de buffers generalizada ralentizó H₂ a aproximadamente 0,36× por el coste de convertir el tuple de arrays a objetos Python. La primera integración también penalizaba mecanismos pequeños aunque devolviera `None`: la firma opcional cambiaba el código compilado. Se retiró. La producción conserva íntegra la función original `jacobian_range` y usa otra función compilada, `jacobian_range_workspace`, sólo en mecanismos grandes con Jacobiano congelado. No se habilitan FMA, fastmath, aproximaciones de rango, eliminación de trazas ni nuevas tolerancias.

## Dónde está el coste y qué falta

Perfil inicial de nueve nodos; las categorías se miden por separado y sus medianas no son fracciones aditivas de una misma llamada. El residual usa el transporte completo y la columna de temperatura permanece exacta.

| Mecanismo | Propiedades (ms) | Jacobiano con propiedades compartidas (ms) | Factorización (ms) | Solve lineal (ms) |
|---|---:|---:|---:|---:|
| H2-O2 | 0.136 | 0.511 | 0.079 | 0.008 |
| DRM19 | 0.364 | 0.841 | 0.099 | 0.011 |
| DRM22 | 0.337 | 1.071 | 0.141 | 0.011 |
| FFCM-1 | 0.614 | 1.957 | 0.380 | 0.019 |
| GRI-Mech 3.0 | 1.020 | 4.233 | 0.543 | 0.016 |
| USC Mech II | 3.519 | 23.726 | 3.182 | 0.048 |
| JetSurF 2.0 | 60.510 | 585.153 | 72.418 | 0.993 |
| AramcoMech 2.0 | 164.037 | 1684.496 | 150.911 | 1.522 |
| AramcoMech 3.0 | 253.166 | 2631.574 | 234.036 | 2.849 |

Para 348/493/581 especies el presupuesto químico de 1 MiB impone un nodo por bloque; antes perdía la cara compartida al cambiar de bloque. Ahora cada rango conserva sus dos caras privadas y sus buffers. No se almacena un banco de toda la malla. La siguiente candidata a investigar es un banco acotado residual→Jacobiano de coeficientes de transporte; requiere incluir construcción, copias e invalidación en su tiempo. Un banco sólo de D para Ns581 ocuparía unos 21,6 MB en nueve nodos y 86,4 MB en 33. Sigue pendiente: no se presenta como una mejora aplicada.

Se aprobaron **168 pruebas seleccionadas** sobre esta fuente, incluyendo el nuevo oráculo escalar de viscosidad. Las 853 pruebas completas corresponden a la campaña anterior; no se presentan como una nueva ejecución completa. Las nueve comparaciones de transporte y los residuales/matrices de la fuente integrada conservan los bytes frente a la nueva referencia. Los datos incluyen comparaciones de todos los campos NPZ de los casos completos elegibles.

Esta revisión valida tres cambios adicionales. El inventario original de quince estrategias conserva sus pendientes y referencias históricas; no se declara cerrado por estas nuevas mediciones.

La decisión GPU sigue la evidencia de la campaña anterior: en doce pruebas de LU térmica exacta, la CPU con cuatro u ocho hilos fue más rápida. Las nuevas mejoras afectan el ensamblaje en CPU. No se ha demostrado que portar ahora el solver completo a GPU mejore el tiempo manteniendo el requisito de exactitud.
