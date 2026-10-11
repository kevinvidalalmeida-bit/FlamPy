# Reutilización exacta del transporte en columnas de temperatura

**Última mejora del 9 de octubre:** [banco de transporte compartido entre propiedades y Jacobiano](hpc-transport-bank.md), con tiempos de la fuente final y comprobaciones dentro de Newton.

[Guía de lectura](README.md) · [Estado de estrategias](hpc-strategy-status.md) · [Datos completos](assets/chemistry/hpc-temperature-reuse-2026-10-09.json)

Se integra una mejora adicional para JetSurF y Aramco: el Jacobiano evita recalcular los coeficientes de transporte de una misma cara cuando las dos perturbaciones producen exactamente la misma temperatura media. Recalcula el flujo, el gradiente, la densidad y la viscosidad con las expresiones y el orden anteriores. No modifica tolerancias, precisión, química, residual ni política PTC/SER.

La referencia es una nueva copia de 70 archivos del estado aceptado en [mallas pequeñas](hpc-small-grids.md). Sólo cambian `flame/diffusion.py` y `flame/diffusion_kernels.py` respecto a esa copia; los consumidores anteriores se conservan en módulos privados independientes para compararlos con la fuente integrada. Las cifras anteriores mantienen su alcance histórico.

## Antes y después del ciclo usado por Newton

Una evaluación de propiedades + residual con propiedades compartidas + Jacobiano con propiedades compartidas. Se incluyen asignaciones, preparación y construcción de coeficientes. Cuatro hilos Numba y un hilo BLAS; siete pares alternados, después del calentamiento. JIT, igualdad de bytes e I/O quedan fuera del timer. El estado fijo tiene temperatura variable y composiciones distintas por nodo; no es una llama convergida.

| Mecanismo | Nodos | Antes (ms) | Ahora (ms) | Cociente emparejado | Victorias | Criterio de ganancia |
|---|---:|---:|---:|---:|---:|---|
| jetsurf2.yaml | 9 | 303.486 | 281.736 | 1.072× | 7/7 | Sí |
| aramcomech2.yaml | 9 | 772.355 | 723.869 | 1.070× | 7/7 | Sí |
| aramcomech3.yaml | 9 | 1292.381 | 1205.804 | 1.049× | 7/7 | Sí |
| jetsurf2.yaml | 33 | 927.992 | 881.387 | 1.049× | 7/7 | Sí |
| aramcomech2.yaml | 33 | 2636.989 | 2318.445 | 1.176× | 5/7 | Sí |
| aramcomech3.yaml | 33 | 4397.540 | 4065.419 | 1.076× | 6/7 | Sí |

El criterio exige mediana de los siete cocientes ≥1,03× y al menos cinco victorias. Las medianas separadas no definen el cociente emparejado. Los seis mecanismos pequeños se incluyen como controles en el JSON; el nuevo camino no se activa en ellos. No se multiplican estas ganancias por las de campañas previas.

## Dónde queda el tiempo

| Mecanismo, 33 nodos | Propiedades antes (ms) | Jacobiano antes (ms) | Jacobiano ahora (ms) |
|---|---:|---:|---:|
| jetsurf2.yaml | 140.877 | 770.451 | 711.359 |
| aramcomech2.yaml | 357.448 | 2194.708 | 1984.557 |
| aramcomech3.yaml | 567.906 | 3626.115 | 3298.707 |

Estas fases se miden fuera de los pares usados para seleccionar: sirven para localizar el coste y no se suman para reconstruir otro benchmark. El residual que ya recibe las propiedades tiene un coste mucho menor. Siguen pendientes las factorizaciones LU y el cierre térmico que realmente requieren una nueva composición o temperatura.

## Reutilización y memoria

La selección requiere al menos 256 especies, transporte multicomponente congelado en las columnas de composición, cierre Schur y química analítica por bloques. Un filtro barato comprueba si existe alguna igualdad exacta de temperatura perturbada dentro del rango. Si no existe, usa el ensamblador anterior.

Cada rango conserva cuatro valores privados: validez, índice de cara, temperatura perturbada y conductividad. Son 32 bytes de payload adicionales por rango, además del encabezado y referencias del array; no es una medición del RSS. Los coeficientes de difusión y Soret y los datos de viscosidad ya residían en su scratch. No se añade un banco de matrices de tamaño cuadrático.

La primera cara de cada rango se calcula normalmente. Sólo se reutiliza la cara inmediatamente anterior con temperatura exactamente igual y coeficientes finitos. Los rangos concurrentes y las invocaciones A–B–A no comparten este estado. La composición y la presión pertenecen al mismo estado del mismo rango.

Cantera también conserva datos de transporte cuando temperatura y composición no cambian; su código comprueba igualdad exacta e invalida los datos térmicos y de composición por separado. Aquí se aplica esa oportunidad entre las perturbaciones adyacentes del Jacobiano. [Código oficial de Cantera 3.2](https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/MultiTransport.cpp).

## Controles y validación

| Control con temperatura por encima de Tref, 9 nodos | Antes (ms) | Ahora (ms) | Cociente emparejado |
|---|---:|---:|---:|
| jetsurf2.yaml | 308.940 | 320.781 | 0.970× |
| aramcomech2.yaml | 806.933 | 818.421 | 1.004× |
| aramcomech3.yaml | 1249.874 | 1242.197 | 1.010× |

Estos controles se mantienen dentro del intervalo NASA común y no tienen caras con coincidencia de temperatura perturbada; no se les atribuye aceleración por reutilización. Los recuentos del JSON son oportunidades predichas por igualdad de temperatura, no contadores instrumentados del kernel.

JetSurF sin caras reutilizables dio 0,970× en la tanda de una llamada y 1.017× al repetir cinco ciclos por muestra. Ambas mediciones se conservan: el filtro no implica coste cero ni una aceleración de estados sin coincidencias.

Los controles pequeños se repitieron en lotes de 20 ciclos por muestra porque la medición de una sola llamada de 1–2 ms variaba mucho. Los seis cocientes de lote y una repetición de H₂ con 100 ciclos por muestra se conservan aparte en el JSON. El timer incluye todos esos ciclos y expresa su media por llamada; sólo se retiene y compara byte a byte el último resultado de cada lote, además de los gates de calentamiento y A–B–A. No se atribuyen ganancias del nuevo camino a estos controles inactivos.

La fuente integrada pasa **117 pruebas seleccionadas**, con 0 omisiones. Los 15 escenarios de ensamblado comparan bytes, shapes, tipos, strides y finitud de siete propiedades, residual, cinco bloques y la columna del borde. Se comprueba además la secuencia A–B–A. Los nueve mecanismos pasan propiedades contra Cantera: densidad con rtol 2e-12, conductividad, D y Soret con rtol 3e-7 y atol 1e-12, en tres estados cada uno. Es una referencia de propiedades, no un nuevo cociente de tiempo frente a Cantera.

Seis escenarios privados de la versión seleccionada comprueban también independencia de memoria mutable, configuración e inputs inalterados, mallas no uniformes, composición dispersa con trazas, radiación, Soret desactivado, temperatura fijada y cruce NASA a 1000 K. Las pruebas de regresión integradas fuerzan el camino anterior como referencia y comprueban que el nuevo camino se activa y se desactiva donde corresponde.

El calentamiento carga los consumidores C/A y los dos caminos antes del solve: las llamadas posteriores con y sin caras reutilizables no añaden especializaciones. Se comprobó en JetSurF, 9 nodos; no se presenta como un benchmark de arranque sin caché.

## Límites y trabajo restante

La campaña privada v2 conserva sus doce escenarios: once superaron el criterio de ganancia; JetSurF/9 nodos/lineal ganó sólo 1,015×. Esa evidencia está separada de las mediciones de la fuente final revisada. El primer borrador falló al compilar una marca booleana; no produjo resultados grandes ni timings aceptados.

No se ejecutó una nueva campaña de llamas completas de JetSurF/Aramco, ni una nueva campaña GPU, ni contadores L1/L2. La decisión GPU anterior sigue documentada en [HPC por mecanismo](hpc-mechanisms.md). Las quince propuestas originales siguen en [su inventario](hpc-strategy-status.md); esta mejora no las da por cerradas. El siguiente trabajo es perfilar los cierres que aún se recalculan y probar disposiciones de memoria que mantengan el orden de las operaciones.
