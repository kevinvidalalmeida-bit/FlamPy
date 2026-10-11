# Revisión del cierre de entalpía y las pérdidas de calor

**Estado actual.** La causa de los nueve fallos de calor con pérdidas pequeñas
era la convección upwind de primer orden de las referencias y de las llamas de
entrenamiento ([convección de segundo orden](#convección-de-segundo-orden-y-biblioteca-regenerada)).
Con la biblioteca regenerada convergen los 44 casos y pasan 43. T, especies
y fuentes se comparan a igual progreso C; la posición del frente se limita
aparte a 20 µm ([detalle y figuras](reduced-burner.md#validación-con-la-biblioteca-regenerada)).
Lo que sigue documenta la investigación en orden cronológico.

La revisión conserva el contorno conservativo del FGM y añade una comprobación
independiente del flujo de especies en la entrada. El balance global de energía
puede cerrar aunque la composición reconstruida no satisfaga ese contorno.
La reconstrucción física y la nueva selección genérica del progreso se han
contrastado con los 44 casos independientes: 35 pasan y los mismos nueve
siguen fallando. La corrección de los datos de entrenamiento no amplía por
sí sola el rango validado. No se introducen ajustes por combustible,
composición o fracción de caudal.

## Comparación con Cantera

Para el quemador plano, Cantera impone la temperatura de entrada y, para cada
especie, `mdot*Y_feed - J - mdot*Y = 0`. En la salida impone gradientes nulos.
Véanse las [ecuaciones y condiciones de contorno de Cantera 3.2](https://www.cantera.org/3.2/reference/onedim/governing-equations.html).
La energía incluye conducción y transporte de entalpía por difusión de especies;
la [discretización de Cantera](https://www.cantera.org/3.2/reference/onedim/discretization.html)
resuelve temperatura con el término químico correspondiente.

El FGM transporta **entalpía total**, incluyendo formación. Por ello no añade
otra fuente `qdot` a su ecuación de h: hacerlo contaría dos veces la energía
química. Sus condiciones de entrada de Z y C son proyecciones del contorno
de especies y no garantizan las condiciones individuales de todas ellas.
La condición de h prescribe `mdot*h_feed + q_conduction` en la primera cara.

Se distinguen ahora tres comprobaciones:

- `relative_energy_closure_error`: calor conducido frente al déficit de h en
  la salida. Comprueba el balance que resuelve el modelo.
- `relative_inlet_enthalpy_flux_error`: valor absoluto de
  `mdot*h(T0,Y0) + sum(hk(T0)*Jk) - mdot*h_feed`, dividido por la mayor
  magnitud del calor de pared y del déficit de salida, con suelo de 1 W/m².
  Comprueba el contorno reconstruido con entalpías del nodo de entrada.
- `relative_inlet_enthalpy_face_correction`: diferencia entre el flujo
  prescrito y el flujo de h calculado antes de sustituirlo en la primera cara,
  con la misma escala. Incluye la discretización de advección y las entalpías
  de especies evaluadas en la cara; no es idéntica al defecto del nodo.

Los diagnósticos del contorno se incluyen tanto en el solver detallado como
en el reducido y en sus comparaciones. Son evidencia adicional, no una nueva
tolerancia universal elegida a partir de estos casos.

## Hallazgos sobre los perfiles guardados

Se reconstruyó la tabla publicada de 432 llamas con la huella exacta
`c879027d67e697d938c256fe0f078ce0e98d7c5a7a9de833310a95c31068103e`.
Se auditaron los 44 perfiles, conservando los resultados de validación
originales: 35 pasan y nueve fallan, todos ellos en calor hacia el quemador.

| Caso φ/r | Error de calor publicado | Defecto de h en entrada / escala térmica | Corrección de la primera cara / escala térmica |
|---|---:|---:|---:|
| 0.735 / 0.925 | 4.74 % | 9.97 % | 9.87 % |
| 0.745 / 0.935 | 5.53 % | 61.35 % | 60.78 % |
| 0.985 / 0.950 | 3.40 % | 98.63 % | 97.76 % |
| 0.745 / 0.875 | 2.41 % | 0.10 % | 0.11 % |

En esos perfiles, los defectos de entrada de las referencias detalladas son
mucho menores. El último caso también muestra que el defecto de entrada no
explica por sí solo todos los errores de calor. Deben revisarse además la
representación de la variedad, sus fuentes y el transporte. La sensibilidad
espacial previamente publicada ya descarta atribuir todos los fallos a la malla.

Un ensayo diagnóstico retiró la sustitución del flujo de h en tres condiciones,
sin incorporarlo al solver. Convergieron los residuos, pero el error de balance
energético pasó aproximadamente a 1.21 %, 9.05 % y 67.23 % en
0.985/0.285, 0.735/0.925 y 0.985/0.950, respectivamente. Retirar esa condición
no constituye una corrección física general.

La auditoría con el código optimizado mantiene todos los residuos guardados
por debajo de `1e-7`; el máximo es `8.25e-8`. La máxima diferencia al recuperar
T es `7.55e-10 K`. Tres resoluciones completas desde sus semillas de entrenamiento
reproducen un caso válido y dos fallidos, con cambio máximo de T de `1.41e-9 K`.
Esta primera comprobación conservó la tabla original. Más adelante se repiten
las 44 resoluciones con la tabla físicamente reconstruida, como se detalla abajo.

## Corrección numérica y aceleración

El residuo reducido escalaba filas de contorno todavía sin inicializar antes
de sobrescribirlas. Se inicializan ahora explícitamente, evitando avisos de
desbordamiento o fallos cuando se activan excepciones de coma flotante.

Newton conserva ahora los estados iniciales válidos que están exactamente en
un límite de la tabla. Sus diferencias finitas apuntan hacia el interior en
el límite superior. Una dirección que sale de un límite activo se proyecta
antes de calcular el paso global: antes, una sola variable podía bloquear
el avance de todos los nodos. Las búsquedas de paso rechazan y registran
estados termodinámicos inadmisibles. Estas correcciones no cambian la ecuación
física ni convierten los nueve casos fallidos en casos válidos.

La recuperación de T evalúa h y cp juntos con un núcleo Numba que conserva
las referencias NASA-7, los puntos de cambio propios de cada especie y la
aritmética sin `fastmath`. No crea un temporal `(nodos,especies,7)` y deja de
reevaluar los estados que ya convergieron. También se elimina la construcción
de bases SciPy que la consulta compilada del inicializador ya había calculado.
La huella del solver incluye los nuevos archivos termodinámicos y de diagnóstico
para impedir reutilizar resultados de código diferente como si fueran actuales.

Medición intercalada con los mismos 9.720 estados, después de calentar el JIT:
la inversión anterior tarda **276.8 ms** y la nueva **19.15 ms**, una mejora
de **14.45×**. Las temperaturas coinciden en esa prueba y el defecto máximo
de h es `1.98e-9 J/kg`. En otra comparación antes/después, el residuo pasa de
15.53 a 10.26 ms en el inicializador y de 14.74 a 11.27 ms en el cierre tensorial.
Estas mediciones describen esas operaciones en esta máquina; no son una
aceleración certificada del tiempo total de resolución.

Pasan **104 pruebas**. Las nuevas regresiones contrastan h y cp con Cantera
para H₂ y CH₄, incluidos los puntos de cambio NASA, perfiles con strides y
broadcasting. También verifican inversión con temperaturas iniciales alejadas,
distinción entre nodo y cara, ausencia de escalado de memoria sin inicializar,
avance con límites activos y rechazo de datos de entrenamiento que contaminan
el presupuesto de entalpía.

## Calidad de las referencias adiabáticas

Una llama libre aceptada por el solver y por sus indicadores de malla todavía
puede tener un defecto de entalpía de salida significativo respecto de una
pérdida de calor pequeña. La auditoría encuentra hasta **1.52 kJ/kg** de defecto
en las referencias de la tabla publicada: hasta **7.25 %** del menor déficit
de entalpía de su familia. **17 de 24 familias** superan el presupuesto del
2 %. Esto es una contaminación comprobada de la referencia; todavía no se ha
demostrado que explique por sí sola los nueve fallos físicos.

`build_nonadiabatic_table` comprueba ahora
`abs(h_feed-h_adiab_out)/min(h_feed-h_burner_out)` contra `max_energy_error`
para cada familia, con suelo de 1 J/kg y rechazo explícito de pérdidas de
salida no positivas. Guarda `training_enthalpy_quality.json` antes de rechazar
una tabla. El criterio compara magnitudes físicas y se aplica a cualquier
mezcla compatible; no usa ajustes por φ ni por combustible. El auditor lo
recalcula también para tablas anteriores. Es una condición adicional necesaria
para tabular, no una certificación de los errores de interpolación o transporte.

La entalpía local de una llama adiabática no tiene por qué ser constante:
conducción y difusión de especies transportan energía dentro de la llama.
La comprobación anterior usa la salida de gradientes nulos y no fuerza un
perfil de h constante ni modifica artificialmente temperaturas o especies.

Dos ensayos independientes de refinamiento contrastan el defecto de salida
con Cantera. En φ=0.975, el defecto nativo pasa de −1480 a −317 J/kg al reducir
`slope` de 0.04 a 0.005; Cantera pasa de −1433 a −110 J/kg con sus propias
mallas. En φ=0.725 el defecto nativo pasa de 451 a 102 J/kg. Los resultados
de Cantera con tolerancias predeterminadas no disminuyen monótonamente en
esa mezcla: hacen falta comprobaciones de cierre además del indicador de
refinamiento. Al resolver de nuevo las mallas más finas de Cantera con
tolerancias estacionarias `rtol=1e-8, atol=1e-13`, los defectos quedan en
−167 J/kg (φ=0.975) y 62 J/kg (φ=0.725). Los dominios finales de estos ensayos son distintos (30 mm
nativo y 60 mm Cantera); estas cifras no certifican independencia del dominio.

`solve_flame(initial_solution=...)` admite ahora continuar una llama libre
aceptada del mismo tipo, mecanismo y alimentación. La velocidad guardada
solo inicializa su caudal; el solver vuelve a determinar Su. En los dos
ensayos anteriores, cada pasada adicional nativa tarda aproximadamente
**1.0–2.7 s**. Esto facilita refinar referencias sin repetir el arranque.

En cinco perfiles, la química nativa evaluada en los mismos T/Y coincide
con Cantera con errores relativos inferiores a 8e-11; los coeficientes de
transporte de cara coinciden aproximadamente a 1e-13. La inversión de los
controles de los perfiles detallados muestra errores de representación
interior de T inferiores a 1.3 K en los estados cubiertos. Permanecen defectos
de cobertura cerca de algunos contornos y errores en las fuentes interpoladas.
Forzar la proyección entrópica o recalcular las fuentes sobre el cierre no
mejoró sistemáticamente los casos débiles; esos ensayos no se incorporaron.

La última repetición desde las semillas reproduce un caso válido y dos
fallidos sin cambios relevantes de T. Los tiempos totales son 5.47, 3.84 y
9.45 s; de ellos, 1.40, 2.87 y 7.92 s corresponden a la resolución tras preparar
el cierre. La inversión termodinámica intercalada obtiene 13.0× en esa corrida;
los benchmarks anteriores de la misma operación obtuvieron 14–15×. Estas
cifras no incluyen la generación offline de la tabla ni garantizan el mismo
tiempo en otra máquina. La nueva comprobación obliga a refinar referencias,
reconstruir familias coherentes con sus caudales y volver a validar antes de
ampliar el rango de uso a pérdidas pequeñas. La reconstrucción y la nueva
validación se completaron después de estas mediciones.

## Biblioteca reconstruida y criterio de Newton

`examples.refine_library_enthalpy` continúa referencias libres aceptadas,
refina contra el presupuesto de la menor pérdida y vuelve a resolver los
quemadores cuyos caudales cambian al recalcular Su. Trabaja por familias
independientes, conserva intentos y comprueba los perfiles antes de reutilizarlos.
Un contorno válido de un quemador vecino permite cambiar la semilla antes de
seguir refinando una trayectoria que no converge.

La biblioteca resultante tiene **432 perfiles nativos aceptados**, 24 familias
y todos sus caudales coherentes con la nueva Su. El mayor defecto adiabático
relativo a la pérdida más pequeña pasa de **7.25 % a 0.939 %**. Todas las
familias pasan el presupuesto del 2 %. No se desplazan artificialmente h, T
o las composiciones para obtenerlo.

Se corrigió también una aceptación interna de Newton por paso ponderado pequeño
con residuo todavía grande. El candidato debe satisfacer el guardia de residuo
estacionario que ya exigía la aceptación final. Se reutiliza la evaluación del
residuo del candidato; no se añade otra evaluación química. En una continuación
de entrenamiento, el residuo máximo que quedaba en `2.54e5` baja a **14.8** y
el perfil pasa la aceptación en **12.1 s**. Antes, la API acababa rechazándolo
aunque Newton hubiese declarado convergencia interna.

## Progreso genérico y ramas químicas

Al reconstruir las llamas, el progreso anterior deja de aumentar en la dirección
química de dos extremos de entrenamiento. Insertar un punto de equilibrio HP
como si fuese la siguiente rama no corrige esa incompatibilidad.

`select_progress_weights` usa los incrementos de las llamas y las fuentes
completas de sus extremos. Normaliza cada dirección por separado y busca pesos
no negativos comunes con programación lineal. No recibe nombres de combustibles
ni casos de validación. Puede maximizar el margen de monotonía o conservar
la definición anterior con el menor cambio L1 compatible con un margen estricto.
La monotonía muestreada es una condición necesaria: después se comprueban las
celdas conectadas y las trayectorias reales de reactor.

La selección con margen máximo permite continuar **las 432 ramas químicas
nativas**, conservando h y elementos. La tabla final tiene 85.104 vértices,
459.816 tetraedros y ninguna celda plegada o degenerada. El equilibrio HP queda
como referencia; una rama monótona puede terminar antes de alcanzarlo. El
generador de colas ejecuta reactores independientes en procesos separados y
conserva su orden original. Una prueba serial/paralela de cuatro filas obtiene
las mismas especies y auditorías; en ese lote pequeño el arranque paralelo
cuesta más y no supone una aceleración.

La tabla candidata de margen máximo tiene huella
`b3e8b24ffb11cc989d4dd91d24ff26ccf0c6394950b0c509222c854de4a171e6`.
La variante más próxima al progreso anterior tiene huella
`f2cbc5ac40553ca0fc52d644a5affd03a142ec3f182ccc6b558ea6cab19cb4b7`.
Ambas se conservan como ensayos; ninguna certifica por sí sola el transporte.

## Nueva validación física y tiempos

Se resuelven de nuevo los **44 casos** con la tabla de margen máximo y semillas
verificadas de entrenamiento. Se mantienen los caudales físicos, mallas y
límites de comparación guardados. Las referencias nativas y Cantera son
independientes y no suministran controles ni semillas. Sus fuentes de C se
reevalúan con los nuevos pesos para comparar la misma magnitud.

Convergen numéricamente 44; **pasan físicamente 35**, incluidos **los 33 con
r ≤ 0.65**. Los mismos nueve fallan en calor, con errores de **2.58–5.76 %**
frente al límite del 2 %. En 0.985/0.950 la nueva definición agrava además el
error de T: **81.0 K**, frente a 30.0 K publicados. El mayor error del balance
global de energía es solo `2.06e-8`: ese cierre no demuestra que el flujo de
calor o la composición sean correctos.

El ensayo con mínimo cambio del progreso se limita a cinco condiciones:
convergen las cinco y solo pasa físicamente la condición ya válida. Reduce
el error anterior de 81.0 a 66.0 K, pero sigue incumpliendo el límite de T.
Tampoco se incorporan las proyecciones euclídea o entrópica probadas en el
contorno: mejoran algunas métricas y empeoran otras o pierden convergencia.
No hay una corrección física general demostrada para los nueve casos débiles.

En las 44 resoluciones, los tiempos informados por el solver son **0.82–5.87 s**,
con mediana **1.76 s** y suma **94.25 s**. La mediana de la fase posterior a
preparar el cierre es **1.39 s**. La tabla se carga una vez por proceso y se
reutilizan sus coeficientes. Son tiempos en esta máquina, reunidos de dos
invocaciones; excluyen la generación offline, carga externa de la tabla y
comparación de referencias. No son una comparación controlada de aceleración
total frente a los tiempos publicados.

`examples.validate_frozen_reduced_cases` evita resolver de nuevo las referencias
detalladas cuando no cambian. Congela tabla, referencias, límites y código,
verifica mecanismo/alimentación/caudal y guarda tanto fallos como éxitos.
Permite reanudar casos ya calculados y comprueba la huella de sus perfiles.
Una convergencia reducida fallida nunca se cuenta como validación física válida.

## Balances que quedan fuera de Z/C/h

El auditor añade opcionalmente balances de todas las especies y elementos,
usando el mismo flujo de volúmenes finitos en los tres tipos de perfil.
Para cada especie calcula `Rk = diff(Fk) - omega_k*volumen`; informa los
defectos locales, el balance integral respecto de la alimentación y el
flujo de cada elemento calculado con la matriz atómica del mecanismo.

En 0.725/0.625, que pasa la comparación de calor, las proyecciones integrales
Z/C cierran cerca de `4e-16`, pero el mayor defecto local de flujo elemental
es **3.38 %**; con la misma ley de auditoría es **0.0568 %** nativo y
**0.0615 %** Cantera. En 0.985/0.950 es **2.07 %** reducido frente a
**0.0447 %** y **0.0485 %**, respectivamente. Las fuentes tabuladas conservan
átomos a `2.30e-14 kmol/(m³ s)` como máximo: esta discrepancia aparece en los
balances de transporte y reconstrucción.

Se calcula además `sum_k(hk*Rk)` y su norma L1 respecto de la escala de pérdida.
Esta mide la proyección calórica de los residuos de especies omitidos. No es
una fuente que deba añadirse a h, ni una predicción del error de calor, ni una
nueva tolerancia universal. Las referencias detalladas tampoco satisfacen
exactamente esta discretización de auditoría, diferente de su propia ecuación.
Los defectos también aparecen en casos físicamente válidos; la auditoría no
prueba que expliquen por sí solos cada fallo débil. Sí demuestra por qué
conservar solamente las proyecciones Z/C y el flujo prescrito de h deja
balances físicos adicionales sin controlar.

## ¿Falta la tasa de enfriamiento?

La literatura distingue la pérdida acumulada de entalpía de la historia térmica.
En [QFM, Efimov et al. (2020)](https://research.tue.nl/en/publications/qfm-quenching-flamelet-generated-manifold-for-modelling-of-flamew/),
sección 5.2 y figura 17, distintas tasas de apagado producen composiciones y
fuentes diferentes para un mismo h y progreso principal. La coordenada adicional
es un **segundo progreso químico transportado**, construido con familias
transitorias de apagado contra una pared. Sus controles son `(h,C1,C2)`.
Conservar además Z supondría ensayar `(Z,h,C1,C2)`; esta última extensión es una
inferencia para nuestro modelo, no una validación del paper.

Tres controles también permiten representar pérdidas en otras configuraciones:
[Mukundakumar et al. (2021)](https://research.tue.nl/en/publications/a-new-preferential-diffusion-model-applied-to-fgm-simulations-of--2/)
mejoran un quemador isotermo de hidrógeno con `(Z,C,h)` y difusión cruzada;
[Kinuta et al. (2024)](https://arxiv.org/html/2406.09727v1)
emplean `(C,Z,Δh)` con transporte diferencial y flamelets con estiramiento.
[Schneider et al. (2026)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/combustion-modelling-for-the-flamewall-interaction-of-thermodiffusively-unstable-hydrogenair-flames/FA2E22BC06570E065C5A087C44CC9C1C)
comparan bibliotecas de propagación y apagado: la física de entrenamiento y las
derivadas de difusión influyen en el resultado, además del número de controles.
Son estudios de hidrógeno; no certifican nuestra tabla de metano.

El auditor permite `--thermal-rates` y distingue, en régimen estacionario:

| Magnitud | Definición | Unidad |
| --- | --- | --- |
| Tasa material de temperatura | `u*dT/dz` | K/s |
| Tasa material de entalpía total | `u*dh/dz` | J/(kg s) |
| Disipación escalar de h | `λ/(ρ cp)*(dh/dz)^2` | (J/kg)²/s |

La última sigue la convención de QFM, sin factor 2. No equivale a una velocidad
de enfriamiento en K/s ni al calor de pared en W/m². Las tasas se estiman en caras
con las propiedades de transporte ya calculadas; no modifican las ecuaciones,
la tabla ni las tolerancias.

La auditoría de los **44 perfiles estacionarios** queda guardada en
`runs/heat_loss_robustness/cooling_audit.json`. En los tres modelos la temperatura
aumenta en el máximo de liberación de calor de los 44 casos. En **los nueve que
fallan**, las referencias nativas y Cantera no presentan ninguna disminución de
temperatura entre nodos. En todo el conjunto, la mayor caída desde un máximo
anterior es **0.546 K** nativa y **0.532 K** Cantera. La fracción de liberación
positiva de calor situada en caras con `u*dT/dz < 0` es menor que **0.0064 %**
en ambas referencias. Esto caracteriza estos perfiles; no excluye un efecto de
los gradientes de h sobre su composición.

La biblioteca actual ya varía caudal y mezcla en llamas estacionarias con pared
isoterma. Añadir una columna calculada de enfriamiento a esos mismos datos no
aporta por sí sola historias independientes. Para decidir si falta una dimensión,
hay que separar el error de reconstrucción a priori del error de transporte y,
si procede, generar un piloto con historias térmicas independientes. Si a igual
`(Z,C,h)` esas historias conservan diferencias relevantes de especies y fuentes,
se justifica seleccionar un C2 genérico con sus fuentes y flujos consistentes.
La selección debe comprobar unicidad; no basta reutilizar los pesos del paper
ni añadir un término químico a la ecuación de entalpía total.

## Convección de segundo orden y biblioteca regenerada

El modelo reducido resuelto exactamente en sus propias llamas de entrenamiento
también subestimaba q (−2 a −16 % para r ≥ 0,9) aunque estaba convergido en
malla. El origen era la referencia: la convección upwind de primer orden
(la misma que Cantera) sesga Su en +0,2–0,8 % y, como q es unas 20 veces
más sensible que Su cerca de r = 1, el calor al quemador en +10–26 %. En
φ = 0,8, r = 0,95 Cantera da 5987, 5425, 5152 y 4988 W/m² con 336–1804
puntos; su extrapolación de Richardson (orden observado 1,0) es ≈ 4805.

`convection="exponential"` combina la derivada centrada de tres puntos con
upwind según el Péclet de celda (Il'in–Allen–Southwell) y usa el balance de
media celda en el quemador. El calor a la pared es el flujo de entalpía total
de alimentación menos el que sale de esa media celda. En el mismo caso da
4809,7 / 4804,7 / 4804,7 W/m² con 465 / 912 / 1750 nodos, y Su queda
convergida al 0,06 % con el refinamiento de producción.

Se regeneró la biblioteca (24 φ, 17 fracciones, mismo progreso), con
`slope=0.02`, sin pasadas de refinamiento de entalpía: el defecto adiabático
máximo es 1,74 % del presupuesto de la menor pérdida y la tabla final tiene
92 016 vértices y ninguna celda plegada. Las referencias de los 44 casos se
recalcularon con el mismo esquema a dos niveles (incertidumbre de q ≤ 0,27 %,
mediana 0,004 %). El calor comparado es la pérdida conservativa
`mdot*(h_feed-h_out)`.

Convergen los 44 casos y pasan 42: los 33 con r ≤ 0,65 y 7 de los 9 que
fallaban (error de q 0,26–1,85 %). El error con signo va de −2,0 % a
+1,5 % (16 negativos de 44): ya no hay sesgo sistemático. Los dos restantes
fallan solo por la temperatura máxima en el frente (74,7 y 20,1 K), que
corresponde a desplazamientos de 18 y 7 µm, por debajo del criterio de 20 µm
de distancia de anclaje; la temperatura de salida coincide a menos de 1 K.
Las referencias son nativas; Cantera solo dispone de upwind y exige mallas
finas con extrapolación para servir de contraste.

**Revalidación a igual progreso (11 de octubre de 2026).** En los dos casos
restantes, toda la diferencia de T y de especies se explica por el
desplazamiento del frente. A igual C los errores de T son 1,7 y 1,2 K (en lugar
de 74,7 y 20,1 K) y los de especies 0,0004 y 0,0003. El validador compara ahora
T, especies y fuentes (ΩC y q̇) a igual C, mantiene la integral espacial de las
fuentes y sigue publicando la comparación a igual z. Resultado: **43 de 44**.
El único fallo es el calor de 1,275/0,935: 2,002 % frente al 2 %. Refinando la
malla del modelo reducido baja a 1,995 % y 1,988 %. Las fuentes cumplen con
margen: pico ≤ 1,64 %, L1 ≤ 1,80 % e integral ≤ 0,25 %.

Tiempos en esta máquina, con cargas solapadas: biblioteca 465 s con 6
procesos, referencias 500 s, tabla 12 s, retabulación y selección 31 s,
colas 90 s con 6 procesos, validación 22,5 s. En una comparación controlada
de dos familias (36 llamas) la generación pasa de 300 s a 116 s en serie.

```sh
python -m examples.generate_exponential_library --like runs/heat_loss_review/table/metadata.json --output runs/regen_exponential --jobs 6
python -c "from kflame.fgm.nonadiabatic3d import build_nonadiabatic_table as b; b('runs/regen_exponential', progress_points=197)"
python -m examples.retabulate_reduced_progress --source runs/regen_exponential --output runs/regen_progress
python -m examples.select_reduced_progress --source runs/regen_progress --output runs/regen_selected --objective maximum_margin
python -m examples.extend_reduced_fgm_tail --source runs/regen_selected --output runs/regen_table --endpoint-policy monotone_branch --jobs 6
python -m examples.regenerate_native_references --cases docs/assets/reduced-burner-hpc --like runs/heat_loss_review/table/metadata.json --output runs/regen_references --jobs 6
python -m examples.validate_against_native_references --table runs/regen_table --profiles runs/regen_exponential --references runs/regen_references --output runs/regen_validation.json --jobs 6
```

## Reproducción

Los tres primeros comandos reconstruyen datos publicados sin resolver llamas:

```sh
python -m examples.reconstruct_hpc_table --output runs/review_body
python -m examples.retabulate_reduced_progress --source runs/review_body --output runs/review_progress
python -m examples.extend_reduced_fgm_tail --source runs/review_progress --output runs/review_table --reactor-cache docs/assets/reduced-burner-hpc/reactor_tail.npz
python -m examples.audit_burner_enthalpy --table runs/review_table --cases docs/assets/reduced-burner-hpc --output runs/review_audit.json
python -m examples.benchmark_reduced_burner --table runs/review_table --case docs/assets/reduced-burner-hpc/case_0.985000_0.285000 --output runs/review_benchmark.json
python -m pytest -q
```

El auditor y el benchmark obtienen mecanismo, presión, pesos de progreso y
composición de los datos de entrada. El cierre reducido conserva su ámbito
actual: transporte promediado por mezcla sin Soret y tablas NASA-7 compatibles.

Para reconstruir a partir de una biblioteca completa de entrenamiento aceptada
y comparar con referencias independientes ya guardadas:

```sh
python -m examples.refine_library_enthalpy --source runs/training_library --output runs/refined_library --jobs 2
python -m examples.retabulate_reduced_progress --source runs/refined_library --output runs/refined_sources
python -m examples.select_reduced_progress --source runs/refined_sources --output runs/selected_progress --objective maximum_margin
python -m examples.extend_reduced_fgm_tail --source runs/selected_progress --output runs/selected_table --endpoint-policy monotone_branch --jobs 2
python -m examples.validate_frozen_reduced_cases --table runs/selected_table --profiles runs/refined_library --references docs/assets/reduced-burner-hpc --output runs/new_validation --settings examples/reduced_burner_hpc_development_settings.json
python -m examples.audit_burner_enthalpy --table runs/selected_table --cases runs/new_validation --reference-cases docs/assets/reduced-burner-hpc --species-balances --thermal-rates --output runs/new_species_audit.json
```

La reutilización de esas referencias exige el mismo mecanismo, alimentación,
temperatura, presión y transporte. Otra biblioteca física requiere su propia
referencia independiente. Cambiar código, límites o tabla requiere una carpeta
de validación nueva; un ensayo parcial con `--case` puede reanudarse dentro del
mismo plan congelado.
