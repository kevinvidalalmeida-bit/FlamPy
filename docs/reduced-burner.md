# Transporte FGM con pérdidas hacia un quemador

La rama `feature/nonadiabatic-enthalpy` incorpora un solver estacionario de
quemador plano que transporta **Z local de Bilger, progreso C sin normalizar y
entalpía total h**. Las especies y las fuentes se recuperan de una tabla congelada;
el solver reducido no evalúa química detallada. Se compara, sin desplazar los
perfiles, con el solver nativo de todas las especies y con `Cantera BurnerFlame`.

**Estado de la validación:** de 13 casos independientes, 12 convergen y 10 cumplen
todas las tolerancias de comparación. Los cuatro casos reservados después de
fijar el algoritmo cumplen los límites. Quedan dos casos con errores físicos
excesivos y uno cuyo residuo se estanca; por tanto, el modelo todavía requiere
mejoras para certificar todo el intervalo ensayado. El balance energético por sí
solo no demuestra que las fuentes ni la posición de la llama sean correctas.

Se conservan las **432 llamas** de la selección publicada. Ese número corresponde
a las condiciones de CH₄-aire a 300 K y 101 325 Pa, con transporte promediado por
mezcla y sin Soret; otras condiciones requieren su propia selección y validación.
La tesis mantiene su alcance adiabático: estos cambios pertenecen a la rama del
código y a este estudio separado.

## Ecuaciones, cierre y condiciones de contorno

El flujo másico superficial impuesto es `ṁ = ρu`. Se resuelven

\[
\frac{dF_Z}{dx}=0,\qquad
\frac{dF_C}{dx}=\Omega_C,\qquad
\frac{dF_h}{dx}=0,
\]

con `F_Z = ṁZ + w_Z·J`, `F_C = ṁC + w_C·J` y
`F_h = ṁh − λ dT/dx + Σ h_k J_k` en el límite continuo. Los flujos de especies
`J_k` emplean difusión promediada por mezcla y una corrección que conserva
`ΣJ_k = 0`. La entalpía incluye formación y calor sensible: **no se añade q̇ química
como fuente a su ecuación**, pues eso contaría dos veces la energía química.
`q̇ química` se guarda únicamente como magnitud de comparación.

En el quemador se fija `T = 300 K`, se imponen los flujos alimentados de Z y C, y
se prescribe el flujo de entalpía alimentada más la pérdida conductiva estimada
en la primera cara. C en la superficie puede ser distinto del C de alimentación
por difusión de productos. A la salida se imponen gradientes nulos. El calor
hacia el quemador se define positivo y se estima como `q = λ(T₁−T₀)/(x₁−x₀)`;
es una estimación en malla finita que se debe comprobar por refinamiento.

Se utiliza una malla **no uniforme** que conserva los nodos de la llama de
entrenamiento y subdivide sus intervalos largos. Así se evita crear nodos casi
coincidentes al superponer una malla uniforme. La advección usa un ajuste por
Peclet que tiende al esquema centrado para Peclet pequeño y al upstream para
Peclet grande; la difusión preferencial y el flujo de entalpía de especies se
mantienen explícitos. No se atribuye al sistema acoplado una garantía de
positividad procedente de un esquema escalar.

Estas ecuaciones siguen la estructura de los balances reducidos y del cierre
de entalpía de [van Oijen y de Goey (2000)](https://doi.org/10.1080/00102200008935814),
que también comparan temperatura de salida, separación del quemador y perfiles
de especies. [Gövert et al. (2018)](https://link.springer.com/article/10.1007/s10494-017-9848-4)
utilizan flamelets de quemador obtenidos variando el caudal y muestran mapas de
fuente en progreso y entalpía, además de perfiles de temperatura, CO y OH.
Los mecanismos, cierres de difusión y condiciones de esos trabajos difieren
de los de este ejemplo: nuestras curvas no son una reproducción cuantitativa
de sus figuras.

## Algoritmo y datos tabulados

Newton opera sobre tres coordenadas numéricas acotadas del mallado estructurado.
Estas coordenadas sirven para buscar estados dentro de la familia; **no sustituyen
los controles físicos ni normalizan Z o C**. La opción predeterminada usa una
aproximación B-spline cuadrática local en progreso y pesos bilineales en las otras
direcciones. Es continua en la primera derivada respecto al progreso. Los mismos
pesos convexos se aplican a Y, h y las fuentes, y T se recupera invirtiendo h(T,Y).

Es una **aproximación**, que conserva los extremos pero no interpola exactamente
cada nodo interior. Se ha evaluado por separado mediante las comparaciones a
posteriori de este documento. La consulta física habitual de la tabla y la opción
`interpolation='barycentric'` del solver conservan la interpolación original entre
tetraedros; el mapa de C y h muestra esa consulta original.
Las [propiedades y soporte local de B-splines](https://docs.scipy.org/doc/scipy/tutorial/interpolate/splines_and_polynomials.html)
justifican el soporte compacto. La elección de este cierre suave y su aplicación
al transporte son decisiones de esta implementación, no un algoritmo copiado
de los artículos de FGM.

El Jacobiano tiene tres bloques por fila y se calcula con **nueve colores**,
reutilizando las temperaturas de los nodos que no cambian. Se resuelve como matriz
dispersa; la búsqueda del paso mantiene las coordenadas dentro de la tabla. Una
regularización dispersa actúa cuando falla el paso de Newton. El solver devuelve
un fallo explícito ante estancamiento, falta de convergencia o controles fuera del
dominio; no convierte un perfil visualmente plausible en una solución aceptada.
Las tolerancias y el presupuesto de iteraciones se pueden modificar en la API.

Los perfiles de 30 mm todavía contienen química lenta cerca de su extremo
quemado. Se conservan, sin cambiar un solo bit, sus 181 muestras originales y se
añaden 16 muestras de un **reactor homogéneo adiabático de presión constante**,
iniciado en cada extremo y evaluado con cinética nativa. Se muestrea la trayectoria
química hacia el equilibrio HP, manteniendo elementos y entalpía; se obtienen
197 muestras por llama, 85 104 vértices y ninguna celda plegada. Esta continuación
es una propuesta de cierre posterior a la llama; no equivale a resolver nuevas
llamas espaciales ni a simular apagado frente a una pared.

El reactor utiliza BDF con `rtol=1e−9`, `atol_Y=1e−15` y `atol_T=1e−8 K`. Se
registra la corrección de fracciones negativas exclusivamente de redondeo:
su masa máxima es 1.25×10⁻¹⁷. La desviación máxima de conservación elemental es
6.67×10⁻¹⁴. La detención por progreso no garantiza que todas las especies lentas
estén igualmente próximas al equilibrio: la separación máxima antes del extremo
HP es 0.00274 en fracción másica. Ese cierre y el comportamiento de especies
lentas requieren validación específica; los datos y esta limitación se publican.

## Resultados y criterios de aceptación

Cada simulación reducida parte de un perfil de **entrenamiento con SHA-256
verificada**. Las soluciones detalladas independientes suministran únicamente
las comparaciones y la definición física del caudal; sus campos no inicializan
los controles reducidos. La tabla, las condiciones y los límites se registran
antes de ejecutar cada conjunto. Los nueve casos iniciales se consideran de
desarrollo porque se utilizaron para mejorar el método. Los cuatro casos
reservados se ejecutaron después de fijarlo.

| Magnitud comparada | Límite |
|---|---:|
| Error máximo de temperatura | 20 K |
| Error absoluto máximo entre todas las fracciones másicas | 0.005 |
| Error de fuente dividido por el pico de la referencia | 5 % |
| Error L1 de cada fuente | 5 % |
| Error de integral de cada fuente | 3 % |
| Error relativo del calor hacia el quemador | 2 % |
| Error absoluto de separación δ | 20 µm |

Las fuentes son `ΩC` y `q̇ química`. El error L1 se calcula como
`∫|predicción−referencia| dx / ∫|referencia| dx`; el error integral usa el valor
absoluto de la integral de la diferencia con el mismo denominador. El porcentaje
respecto al pico no es un error relativo local en puntos de fuente casi nula.
δ se define mediante el máximo de `|ΩC|`, con interpolación cuadrática de tres
nodos; esta definición difiere del máximo de consumo de O₂ utilizado en otros
trabajos y se mantiene idéntica entre nuestros tres modelos.

| Caso pendiente, φ / r | Resultado |
|---|---|
| 0.815 / 0.575 | Converge; calor 2.27 %, pico ΩC 5.21 %, L1 ΩC 5.94 %, pico q̇ 5.27 % y L1 q̇ 5.22 % |
| 1.235 / 0.575 | Converge; L1 de q̇ = 5.47 % |
| 1.235 / 0.095 | Se estanca el residuo; se conserva la iteración fallida y no se acepta |

Los cuatro casos reservados (`φ = 0.875, 1.165`, `r = 0.410, 0.135`) cumplen
todos los límites. La referencia detallada nativa y Cantera difieren como máximo
3.65 K y 0.177 % en calor entre los 13 casos. La malla se duplicó dos veces en
tres condiciones: el mayor cambio final de temperatura fue 0.184 K y el del
calor, 0.00149 %. En la condición `φ=0.985, r=0.285`, ampliar de 60 a 90 mm cambia
la temperatura de salida 0.468 K en el FGM, 0.094 K en el nativo y 0.255 K en
Cantera; las tres pruebas cumplen los límites de dominio. Son comprobaciones
de esas condiciones, no una garantía para toda la biblioteca.

En la máquina utilizada, los ocho casos iniciales que convergen tardan
0.29–0.67 s, con mediana 0.47 s; cargar la tabla tarda aproximadamente 2.45 s.
La caché reduce el tiempo mediano de nueve evaluaciones del Jacobiano de
95.6 a 52.8 ms, **1.81×**, con diferencia numérica cero en los residuos medidos.
Son tiempos de una máquina concreta, sin construcción de la tabla ni validación
detallada. Las cuatro comprobaciones reservadas tardan 0.37–0.91 s. No se ha
evaluado escalabilidad a geometrías multidimensionales.

## Figuras y reproducción

![Sección física de la tabla en progreso y entalpía](assets/reduced-burner/01_mapas_C_h.png)

![Perfiles de temperatura, progreso y entalpía](assets/reduced-burner/02_perfiles_T_C_h.png)

![Temperatura de salida y separación del quemador](assets/reduced-burner/03_temperatura_separacion.png)

![Perfiles de CO y OH](assets/reduced-burner/04_CO_OH.png)

![Flujos de calor y balance de entalpía](assets/reduced-burner/05_flujos_balance.png)

![Perfiles de fuentes químicas](assets/reduced-burner/06_fuentes.png)

![Errores, malla y dominio](assets/reduced-burner/07_errores_convergencia.png)

Las curvas no se desplazan para mejorar el acuerdo. La cruz identifica la
iteración sin converger; los recuadros muestran tolerancias excedidas. Se utiliza
Cividis en campos escalares y líneas diferenciadas; no existe un único color
obligatorio para presentar un FGM. El gris identifica consultas fuera de la tabla.

Para redibujar las siete figuras y el PDF a partir de los arrays publicados:

~~~sh
python -m examples.plot_reduced_burner --output output/figures/reduced-burner --pdf output/pdf/Validacion_FGM_perdidas.pdf
~~~

Para reconstruir exactamente la tabla de 432 llamas con su continuación,
sin repetir las simulaciones detalladas:

~~~sh
python -m examples.reconstruct_adaptive_table --output runs/fgm432
python -m examples.extend_reduced_fgm_tail --source runs/fgm432 --output runs/fgm432_extended --reactor-cache docs/assets/reduced-burner/reactor_tail.npz
python -m examples.example_reduced_burner --table runs/fgm432_extended --output runs/reduced_example
~~~

La reconstrucción reproduce exactamente la SHA-256 de la tabla usada:
`52db341e2c195debc80fb83c20eface93f8875e24f67bdfee451ad58a0f45700`.
Al omitir `--reactor-cache` se vuelven a integrar los reactores nativos; eso no
añade flamelets. Los datos incluyen semillas, casos fallidos, planes y huellas.

Para repetir las comparaciones detalladas y los controles espaciales:

~~~sh
python -m examples.validate_reduced_burner --table runs/fgm432_extended --profiles docs/assets/reduced-burner/seeds --output runs/reduced_validation
python -m examples.validate_reduced_burner --table runs/fgm432_extended --profiles docs/assets/reduced-burner/seeds --output runs/reduced_reserved --settings examples/reduced_burner_reserved_settings.json
python -m examples.check_reduced_burner_convergence --table runs/fgm432_extended --validation runs/reduced_validation --profiles docs/assets/reduced-burner/seeds
python -m examples.benchmark_reduced_burner --table runs/fgm432_extended --case runs/reduced_validation/case_0.985000_0.285000 --output runs/reduced_benchmark.json
~~~

La validación detallada requiere el extra `[reference]`; el solver reducido y
la construcción nativa no requieren Cantera. Cambiar tolerancias o condiciones
requiere una carpeta nueva: el ejemplo rechaza cachés con otro plan, tabla,
mecanismo, parámetros físicos o código del solver reducido.

Para cerrar el estudio quedan resolver el estancamiento de la condición rica
con mayor déficit y reducir los errores de los dos casos restantes sin relajar
sus límites. Si se cambia el método, se deben reservar nuevos casos de confirmación.
La extensión a apagado de pared necesita además estudiar la historia térmica:
[Efimov et al. (2020)](https://doi.org/10.1080/13647830.2019.1658901)
muestran que una entalpía adicional por sí sola puede ser insuficiente para CO
en interacción llama-pared. Sus controles de QFM y su geometría difieren de
este quemador plano. Tampoco se incluyen radiación ni conducción dentro del sólido.
