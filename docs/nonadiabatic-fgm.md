# FGM con pérdidas de calor: composición, progreso y entalpía

[Inicio](../README.md) · [Modelo de quemador](heat-loss.md) · [API](api.md)

Esta rama incorpora una biblioteca nativa de llamas planas estabilizadas sobre
un quemador isotérmico. La tabla se consulta mediante `(Z,C,h)`; modificar la
entalpía cambia también la composición y la fuente química. Las pérdidas se
obtienen resolviendo las llamas con conducción hacia la superficie.

## Fundamento y decisiones

La generación de niveles de entalpía mediante llamas de quemador con distintos
caudales sigue la estrategia de [Gövert et al. (2015)](https://doi.org/10.1016/j.apenergy.2015.06.031)
y [Gövert et al. (2018), sección 2.3.3](https://link.springer.com/article/10.1007/s10494-017-9848-4).
La reducción del caudal aumenta la pérdida de entalpía por kg, aunque el flujo
absoluto de calor hacia el quemador puede disminuir.

Los controles y las unidades son:

| Campo | Definición | Unidad |
|---|---|---|
| `Z` | Bilger local, calculado desde las fracciones másicas de especies; combustible y oxidante definidos en base molar | 1 |
| `C` | `Y_CO2 + Y_H2O + Y_CO + 0.5 Y_H2` para el ejemplo de CH₄ | 1 |
| `h` | Entalpía específica total, sensible más formación, con la referencia del mecanismo | J/kg |
| `delta_h` | `h_ad(Z,C) - h`, donde existe la superficie adiabática de referencia | J/kg |
| `omega_C` | Fuente volumétrica `sum(a_k W_k omega_k_molar)` | kg/(m³ s) |
| `qdot` | Liberación volumétrica de calor | W/m³ |

`C` tiene los mismos pesos en todas las llamas y no se estira para forzar que
cada extremo quemado valga uno. Se comprueba su monotonía en cada perfil;
otros combustibles pueden necesitar pesos distintos. El progreso sin escalar
evita los términos adicionales asociados a una normalización dependiente de
la composición, como explica [Gövert et al., sección 2.1.2](https://link.springer.com/article/10.1007/s10494-017-9848-4).

`Z` conserva su definición de Bilger: cero en el oxidante y uno en el
combustible. No se normaliza al intervalo de la figura ni se sustituye por
`Z_in`. El transporte de especies puede hacer que varíe dentro de una llama.
[Kinuta et al. (2024)](https://arxiv.org/html/2406.09727v1) emplean controles
`(C,Z,delta_h)` y muestran la importancia de la difusión preferencial para H₂.
Su modelo incluye estiramiento y cierres de transporte que esta biblioteca
plana todavía no incorpora.

La entalpía total tiene balance conservativo sin fuente química independiente:
la energía de formación ya forma parte de `h`. Para acoplar un futuro solver
reducido habrá que transportar `Z`, `C` y `h`, conservando los flujos de difusión
de especies y entalpía; disponer de una tabla no sustituye esas ecuaciones.

## Generación e interpolación

El ejemplo publicado utiliza GRI-Mech 3.0, CH₄-aire, 300 K, 101325 Pa,
transporte promediado por mezcla sin Soret y un dominio de 30 mm:

- Composiciones: `phi = [0.7, 0.85, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]`.
- Caudales de quemador: `r = [0.65, 0.45, 0.25, 0.20, 0.16, 0.12, 0.10, 0.08, 0.06]`,
  con `mdot = r rho_u Su` y `Su` resuelta para la referencia adiabática de cada mezcla.
- 90 llamas aceptadas: 9 adiabáticas y 81 en quemador; 181 muestras por trayectoria.
- 16290 vértices y 77760 tetraedros; ninguna celda invertida o degenerada.

La malla física de cada llama es no uniforme y adaptativa. La continuación
entre caudales proporciona una estimación inicial nativa; cada llama se vuelve
a resolver, refinar y comprobar. Un fallo numérico se registra como tal y no
se interpreta como extinción física.

El parámetro interno `s` distribuye las muestras sobre cada trayectoria. No es
el control físico `C`. Se conservan los extremos de cada llama, incluidos los
productos de enfriamiento que pueden superar el progreso final adiabático.

La malla de consulta conecta únicamente composiciones, caudales y muestras
contiguos. No rellena un casco convexo ni extrapola en los huecos. Las celdas
invertidas se excluyen y se cuentan; las consultas en interiores superpuestos
producen `AmbiguousManifoldError`. Un estado exterior produce
`OutsideManifoldError`.

Las especies y fuentes se interpolan con coordenadas baricéntricas. La
temperatura se recupera resolviendo `h(T,Y)=h_consulta` con los polinomios NASA;
no se interpola de forma independiente de la entalpía. Esta construcción
conserva `sum(Y)`, `Z` y `C` dentro de la precisión numérica. La producción
y la consulta utilizan química y termodinámica nativas; Cantera se importa
solo en las comparaciones de referencia.

~~~python
from kflame.fgm.nonadiabatic3d import NonAdiabaticFGM

fgm = NonAdiabaticFGM("docs/assets/nonadiabatic3d")
# Controles de un estado calculado; reemplázalos por los de tu caso.
Z, C, h = fgm.table["controls"][8000]
state = fgm.lookup(Z=float(Z), C=float(C), h=float(h))
print(state["T"], state["omega_C"])
~~~

## Validación numérica

Se resuelven llamas detalladas con composiciones **y** caudales ausentes del
entrenamiento. Sus controles locales sirven para consultar el FGM: es una
prueba **a priori de tabulación**. Cada llama detallada se compara además con
una solución independiente de `Cantera 3.2.0 BurnerFlame`, con el mismo
mecanismo, condiciones y transporte. Las llamas retenidas y las referencias
utilizan refinamiento más fino: pendiente 0.02 y curvatura 0.04.

Los umbrales de interpolación se fijaron antes de evaluar la primera tabla:
temperatura ≤20 K, especies ≤0.005, fuentes ≤15 % de su pico nativo y cobertura
≥85 %. Para la comparación nativa/Cantera se exigen temperatura ≤5 K,
especies ≤0.002, pérdida de entalpía ≤1 % de error y cierre energético ≤2 %.

La biblioteca inicial de 30 llamas falló en las fuentes de dos casos ricos,
con errores de aproximadamente 49 % y 75 %. Se añadieron composiciones y
caudales, manteniendo esos casos fuera del entrenamiento y los umbrales
originales. La biblioteca de 90 llamas pasa las comprobaciones siguientes:

| φ retenida | r retenido | Máx. error T FGM/nativa [K] | Error Ω_C / pico [%] | Cobertura de nodos [%] |
|---:|---:|---:|---:|---:|
| 0.771362 | 0.55 | 1.244 | 5.25 | 96.20 |
| 0.921954 | 0.35 | 1.999 | 4.86 | 95.40 |
| 1.072381 | 0.18 | 3.093 | 8.37 | 98.54 |
| 1.222702 | 0.085 | 0.457 | 8.56 | 98.66 |

El error máximo de liberación de calor es 8.99 % de su pico; el de especies,
6.79×10⁻⁴. La comparación nativa/Cantera da como máximo 1.889 K y 2.35×10⁻⁴
en especies. El cierre energético máximo de estas cuatro llamas es 0.359 %;
en la biblioteca de entrenamiento es 1.639 %. Se comprueba también la
conservación de los tres controles y de la suma de fracciones másicas.

Después de fijar la biblioteca se comprobaron otros dos casos, que no
intervinieron en el refinamiento:

| φ retenida | r retenido | Máx. error T FGM/nativa [K] | Error Ω_C / pico [%] | Cobertura de nodos [%] |
|---:|---:|---:|---:|---:|
| 1.025 | 0.28 | 2.374 | 5.37 | 97.04 |
| 1.275 | 0.11 | 0.146 | 7.56 | 98.52 |

Ambos pasan los mismos umbrales. Los informes completos y perfiles están en
[los datos publicados](assets/nonadiabatic3d/); incluyen huellas SHA-256 del
mecanismo, código de análisis, tabla y perfiles de entrenamiento.

La cobertura es la fracción de **nodos espaciales** dentro del dominio
calculado; no mide una fracción de volumen ni garantiza cobertura de un caso
CFD. Los errores de interpolación se calculan únicamente en esos nodos y
los restantes se registran como no cubiertos. En las figuras permanecen los
huecos; no se reemplazan por ceros ni por valores recortados.

Se comprobaron tres mallas para el caso φ=0.921954, r=0.35: 256, 391 y 746
nodos. Entre las dos más finas, `Tmax` cambia 0.715 K y el flujo de calor
0.00853 %. El cierre es 0.162 %, 0.186 % y 0.108 %, respectivamente. Este
estimador de frontera no decrece de forma monótona cuando se redistribuyen
nodos; el informe conserva sus tres valores y comprueba también los cambios
de temperatura y flujo.

## Figuras y reproducción

![Biblioteca de pérdidas hacia el quemador](assets/nonadiabatic3d/01_familia_perdidas.png)

![Cortes de temperatura y fuente en Z y C](assets/nonadiabatic3d/02_fgm_enthalpia.png)

![Perfiles retenidos y referencia independiente](assets/nonadiabatic3d/03_validacion_perfiles.png)

![Errores de tabulación](assets/nonadiabatic3d/04_validacion_errores.png)

Los mapas usan `Z` horizontal y `C` vertical, Cividis e isolíneas. Cada campo
comparte escala en ambos niveles de pérdida de entalpía; gris significa un
estado no resuelto. Las curvas utilizan colores diferenciados y tipos de línea.

Para reproducir las figuras a partir de los arrays publicados:

~~~sh
python examples/plot_nonadiabatic_3d.py docs/assets/nonadiabatic3d --output output/figures/nonadiabatic3d --pdf output/pdf/FGM_no_adiabatico_validacion.pdf
~~~

Para regenerar las llamas y las comparaciones:

~~~sh
python examples/example_nonadiabatic_3d.py --output runs/nonadiabatic
python examples/validate_nonadiabatic_3d.py runs/nonadiabatic --output runs/validation3d --fine-native --mesh-check
python examples/validate_nonadiabatic_3d.py runs/nonadiabatic --output runs/validation3d_extra --phis 1.025 1.275 --fractions 0.28 0.11 --fine-native
python examples/plot_nonadiabatic_3d.py runs/nonadiabatic --validation runs/validation3d --output output/figures/nonadiabatic3d
~~~

El paquete de datos permite consultar la tabla y redibujar las figuras; la
regeneración resuelve todas las llamas y guarda las trazas completas en `runs/`.
`--reuse-from` permite aprovechar otra biblioteca aceptada con condiciones y
refinamiento coincidentes. `--build-only` reconstruye la tabla desde perfiles
aceptados, y `--recheck` recalcula las métricas sin volver a resolverlos.

## Alcance

Se ha validado numéricamente una biblioteca de CH₄ con pérdidas hacia un
quemador plano a 300 K. Falta acoplar y validar las ecuaciones reducidas de
transporte. Tampoco se han validado experimentalmente extinción, apagado
transitorio, radiación, conducción dentro del sólido, una geometría de pared
multidimensional o un FGM no adiabático de H₂. El caso nativo de H₂ con Soret
de la primera fase está documentado en [el modelo de quemador](heat-loss.md).
