# Resultados ilustrados

[Inicio](../README.md) · [API](api.md) · [Archivos y consulta](outputs.md)

La galería muestra campos de ejecuciones numéricas guardadas del núcleo
de FlamPy. Los datos de las figuras se incluyen en el repositorio para
reproducirlas sin volver a resolver las llamas.

## Llama de metano–aire

![Temperatura, velocidad, liberación de calor y composición de CH₄–aire](assets/flame-ch4.png)

| Condición o resultado | Valor |
|---|---|
| Mecanismo | `gri30.yaml`, 53 especies |
| Combustible / oxidante | CH₄ / aire, O₂:N₂ = 1:3,76 en base molar |
| Relación de equivalencia | φ = 1 |
| Temperatura de entrada | 300 K |
| Presión | 101 325 Pa |
| Transporte | Promediado por mezcla, sin Soret |
| Velocidad laminar Sᵤ | 0,378831 m/s |
| Nodos finales | 261 |
| Aceptación numérica registrada | Sí |

Se dibujan los nodos originales. La posición se desplaza al punto medio
del salto de temperatura y la ventana se concentra en el frente de llama;
el archivo conserva el dominio completo. La velocidad local aumenta con
la expansión del gas. La velocidad de propagación Sᵤ corresponde a la
mezcla no quemada y se distingue de esa velocidad local.

## Tablas FGM de metano e hidrógeno

![Temperatura tabulada en las familias CH₄–aire y H₂–aire](assets/fgm-temperature.png)

Ambas familias corresponden a entrada a 300 K, presión de 101 325 Pa,
transporte promediado por mezcla sin Soret y φ entre 0,7 y 1,4.

| Resultado de estas construcciones | CH₄–aire | H₂–aire |
|---|---|---|
| Mecanismo | `gri30.yaml` | `h2o2.yaml` |
| Flamelets finales | 40 | 25 |
| Puntos de progreso | 241 | 241 |
| Todas las llamas finales aceptadas | Sí | Sí |
| Defecto final por exclusión registrado | 0,903 % | 0,500 % |

Las dos imágenes usan la misma escala de color para temperatura. El eje
horizontal Z⋆ normaliza el intervalo de composición de entrada de cada
familia a [0, 1]; no implica que CH₄ y H₂ compartan los mismos valores
físicos de Z. El eje vertical es el progreso normalizado c.

El número final de flamelets es propio de cada construcción y no un
resultado garantizado para cualquier configuración o versión. El defecto
por exclusión mide consistencia de interpolación entre filas; no es una
medida de error frente a experimentos ni una cota global del error físico.

## Reproducir las imágenes

Desde la raíz del repositorio:

~~~sh
python -m pip install -e ".[plots]"
python docs/assets/generate_gallery.py
~~~

El script regenera `flame-ch4.png` y `fgm-temperature.png` con Matplotlib.
Utiliza una tipografía, paleta y número de marcas comunes; no modifica
los datos ni llama al solver.

Los archivos compactos de [assets/data/](assets/data/) contienen solo los
campos necesarios para dibujar. La [procedencia](assets/data/provenance.json)
registra condiciones, identificadores de origen y SHA-256 de cada archivo
publicado. Los identificadores históricos de las ejecuciones se conservan
para trazabilidad; el nombre público del proyecto es FlamPy.

Estos archivos ilustrativos no son tablas completas para consulta de
composición o exportación. Para generar resultados completos utiliza los
[ejemplos](../examples/README.md) y consulta su [estructura](outputs.md).
