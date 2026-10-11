# Mecanismos locales

El cierre Soret promediado por mezcla se adapta de
`Cantera v3.2.0/src/transport/MixTransport.cpp`; el término de flujo sigue
`src/oneD/Flow1D.cpp`. La licencia BSD-3 y atribuciones se conservan en
`CANTERA_TRANSPORT_LICENSE.txt`. Se evalúa con ajustes moleculares nativos.

`gri30.yaml` y `h2o2.yaml` son las copias distribuidas con Cantera 3.2.0,
incorporadas sin modificar sus datos el 19 de septiembre de 2026. Conservan
las descripciones, generador, fecha y atribuciones incluidas en cada YAML.
El primero corresponde a GRI-Mech 3.0; el segundo es su submecanismo H2/O2
con N2. El lector usa la primera fase ideal-gas con termodinámica NASA-7.

Son datos de entrada locales, no llamadas a Cantera ni resultados de llamas.
La licencia de distribución de Cantera se conserva en
`CANTERA_TRANSPORT_LICENSE.txt`. La procedencia original de GRI-Mech se
indica en la cabecera de `gri30.yaml`.

La ampliación del 9 de octubre incorpora `drm19.yaml`, `drm22.yaml`,
`ffcm1.yaml`, `uscmech2.yaml` y `jetsurf2.yaml` desde archivos oficiales de
Berkeley y Stanford. Se conservan atribuciones y procedencia en los YAML;
los archivos USC/JetSurF conservan también sus fuentes originales y datos
suplementarios. La licencia de FlamPy no se asigna a los datos de los autores.
Los recuentos y el alcance de la conversión están en
[mecanismos químicos](../../../../docs/mechanisms.md).

AramcoMech se conserva fuera del paquete, en el directorio local del usuario
`~/.kflame/mechanisms`, configurable mediante `KFLAME_MECHANISM_DIR`.
Para otro mecanismo, suministrar una ruta explícita a un YAML compatible.
Una ruta inexistente produce un error; no se sustituye por otro archivo con
el mismo nombre ni se busca dentro de la instalación de Cantera.
