# Campaña actual: L3, inicio directo

Elección del usuario: L3 para ambos mecanismos, sin exigir una fase previa
de sensibilidad de refinamiento o dominio.

- GRI30: Newton estacionario + PTC–SER y rescate Euler.
- h2o2: Newton estacionario + Euler implícito directo.
- L3: slope=0.01, curve=0.02, ratio=2.5, prune=0.
- 56 condiciones, cinco pares KFLAME–Cantera: 560 resoluciones medidas.
- Se mantienen aceptación interna, adaptación de malla, expansión de dominio,
  diagnósticos, guardado de perfiles y registro de fallos.

## Ejecutar ahora

Desde la raíz del proyecto:
~~~powershell
python benchmarks/benchmark_flame_sweeps.py --phase main --spatial-policy fixed-L3 --pairs 5 --output runs/thesis_flames_L3 --resume
~~~

Este comando inicia directamente la campaña principal. No ejecutar verify.
Si se interrumpe, repetir exactamente el comando para reanudar. Los pares
completos compatibles se omiten; los interrumpidos se repiten conservando
el intento anterior. Los fallos completados no se reintentan automáticamente.

## Después de la campaña

La reparación de almacenamiento por WinError 5 conserva los resultados y
añade reintentos acotados de sustitución atómica y nombres temporales únicos
para JSON. En `storage_repairs/` se conservan los manifiestos y fuentes antes
y después del cambio, junto con la auditoría que comprueba que el código de
cálculo y el resto del entorno no cambiaron. El par cuyo guardado se interrumpió
se recuperó validando los resultados de ambos procesos, los metadatos del
coordinador y las huellas SHA-256 de los perfiles, sin repetir sus resoluciones.
Se reanuda con el mismo comando anterior. Si un bloqueo persiste más allá
de los reintentos, el destino previo y el temporal se conservan para revisión.

~~~powershell
# Dos ablaciones, cinco pares cada una
python benchmarks/benchmark_flame_sweeps.py --phase ablation --spatial-policy fixed-L3 --pairs 5 --output runs/thesis_flames_L3 --resume

# Tablas, estadísticas y ocho figuras PDF/PNG desde los datos guardados
python tools/postprocess_flame_sweeps.py --input runs/thesis_flames_L3 --output runs/thesis_flames_L3/report
~~~

Los gráficos identifican L3 y las estrategias por mecanismo. El apartado
de sensibilidad indica que ese estudio no se realizó en esta campaña.
No se declara que L3 haya superado un umbral de independencia espacial.

## Barridos y medición

Por mecanismo: 20 condiciones de composición (cinco phi por cuatro
transportes), cuatro presiones adicionales y cuatro temperaturas adicionales:
28 condiciones únicas. Los puntos compartidos se calculan una sola vez por
repetición y solver. Las rutas de presión y temperatura son CH4 promediado sin
Soret y H2 multicomponente/Soret.

Phi=0.7,0.9,1,1.1,1.4; presión=1,2,3,5,10 atm; temperatura=300,350,400,450,500 K.
Los barridos varían una condición cada vez alrededor de phi=1, 300 K y 1 atm.
Aire O2:N2=1:3.76; base de gradiente molar. Máximo 6000 nodos, dominio inicial
0.03 m con expansión automática, tolerancias estacionarias 1e-5/1e-10.
L3 define criterios de refinamiento, no un número fijo de nodos.

Cada ejecución usa un proceso independiente y un calentamiento separado.
Se descarta el perfil de calentamiento y se vacían las cachés moleculares
antes de medir. El tiempo incluye construcción, inicialización, recuperación
y adaptación, excluyendo calentamiento, escritura y diagnóstico. Se alterna
el orden de los solvers; Numba usa 4 hilos, BLAS/MKL/OpenMP 1.
El límite es 600 s por resolución. Mantener equipo, entorno y código fijos.

Se presentan tiempos individuales, mediana, cuartiles, aceptación y razón
Cantera/KFLAME. El IC95 bootstrap pareado usa 20000 remuestreos y semilla
20260927 sobre pares utilizables; requiere al menos tres pares.
Las ablaciones de bloques compilados (CH4) y reutilización molecular (H2)
mantienen la estrategia correspondiente a su mecanismo.

## Procedencia y archivos

runs/thesis_flames_L3 es un directorio nuevo. Los resultados anteriores de
runs/thesis_flames, runs/thesis_flames_L4 y runs/h2_euler_probe_L4_p10
conservan sus protocolos. No se mezclan en las estadísticas principales.

manifest.json guarda el nivel elegido, la ausencia de verificación previa,
la política por mecanismo, entorno, hilos y huellas del código. La selección
se registra en spatial_selection.json sin un certificado de verificación.
Cada ejecución guarda request.json, warmup.json, result.json, profile.npz
y console.log. Los perfiles incluyen todas las especies, propiedades,
fuentes y flujos reconstruidos. inputs/ guarda datos físicos; code/ copia
las fuentes identificadas; index.csv/json resume las ejecuciones completas.
Los controles de compatibilidad y el bloqueo contra escritores simultáneos
se mantienen activos.

La tesis carga runs/thesis_flames_L3/report/report.tex. El postprocesador
admite campañas incompletas y no invoca los solvers. Para incorporar el informe:
~~~powershell
Push-Location .local/research/thesis
pdflatex -interaction=nonstopmode -halt-on-error TFM_FGM_FINAL.tex
pdflatex -interaction=nonstopmode -halt-on-error TFM_FGM_FINAL.tex
Pop-Location
~~~

