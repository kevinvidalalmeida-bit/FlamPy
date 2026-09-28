# Rutas de recuperación en las 104 condiciones L3

## Presentación actual en el capítulo: trabajo por llama

El capítulo presenta ahora `17_trabajo_por_llama.pdf`, con la comparación
KFLAME–Cantera en ocho casos centrales: ambos mecanismos y cuatro transportes
a phi=1, 300 K y 1 atm. Se utilizan las mismas dieciséis ejecuciones
instrumentadas que en la figura 16 de tiempos, con coincidencia exacta de
perfiles frente a producción. La figura
de frecuencias de 104 condiciones se conserva como análisis auxiliar.

```powershell
python tools/plot_flame_work_counts.py
```

El comando solo lee resultados guardados. Exporta PDF/PNG, contadores CSV y
procedencia JSON en `runs/thesis_flames_L3/report_expanded/revision_figures`.
Se comparan recuentos de residual y Jacobiano, tiempo registrado por construcción
de Jacobiano y pasos Euler aceptados. El cociente t_J/N_J resume operaciones
dentro de una ejecución, no repeticiones de la campaña. Los registros de Cantera
excluyen del residual las evaluaciones internas del Jacobiano por diferencias
finitas y pueden omitir etapas interrumpidas por ampliación de dominio. Sus
pasos Euler aceptados proceden de callbacks. Los contadores específicos de
química, transporte y álgebra lineal de KFLAME se conservan en el CSV y los
cuadros del capítulo; los ausentes en Cantera quedan sin dato.

## Diagnóstico global auxiliar

Este diagnóstico corresponde a KFLAME en las llamas individuales del capítulo 4.
Los tiempos originales, sus cinco repeticiones y los resultados FGM se conservan.

Desde la raíz del proyecto, ejecutar en PowerShell:

```powershell
python benchmarks/diagnose_flame_recovery.py --resume
```

El ejecutor verifica el código, entorno y configuración de la campaña original,
ejecuta secuencialmente una resolución instrumentada por condición y genera el
gráfico al finalizar. Cada proceso realiza además un calentamiento excluido:
son **104 resoluciones de diagnóstico + 104 calentamientos**, sin cinco
repeticiones. El diagnóstico incluye el problema preliminar multicomponente.

Comprobación previa sin resolver ni crear resultados:

```powershell
python benchmarks/diagnose_flame_recovery.py --dry-run
```

Al interrumpir con Ctrl+C, repetir el primer comando. Las ejecuciones terminadas,
incluidos los fallos, quedan registradas y no se repiten silenciosamente. Los
intentos interrumpidos se archivan al reanudar. Un fallo completado o un perfil
diferente queda identificado en el informe y requiere revisión.

## Figura y datos

Los registros completos quedan en `runs/thesis_flames_L3_recovery`. El informe se
escribe en `runs/thesis_flames_L3/report_expanded/revision_figures`:

- `17_recuperacion_global.pdf` y `.png`: barras horizontales con el total de
  llamas que utilizaron cada etapa, separadas por mecanismo (52 condiciones cada
  uno). Los colores distinguen los cuatro transportes, con 13 condiciones cada uno.
- `17_recuperacion_condiciones.csv`: indicador individual de cada ruta y contadores.
- `17_recuperacion_resumen.json`: grupos, cobertura y casos pendientes/fallidos.
- `17_recuperacion_evidencia.json`: fuentes y hashes de perfiles y registros.
- `17_recuperacion_analisis.tex`: interpretación auxiliar. Se publica únicamente
  cuando las 104 condiciones están verificadas; la selección actual del capítulo
  usa la figura de trabajo por llama descrita arriba.

El postprocesado verifica igualdad exacta de coordenadas, temperatura, velocidad y
composición con la primera repetición utilizable original. Sus frecuencias
describen una trayectoria por condición, no la dispersión entre las cinco
repeticiones originales. Los tiempos instrumentados no se añaden a las
estadísticas de rendimiento.

Las categorías son **solo Newton**, **PTC–SER**, **Euler implícito/BE**, **rescate
térmico**, **ampliación de dominio** y **adaptación de malla**. PTC/BE cuentan
activación, aunque sus pasos se rechacen; rescate térmico significa que se intentó
resolver con temperatura fijada. Solo Newton exige ausencia de los tres mecanismos
de recuperación no lineal a lo largo de todas las etapas. Ampliación y adaptación
son controles espaciales y pueden coexistir con cualquier ruta. Las categorías
no suman necesariamente el 100 %. La estrategia configurada no demuestra su uso.

Regeneración sin resolver llamas:

```powershell
python tools/postprocess_flame_recovery.py
```

Para actualizar el PDF independiente del capítulo después de terminar:

```powershell
Push-Location .local/research/thesis
pdflatex -interaction=nonstopmode -halt-on-error resultados_ampliados.tex
pdflatex -interaction=nonstopmode -halt-on-error resultados_ampliados.tex
Pop-Location
```
