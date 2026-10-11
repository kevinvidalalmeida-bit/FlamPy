# Jacobiano térmico: derivadas de transporte y reutilización de Schur

La nueva opción experimental `thermal_transport_tangents=True` reutiliza la
inversa de difusión y la factorización del sistema térmico de cada cara.
Calcula las derivadas de difusión, Soret, conductividad y viscosidad para
evitar volver a resolver el transporte en las columnas de temperatura.
La opción permanece **desactivada por defecto**.
Se selecciona únicamente con el banco admitido: al menos 256 especies,
química analítica por bloques, transporte multicomponente congelado y Schur.
Las pruebas de H₂/O₂ y GRI validan las derivadas, sin activar esta ruta en
los mecanismos pequeños.

Para reproducir la configuración medida con un problema existente:

```python
problem.algebraic_frozen_transport = True
problem.thermal_transport_tangents = True
```

Newton prepara y consume el banco automáticamente cuando está admitido.
Una llamada manual a `jacobian_blocks` necesita las propiedades y el banco
de `properties_for_newton` del mismo estado para usar estas derivadas.

## Antes y después

Referencia: versión anterior con `algebraic_frozen_transport=True` y columnas
de temperatura por diferencias finitas. Candidata: la misma versión con la
nueva opción térmica. 33 nodos, Numba 4 y BLAS 1, siete pares alternados,
compilación excluida. El tiempo incluye propiedades, banco, preparación de
derivadas, residual y Jacobiano, con sus asignaciones. No incluye factorización
del Jacobiano ni una llama completa convergida.
El cociente emparejado es la mediana de siete cocientes; puede diferir del
cociente de las medianas mostradas.

| Mecanismo | Antes (s) | Ahora (s) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|
| JetSurF 2.0 | 0.542 | 0.415 | 1.325× | 7/7 |
| AramcoMech 2.0 | 1.310 | 1.007 | 1.306× | 7/7 |
| AramcoMech 3.0 | 1.894 | 1.498 | 1.291× | 7/7 |

## Dónde queda el tiempo

Medianas de las mismas muestras, en segundos. Propiedades incluye toda la
preparación adicional del banco térmico; el porcentaje no es un contador
interno exclusivo del transporte térmico.

| Mecanismo | Versión | Propiedades y banco | Jacobiano | Jacobiano / total |
|---|---|---:|---:|---:|
| JetSurF 2.0 | Antes | 0.186 | 0.355 | 65.6% |
| JetSurF 2.0 | Ahora | 0.323 | 0.090 | 21.8% |
| AramcoMech 2.0 | Antes | 0.451 | 0.833 | 63.6% |
| AramcoMech 2.0 | Ahora | 0.818 | 0.181 | 17.9% |
| AramcoMech 3.0 | Antes | 0.673 | 1.257 | 66.4% |
| AramcoMech 3.0 | Ahora | 1.231 | 0.271 | 18.1% |

## Qué cambió

Se diferencian las funciones ajustadas de temperatura y los cinco bloques
Dixon–Lewis. El sistema implícito usa la inversa y la LU de Schur retenidas,
sin repetir la solución base. La difusión ordinaria conserva toda la resta
de la diagonal: no se descarta el término suponiendo que una suma es cero
exactamente. La derivada completa de la inversa aún requiere dos productos
de matrices, incluidos en las mediciones.

Cada cara exporta dos vectores de derivadas de flujo y las derivadas de
conductividad y viscosidad. El banco persistente adicional cuesta
`8*(caras)*(2*especies+3)` bytes. Las matrices temporales son privadas de cada
trabajador. El límite conjunto del banco sigue siendo 96 MiB; Aramco3 con
65 nodos continúa por la ruta ordinaria.

Las columnas conservan el incremento, la termodinámica nodal y la evaluación
de filas existentes. Solo el transporte de la prueba se linealiza. Esto es
una derivada analítica del transporte dentro de la columna térmica; el
Jacobiano térmico completo aún conserva diferencias de filas.

Las caras que cruzan una unión NASA, un umbral de modos internos, un piso de
coeficientes, o presentan datos inseguros vuelven al cálculo anterior. El
banco valida propietario, estado, malla, opciones y `Tref`. La compilación
previa incluye las nuevas firmas. No se cambiaron Newton, PTC/SER, tolerancias,
factorización del Jacobiano ni mecanismos químicos; se comprobó su AST.

## Validación

Las derivadas completas y los flujos de ambos extremos se compararon con
diferencias centradas en cinco mecanismos, con composiciones densas y con
especies ausentes, Soret activado/desactivado. Cada campo exige al menos dos
pasos que pasen el barrido. La tolerancia es `rtol=3e-5` más
`1e-8*max(max|derivada|, max|base|/T)`; se guardan curvas y errores absolutos.
Un primer control con tolerancia absoluta implícita demasiado pequeña falló
por cancelación en especies ausentes; se conserva como ensayo incompleto.

Cantera 3.2.0 valida los coeficientes base con `rtol=3e-7`, `atol=1e-12`.
El control direccional de temperatura usa el residual físico completo,
malla no uniforme, radiación, temperatura libre/prescrita y Soret
activado/desactivado; exige dos pasos con error escalado menor que `2e-6`.
Las propiedades y el residual base, las columnas de caudal y composición y
la opción desactivada conservaron sus bytes frente a la referencia congelada.

Las pruebas pequeñas añaden controles independientes de los cinco bloques,
asimetría, entradas de solo lectura, disposición C/Fortran/estriada, propiedad
de memoria, reutilización A/B/A, cambios de rama y presupuesto del banco.

Pruebas seleccionadas: **108 comprobaciones validadas**, ninguna pendiente de fallo.

Se conservan 79 controles que pasaron en la ejecución completa inicial y
29 pruebas del módulo térmico que pasaron en su ejecución final. Se corrigió
un criterio del test: Numba posee sus buffers mediante MemInfo y NumPy puede
mostrar `OWNDATA=False`. El control final verifica independencia y duración
del almacenamiento. La fuente numérica y sus tolerancias no cambiaron por
esta corrección. Ninguna ejecución interrumpida aporta pruebas al resultado.

## Newton y alcance pendiente

Tres iteraciones reales, nueve nodos, siete pares. Incluye residual completo,
factorización, solución y búsqueda de paso. Estas pruebas están limitadas por
iteraciones; sus tiempos no demuestran llamas convergidas.
JetSurF y Aramco2 no mostraron una ganancia consistente en esta prueba;
Aramco3 ganó aproximadamente un 6 %. Las diferencias máximas entre estados
escalados fueron 1,03e-14, 8,94e-10 y 1,39e-10 respectivamente. Todas las
ejecuciones alcanzaron el límite de tres iteraciones, con una evaluación
del Jacobiano y cuatro del residual por versión.

| Mecanismo | Antes (s) | Ahora (s) | Cociente | Ambas convergidas |
|---|---:|---:|---:|---|
| JetSurF 2.0 | 0.456 | 0.458 | 0.986× | No |
| AramcoMech 2.0 | 1.127 | 1.113 | 0.999× | No |
| AramcoMech 3.0 | 1.849 | 1.749 | 1.059× | No |

La nueva ruta queda experimental hasta completar validación de convergencia
de llamas. No se activa GPU ni se afirma una victoria del solver completo
frente a Cantera.

[Datos, muestras y huellas completos](assets/chemistry/hpc-thermal-jacobian-2026-10-10.json) ·
[Mejora anterior de composición](hpc-analytic-transport.md) ·
[Estado de estrategias](hpc-strategy-status.md)

La formulación base se contrastó con el
[código oficial MultiTransport de Cantera 3.2](https://www.cantera.org/3.2/cxx/d2/d60/MultiTransport_8cpp_source.html).
