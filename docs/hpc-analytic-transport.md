# HPC: derivadas y actualizaciones algebraicas del transporte

Revisión del 10 de octubre de 2026. [Guía de lectura](README.md) · [Estado de estrategias](hpc-strategy-status.md) · [Datos y fuentes](assets/chemistry/hpc-analytic-transport-2026-10-10.json)

Se implementaron dos opciones experimentales de composición: derivadas directas y actualizaciones algebraicas de las perturbaciones finitas. **Ambas están desactivadas por defecto.** La segunda conserva los pasos anteriores y mejora el ensamblaje y el Newton acotado de los tres mecanismos grandes ensayados. La primera mejora el ensamblaje, pero empeora el Newton acotado de Aramco 2. No hay evidencia nueva de aceleración hasta obtener llamas grandes convergidas ni de una victoria en tiempo frente a Cantera.

## Tiempo de la fuente integrada

Ciclo completo de **propiedades/banco + residual + Jacobiano**, con todas sus asignaciones. Numba=4, BLAS=1, siete pares alternados tras calentamiento; JIT y factorización posterior excluidos. «Antes» es la instantánea al inicio del 10 de octubre, que ya contiene las mejoras anteriores. «Ahora» habilita únicamente `algebraic_frozen_transport`.

| Mecanismo | Nodos | Antes (s) | Ahora (s) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|---:|
| JetSurF 2.0 | 9 | 0,230 | 0,161 | 1,44× | 7/7 |
| JetSurF 2.0 | 33 | 0,673 | 0,426 | 1,58× | 7/7 |
| AramcoMech 2.0 | 9 | 0,573 | 0,380 | 1,50× | 7/7 |
| AramcoMech 2.0 | 33 | 1,715 | 1,054 | 1,63× | 7/7 |
| AramcoMech 3.0 | 9 | 0,903 | 0,587 | 1,56× | 7/7 |
| AramcoMech 3.0 | 33 | 2,757 | 1,641 | 1,67× | 7/7 |

Se muestran medianas de tiempos independientes y mediana de cocientes emparejados; no tienen que dar el mismo cociente. Estos estados no son llamas convergidas. Los registros conservan muestras, controles y hashes; las mediciones del prototipo y de la variante analítica anterior están identificadas por separado.

## Coste dentro de Newton

Nueve nodos y tres iteraciones, siete pares, incluyendo propiedades, Jacobiano, factorización, solves y búsquedas de paso. **Todas las ejecuciones de este ensayo terminan en `iteration_limit`.** La reevaluación final que informa el residual queda fuera del cronómetro.

| Mecanismo | Nodos | Antes (s) | Ahora (s) | Cociente emparejado | Victorias |
|---|---:|---:|---:|---:|---:|
| JetSurF 2.0 | 9 | 0,475 | 0,415 | 1,15× | 7/7 |
| AramcoMech 2.0 | 9 | 1,208 | 1,012 | 1,19× | 7/7 |
| AramcoMech 3.0 | 9 | 1,910 | 1,599 | 1,20× | 7/7 |

La campaña analítica anterior observó en Aramco 2 un Jacobiano y **10 evaluaciones del residual frente a 4** del control: el ahorro de ensamblaje no compensó los intentos adicionales. Esa medición corresponde a la fuente de 65 archivos anterior a la integración algebraica y sigue registrada. Una derivada más precisa localmente no garantiza que una aproximación congelada produzca mejores pasos globales de Newton.

Se probó también el solver completo de JetSurF en nueve nodos, sin refinamiento y con límite de 60 s del solver por versión. El tiempo externo incluye compilación y no se usa como comparación de rendimiento. Referencia: convergencia=False, aceptada=False, motivo=`time_limit`, residual escalado máximo=0,834042. Opción algebraica: convergencia=False, aceptada=False, motivo=`time_limit`, residual escalado máximo=0,834042. Este par no acredita una ganancia en llamas refinadas ni sustituye la comparación física completa con Cantera.

## Dónde permanece el tiempo

Perfil de tiempo de pared de la opción algebraica, 33 nodos y cinco pares. Los porcentajes suman las muestras de cada fase; no son suma de tiempos de trabajadores ni perfil de una llama completa. Excluyen factorización y trayectoria de Newton.

| Mecanismo | Propiedades y banco | Residual con propiedades | Jacobiano |
|---|---:|---:|---:|
| JetSurF 2.0 | 33,7 % | 0,04 % | 66,2 % |
| AramcoMech 2.0 | 34,6 % | 0,02 % | 65,4 % |
| AramcoMech 3.0 | 35,0 % | 0,01 % | 65,0 % |

Las columnas de temperatura y el cierre multicomponente completo conservan sus cálculos anteriores. El siguiente trabajo analítico con mayor interés es diferenciar sus sistemas implícitos y reutilizar las factorizaciones; este perfil todavía no separa los componentes internos del Jacobiano y no cuantifica esa ganancia futura.

## Qué cambia y qué se valida

En una cara, se comparten las proyecciones base del flujo y de Wilke. Todas las columnas perturbadas pasan de repetir productos matriciales a **O(Ns²)** con recorridos por filas contiguas. Este orden solo describe esas columnas de cara; el cierre completo y la factorización global mantienen sus costes. La opción algebraica conserva los incrementos representables, el paso nominal del cociente, las filas y la química nodal anteriores; admite diferencias de redondeo.

La ruta analítica deriva además densidad, cp, advección, entalpía y radiación. En ambas opciones siguen congelados `D`, `DT` y `lambda` para las columnas de composición, conforme a la política congelada existente: **no es la derivada completa de `D(Y)`, `DT(Y)` y `lambda(Y)`**. El residual físico sigue recalculando sus propiedades reales.

- **203 pruebas seleccionadas aprobadas**, sin omisiones, con código de salida 0 y recibo por caso. Incluyen 47 de derivadas de cara, 10 de filas analíticas y 38 de perturbaciones algebraicas, además de controles de bancos, caras, rangos y propiedades. El primer log interrumpido se conserva y no se cuenta como cierre.
- Seis ensamblajes con propiedades y residual iguales por bytes, incluyendo cambio de estado y retorno. Desactivar las opciones reproduce los cinco bloques anteriores por bytes. Las columnas de flujo/temperatura y la masa también conservan sus bytes con la opción activa; la composición usa tolerancias condicionadas por el redondeo del cociente antiguo.
- Difusión, Soret y conductividad contra Cantera en tres caras de cada ensamblaje, `rtol=3e-7`, `atol=1e-12`. Son comprobaciones de propiedades, no tiempos de Cantera.
- Acciones direccionales de ambas opciones contra diferencias centradas del residual congelado en los tres mecanismos, con composición densa/dispersa, malla no uniforme, radiación y temperatura fija/libre. El control de la opción algebraica conserva también el error del método finito anterior; los seis casos finales pasan el umbral absoluto de 2e-6. El JSON especifica los gates y todos los pasos. Se verifica además la prioridad analítica cuando ambas opciones están activas.
- Incrementos negativos, estados inseguros y bancos ausentes/obsoletos usan las rutas anteriores. Los términos flooreados de Wilke se evalúan con pesos normalizados para evitar subdesbordamiento prematuro; se probaron caras inactivas, entradas de solo lectura, layouts y propiedad privada de salidas.

El banco sigue limitado a **96 MiB**, con Ns≥256 y química por tiles. Aramco 3 con 33 nodos usa aproximadamente **82,55 MiB**; con 65 supera el límite y estas opciones no se aplican automáticamente. Windows informa L2 agregado de 12 MiB y L3 de 64 MiB en este Ryzen 9 7845HX: el banco completo no cabe en esa capacidad L3 reportada. No se midieron contadores de fallos L1/L2/L3, por lo que la ganancia se atribuye al algoritmo medido y no a un porcentaje supuesto de cache hits.

## Derivadas térmicas y estrategias siguientes

El prototipo privado de primitivas de temperatura pasa **30 estados de 5 mecanismos**: viscosidad de especie, relajación rotacional, difusión binaria, A*/B*/C* y cp NASA, con barrido de pasos. Conserva el primal por bytes y contrasta el transporte primal con Cantera 3.2.0. No está integrado en el solver y todavía no produce `D′`, `DT′` ni `lambda′` completos; no se le atribuye aceleración.

Para `L(T) a(T)=b(T)`, la tangente satisface `L a′=b′−L′a`. Esta identidad permite reutilizar factores del estado base; la integración debe respetar los cambios de rama de NASA y de modos internos, y proyectar el flujo para evitar formar innecesariamente todas las derivadas matriciales. La formulación de bloques y el cálculo de conductividad/Soret pueden revisarse en el [código primario de MultiTransport de Cantera 3.2](https://www.cantera.org/3.2/cxx/d2/d60/MultiTransport_8cpp_source.html).

Las alternativas de transporte semiimplícito del [trabajo de Fillo et al.](https://arxiv.org/abs/1808.05463) tienen otro alcance y requieren validación propia en este solver. Permitir redondeo elimina una restricción anterior de aceptación, pero no convierte automáticamente en rápidas las variantes antes medidas como más lentas. No se activó GPU en esta campaña; la [comparación CPU/GPU anterior](hpc-mechanisms.md) conserva su alcance.

## Uso experimental y procedencia

Sobre un `DiffusionFlameProblem` ya creado:

```python
problem.algebraic_frozen_transport = True
problem.analytic_frozen_transport = False
```

Para probar las filas analíticas, invertir los valores. Si ambas están activas, las filas analíticas válidas tienen prioridad; los estados no admitidos vuelven al helper anterior. Esto no añade una opción CLI ni modifica las tolerancias, Newton, PTC/SER, damping o adaptador de malla.

Se revisaron por separado integración, oráculos/pruebas e informe con tres agentes. Las ejecuciones numéricas fueron secuenciales. La referencia conserva 63 fuentes y la integración final 66; solo cambian los dos archivos de difusión y se añaden tres módulos de composición. Los cuerpos de Newton, pseudo-transitorio, solver completo y factorización conservan su AST. Los [datos](assets/chemistry/hpc-analytic-transport-2026-10-10.json) incluyen los hashes y distinguen cada instantánea.
