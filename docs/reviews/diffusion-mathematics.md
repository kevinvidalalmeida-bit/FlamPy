# Revisión matemática independiente del solver de contraflujo

7 de octubre de 2026. Base: `70708b41c5f670ba7e7701dd4ca682258622fdb6`. Alcance: `src/kflame/flame/diffusion.py` y `diffusion_kernels.py`, con lectura del cierre de flujo, pruebas y evidencia archivada. No se modificaron el núcleo, referencias, metadatos de medición ni manuscrito. Las comprobaciones nuevas son evaluaciones pequeñas en memoria; no se resolvieron llamas ni se ejecutaron benchmarks completos.

La corrección concurrente de transporte/backend comunicada por el agente principal queda fuera de esta auditoría del checkpoint. Puede cambiar las cifras de perturbaciones fuera del simplex del hallazgo 3 y deberá verificarse tras integrar; no invalida la reproducción de P0 ni modifica las ecuaciones discretas auditadas. No se duplica aquí el bug de transporte hallado en la revisión independiente correspondiente.

**Resultado:** los signos y las ecuaciones físicas revisados son coherentes. Se reprodujo **un bug del diagnóstico integral de entalpía**. Otros cuatro hallazgos corresponden a políticas de precisión, limitaciones de estancamiento y propuestas no medidas. Ninguno permite prometer una aceleración.

## Comprobación de la formulación

Con `m=rho*u`, `s=2*rho*V`, fuentes másicas `omega_k` y entalpías específicas `h_k`, los ceros del residual son:

```text
m' + s = 0
m V' + rho V² + Lambda - (mu V')' = 0
m Y_k' + J_k' - omega_k = 0
m cp T' - (lambda T')' + sum(J_k h_k') + sum(h_k omega_k) = 0
```

Son equivalentes a las [ecuaciones de Cantera 3.2](https://cantera.org/3.2/reference/onedim/governing-equations.html), que expresan la producción química en unidades molares. `V=v/r` y Lambda es constante. No falta calor de formación ni exportación radial en la interpretación continua.

| Aspecto | Evidencia local | Veredicto |
|---|---|---|
| Continuidad | `diffusion_kernels.py:33`, `:62` | Todos los n−1 intervalos incluidos; factor 2 y regla trapezoidal correctos |
| Momento | `diffusion_kernels.py:38–41` | Convección, rho V², Lambda y viscosidad con signos correctos |
| Especies/energía | `diffusion_kernels.py:44–57` | Producción y reacción entálpica correctas; divergencias de cara compartidas |
| Unidades | `diffusion.py:157–158`, `:202–203` | h molar dividida por W antes de combinarla con fuentes másicas |
| Entradas | `diffusion_kernels.py:17–29` | m combustible positivo, m oxidante negativo; ambas cumplen `m*(Y-Yfeed)+J=0`, T fija y V=0 |
| Autovalor | `diffusion.py:149–151`; `diffusion_kernels.py:62` | Escalar global válido; hay n*(ns+3)+1 ecuaciones e incógnitas, sin fijar el estancamiento |
| Normalización | `diffusion_kernels.py:29`; `diffusion.py:403`, `:415–416` | Una especie sustituida por suma(Y)=1 en cada entrada; suma comprobada en todo el dominio |
| Continuación temporal | `diffusion_kernels.py:47`, `:59`, `:61` | Backward Euler correcto para Y/T/V; flujo y entradas algebraicos |

La condición derecha coincide con la combinación de dominios e inlet de [Boundary1D.cpp v3.2.0, líneas 238–260](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/Boundary1D.cpp#L238-L260). No cambiar el signo de J allí. Algunas fórmulas de la documentación de discretización presentan signos inconsistentes; para momento se comprobó el código: `shear-Lambda-m*V'-rho*V²`, equivalente al residual nativo de signo opuesto. [Flow1D.cpp v3.2.0, líneas 597–600](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/Flow1D.cpp#L597-L600).

## 1. P0 — Bug reproducido: diagnóstico de entalpía incompatible con continuidad

**Gravedad media, alta si se interpreta como pérdida física pequeña.** `diffusion.py:390` promedia productos nodales m*h; `:391–394` integra la interpolación lineal de s*h entre las caras extremas. Las dos reconstrucciones no satisfacen conjuntamente la continuidad discreta.

Con suma(Y)=1, suma(J)=0 y suma(omega)=0, cambiar el cero de entalpía `h→h+C`, `h_k→h_k+C` no altera el residual energético. El defecto absoluto tampoco debería cambiar para una continuidad satisfecha. El diagnóstico actual, en cambio, añade:

```text
delta(defecto)/C = dz[0]*(s[1]-s[0])/8
                 + dz[-1]*(s[-2]-s[-1])/8
```

Es una derivación propia por suma de continuidad y resta de las medias celdas extremas.

**Prueba ejecutada:** z=[0,.5,1], m=[.5,0,−.5], V=[0,1,0], rho=cp=T=Y=1, mu=lambda=1, Lambda=−9, J=omega=0, h_k=h=H. Entradas .5/.5 y T=1; todas las escalas 1. `assemble_residual.py_func` da **residual completo exactamente cero**, incluido momento, entradas y última continuidad.

| H | Máximo residual | Balance nodal completo | Defecto actual entre caras |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | .25 |
| 1 000 000 | 0 | 0 | 250 000 |

Para H=10⁶, el diagnóstico relativo da **33.33%**. Es una prueba discreta fabricada, no una llama física. El flujo entre caras cambia en −H/2 y la cuadratura radial aporta 3H/4; el balance completo es −H+H=0.

**Acción:** reconstruir m en caras integrando `m'=-s` con la misma interpolación de s:

```text
m_face = (m_L+m_R)/2 + dz*(s_R-s_L)/8
```

Combinarlo con una entalpía interpolada de cara, o diseñar un balance completo con medias celdas explícitas. Esto corrige la compatibilidad del diagnóstico con el cero de entalpía; no convierte automáticamente el residual energético en conservativo. Añadir una escala independiente de ese cero, como `mref*cpref*Tref`, al indicador relativo.

**Validación:** caso fabricado para varios H, malla irregular y cambio de referencia entálpica; defecto absoluto dentro del redondeo. Después descomponer el defecto sobre las soluciones guardadas. Los **923.806 W/m² y 4.254% de H2** en `docs/assets/diffusion/validation.json:151–160` incluyen este diagnóstico: no se sabe cuánto proviene de P0 y cuánto de discretización. Los perfiles comparados no quedan invalidados. No se tocaron esas mediciones.

## 2. P1 — Política: refinamiento satisfecho no equivale a precisión física

**Gravedad media; no es un bug oculto.** `diffusion.py:365–369` refina T/m/V/Y por variación y curvatura. `grid_converged=True` significa que no se solicitan más inserciones, dentro del límite de nodos. La aceptación (`:415–416`) exige residual completo, normalización, positividad tolerada y rama caliente; no un balance integral ni un estimador espacial independiente.

La suma trapezoidal de continuidad y las divergencias difusivas/conductivas son conservativas en sus respectivos operadores. Sin embargo, `m_j*D_upwind(Y_k)` no forma un flujo convectivo telescópico con exportación radial compatible. La identidad continua es `(mY_k+J_k)'+sY_k=omega_k`. Suma(Y)=1 y suma(J)=0 no prueban el balance integral de cada elemento. Las pruebas `tests/test_diffusion_flame.py:68–79`, `:93–95` verifican esas sumas, no los balances elementales integrales.

**Acción:** separar aceptación algebraica, criterio de refinamiento y precisión física; después de P0, incorporar balances elementales/energéticos y una comparación espacial independiente para usos sensibles a entalpía. Un residuo del mismo esquema ya convergido no estima su truncación.

**Validación:** continuación a una malla más fina con mismo caso físico; diferencias de T/Y/Lambda/posición de llama y balances corregidos. Elegir la tolerancia energética según la aplicación, sin confundirla con el 1e−7 algebraico. El umbral `Tmax>Tin,max+100 K` (`diffusion.py:398`) es selección de rama; documentarlo/configurarlo para entradas calientes o extinción, conservando el residual completo.

## 3. P1 — Estancamiento: normalización interior con control débil

**Limitación de robustez, gravedad potencial media; no se reprodujo una llama fallida.** La suma de ecuaciones interiores es:

```text
sum(R_k)*species_scale = m*D(sum(Y)) + div(sum(J)) - sum(omega)
```

Si m=0 y las fuentes/flujos suman cero, desaparece el control advectivo de normalización. La mezcla promediada conserva dependencia fuera del simplex por la corrección con Y_L: `sum(J)=(1-sum(Y_L))*sum(J_star)`. En multicomponente ese control incidental puede desaparecer.

**Prueba ejecutada:** construcción de h2o2 y siete nodos de `tests/test_diffusion_flame.py:13–20`; fijar m[3]=0, multiplicar sólo Y[:,3] por 1.001 y recalcular todas las propiedades.

| Transporte | Suma de filas de especies, Y normalizada | Con suma(Y_3)=1.001 |
|---|---:|---:|
| Mezcla sin Soret | 5.74e−14 | 3.106e−5 |
| Mezcla con Soret | 4.54e−14 | 3.296e−5 |
| Multicomponente sin Soret | 3.97e−14 | 3.80e−14 |
| Multicomponente con Soret | 3.10e−14 | 6.12e−14 |

La prueba acredita pérdida de información en la ecuación sumada, no insensibilidad de todas las ecuaciones individuales ni un resultado aceptado incorrecto. La comprobación final de suma(Y)<1e−7 protege la aceptación.

**Acción/validación:** añadir primero casos con estancamiento exactamente sobre un nodo y entre nodos, mallas desplazadas y perturbación de normalización, para los cuatro modos. Si se reproduce un fallo, evaluar un cierre explícito de suma(Y)=1 sustituyendo una ecuación redundante, o un cierre conservativo cuya suma recupere continuidad. Verificar después la ecuación omitida completa y balances elementales; no proyectar Y al final para esconder residuos. No se propone un cambio de álgebra lineal.

## 4. P2 — Limitación compartida con Cantera: salto del término J·h′ al cruzar m=0

En `diffusion_kernels.py:34–50` el donante cambia correctamente con el flujo. Pero `gradH` usa ese selector aunque el término `Jbar*gradH` no se multiplica por m. Por eso el residual energético puede tener límites distintos en m→0± a paso fijo.

**Prueba ejecutada:** tres nodos con paso .5; T=[300,500,900], Y_1=Y_2=.5, h_1=T, h_2=2T, cp=1.5, J=[.001,−.001] igual en ambas caras; fuentes/conducción cero y escala energética 1:

| m interior | Fila energética |
|---:|---:|
| −1e−12 | −.8000000012 |
| 0 | −.4 |
| +1e−12 | −.3999999994 |

Es una evaluación del kernel, no un estado estacionario. Cambiar `>=` por `>` sólo elige el valor en cero. **No clasificar como error exclusivo del nativo:** Cantera 3.2 también usa upwind para h_k y selecciona el lado derecho cuando u=0. [Flow1D.cpp v3.2.0, grad_hk](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/Flow1D.cpp#L1259-L1264).

**Propuesta no medida:** probar un gradiente de h_k independiente del signo de m o un flujo entálpico conservativo. Validar límites en m=0, perfiles, conservación y robustez en inicialización y cerca de extinción. No atribuirle por anticipado una reducción de iteraciones.

## 5. P2 — Propuesta no medida: reducir coste de malla mediante conservación y orden coherentes

La energía continua equivalente es:

```text
F_h = m*h - lambda*T' + sum(h_k*J_k)
F_h' + s*h = 0
```

La reacción cancela porque h incluye formación. La salida la incluye (`diffusion.py:382`, `:388–389`). La exportación radial puede ser negativa por la referencia entálpica; no es radiación. La combinación discreta de cp*D(T), D(h_k), D(Y) y entalpías de cara no garantiza las identidades de cadena/producto ni un balance integral exacto, incluso después de P0.

**Prototipo acotado propuesto:** mantener T/Y como incógnitas y evaluar energía/especies con flujos de cara compartidos y cuadratura radial compatible con continuidad. No añadir otro término de calor de reacción al balance conservativo de h total. Tratar explícitamente las medias celdas de entrada: la condición convencional actual combina m/Y nodales y J de cara; no imponer además Y=Yfeed.

Cambiar únicamente convección a central no asegura segundo orden:

- `diffusion.py:172` invoca `_corrected_flux_frozen`, cuya corrección usa Y izquierdo (`src/kflame/flame/equations.py:925–943`); frente a Y_face introduce diferencia O(dz) cuando suma(J_star)≠0. Cantera 3.2 usa la misma elección: es discretización deliberada, no bug. [Flow1D.cpp, updateDiffFluxes](https://github.com/Cantera/cantera/blob/v3.2.0/src/oneD/Flow1D.cpp#L448-L466).
- h_k sigue usando upwind. V/T/Y también son upwind en la referencia 3.2. [Flow1D.h v3.2.0, líneas 734–783](https://github.com/Cantera/cantera/blob/v3.2.0/include/cantera/oneD/Flow1D.h#L734-L783).
- Difusión central de tres nodos en malla irregular tiene error local `((hp-hm)/3)*phi''' + O(h²)`. `ratio=3` no garantiza hp−hm=O(h²).

**Comprobación ejecutada de truncación:** phi(z)=z³ en z=1.

| h | Error upwind uniforme | Error central uniforme | Error difusión, hm=h/hp=2h |
|---:|---:|---:|---:|
| .1 | −.290000 | .010000 | .200000 |
| .05 | −.147500 | .002500 | .100000 |
| .025 | −.074375 | .000625 | .050000 |

Son errores locales fabricados, no órdenes globales medidos en llamas. Para una derivada central exacta en cuadráticas sobre malla irregular, usar `(hp*slope_left+hm*slope_right)/(hm+hp)`; una secante simple no basta. Central puro puede oscilar con convección dominante: comparar una reconstrucción limitada/conservativa, preservando el esquema robusto para inicialización.

La familia clásica de esquemas upwind/centrales y las condiciones de flujo total se documentan en [Lutz et al., OPPDIF, SAND96-8243, impreso en mayo de 1997, pp. 8–10](https://digital.library.unt.edu/ark:/67531/metadc696689/m2/1/high_res_d/568983.pdf). Un estudio primario de [Orlac’h et al. sobre conservación de masa/entalpía](https://arxiv.org/pdf/1911.13210) incluye contraflujo, pero sus errores por omitir nanopartículas no son transferibles a estas llamas gaseosas ni prueban aceleración.

**Criterio de éxito:** menos nodos y menor tiempo total a igual error físico de T/Y/Lambda, posición de llama y balances; registrar coste adicional y fallos de robustez. En H2 conviene priorizar difusión diferencial/entalpía; en CH4, una reducción válida de nodos puede importar por sus 53 especies. Son hipótesis. CH4 ya usa 1322 nodos frente a 1338 de Cantera (`timing-review.json:12–13`): la malla final no explica por sí sola el desfase temporal. Los tiempos archivados 33.37/27.63, 1.27/.57 y 4.00/2.22 s permanecen como antecedentes; no se reatribuyen ni se repitieron.

## Pasos siguientes ordenados

1. **Corregir P0 y añadir su prueba mínima.** Separar diagnóstico compatible, error espacial y escala relativa; el agente principal descompone después los resultados guardados.
2. **Añadir comprobaciones de precisión y estancamiento** de los hallazgos 2–4, sin cambiar todavía el núcleo. Confirmar si aparece un fallo real de normalización o de convergencia.
3. **Prototipar un único cambio discreto acotado**, elegido con esas pruebas: gradiente entálpico independiente del donante o flujo conservativo. Mantener la solución actual como control.
4. **Sólo después medir rendimiento**, con la suite requerida y los tres casos físicos. Comparar a igual error físico; no a igual número de nodos o umbrales de refinamiento. Integración, referencias y repeticiones corresponden al agente principal.

## Reproducción mínima del bug P0

Desde la raíz, con Python autorizado y `PYTHONPATH=src`; sin escribir resultados:

```python
import numpy as np
from kflame.flame.diffusion_kernels import assemble_residual
z = np.array([0., .5, 1.]); m = np.array([.5, 0., -.5])
V = np.array([0., 1., 0.]); T = rho = cp = np.ones(3)
Y = np.ones((1, 3)); omega = np.zeros((1, 3))
J = np.zeros((1, 2)); lam = mu = np.ones(2)
dz = np.diff(z); excess = np.zeros(3, dtype=np.int64)
for H in (0., 1., 1e6):
    hk = np.full((1, 3), H)
    r = assemble_residual.py_func(
        m,V,T,Y,-9.,rho,cp,omega,hk,lam,J,mu,dz,
        np.ones(1),np.ones(1),.5,.5,1.,1.,
        1.,1.,1.,1.,1.,1.,excess,False,V,T,Y,1.,False,T)
    faces = .5*(z[:-1]+z[1:]); h = np.full(3,H)
    flux = .5*(m[:-1]*h[:-1]+m[1:]*h[1:])
    radial = 2.*rho*V*h
    iz = np.r_[faces[0],z[1:-1],faces[-1]]
    export = np.trapezoid(np.interp(iz,z,radial),iz)
    whole = m[-1]*h[-1]-m[0]*h[0]+np.trapezoid(radial,z)
    print(H,abs(r).max(),whole,flux[-1]-flux[0]+export)
# Esperado: 0 0 0 0; 1 0 0 .25; 1000000 0 0 250000
```

**Archivo cambiado por esta revisión:** `docs/reviews/diffusion-mathematics.md`. Ningún cambio al solver o a evidencia de medición. Las comprobaciones nuevas no sustituyen la suite completa ni las comparaciones archivadas.
