# Lensing Gravitacional en Andromeda

Documento de estado del módulo de lensing gravitacional del proyecto **Andromeda**
(ray-tracing en fondos curvos). Resume las implementaciones existentes y traza el
plan para expandirlo hasta poder **generar imágenes realistas de lensing con
cualquier métrica**.

---

## 1. Arquitectura actual

El pipeline de lensing sigue un esquema de *backward ray-tracing*:

```
Métrica (BH o LensMetric)      →  define g_μν, geodésicas y hooks Numba
        │
Detector (image_plane)          →  genera 4-momentos nulos por píxel
        │
LensImage.create_photons()      →  construye la grilla de fotones (α, β)
        │
parallel.trace_* (mode=lensing) →  integra cada geodésica nula
        │    (integrator / _solvers / _numba_kernels)
        │
SourcePlane                     →  intersección con el plano fuente en D_LS
        │
light_profiles (Gaussian/Sérsic)→  evalúa brillo I(x,y) en el plano fuente
        │
visual.py                       →  stretch asinh, RGB, ruido, starfield
```

Puntos de extensión limpios:

- **Métricas nuevas**: cumplir el *duck-type* `metric(x)`, `geodesics(q, λ)`,
  `EH`, y los hooks Numba `_rhs_nb`, `_metric_nb`, `_omega_nb`. Ver
  [sis.py](../scr/lens_metrics/sis.py) líneas 109-187 como referencia mínima.
- **Perfiles de fuente nuevos**: añadir un `KIND_*` y el caso en
  [_eval_profile_nb](../scr/sources/light_profiles.py) (líneas 67-89).
- **Modos de trazado**: `_MODE_CODES = {"no_doppler":0, "doppler":1, "shadow":2, "lensing":3}`
  en [parallel.py](../scr/common/parallel.py).

---

## 2. Implementaciones de lensing existentes

### 2.1 Métricas soportadas

| Métrica | Archivo | Régimen | Notas |
|---|---|---|---|
| Schwarzschild | [schwarzschild.py](../scr/black_holes/schwarzschild.py) | Fuerte | Punto-masa, M=1 |
| Kerr | [kerr.py](../scr/black_holes/kerr.py) | Fuerte | Spin a, frame-dragging |
| Kerr-MOG | [kerr_mog.py](../scr/black_holes/kerr_mog.py) | Fuerte | Gravedad modificada (α, c) |
| Scalar-hair BH | [scalar_hair_BH.py](../scr/black_holes/scalar_hair_BH.py) | Fuerte | Hair escalar |
| Schwarzschild numérica | [num_schwarzschild.py](../scr/black_holes/num_schwarzschild.py) | Fuerte | Datos tabulados |
| SIS (Singular Isothermal Sphere) | [sis.py](../scr/lens_metrics/sis.py) | Débil | `sigma_v`, axisimétrica |
| SIE (Singular Isothermal Ellipsoid) | [sie.py](../scr/lens_metrics/sie.py) | Débil | `sigma_v`, `q_ax` (aplanamiento) |

### 2.2 Núcleo de trazado

- **Integradores**: LSODA, DOP853, RK45 (SciPy) + RK45 propio con Brent
  bisection para eventos + Verlet (splitting 2º orden).
  Ver [integrator.py](../scr/common/integrator.py) líneas 35-115 y
  [_solvers.py](../scr/common/_solvers.py).
- **Kernels Numba**: hot-paths compilados AOT para RHS de geodésica, métrica y
  frecuencia angular. Ver [_numba_kernels.py](../scr/common/_numba_kernels.py).
- **Paralelización**: `trace_threads_nb` (Numba `prange`) y `trace_parallel`
  (multiprocessing.Pool). Ver [parallel.py](../scr/common/parallel.py).
- **Eventos**: horizonte (terminal, dir=-1), escape asintótico (terminal, dir=+1),
  cruce de plano ecuatorial (no-terminal).
- **Límites de rayo**: `set_ray_bounds(r_escape, final_lmbda)` en
  [common.py](../scr/common/common.py) líneas 36-47 — ajusta el radio de escape
  y el parámetro afín máximo para el round-trip a través del lente.

### 2.3 Fuentes y detectores

- **Perfiles de luz** [light_profiles.py](../scr/sources/light_profiles.py):
  - `Gaussian(x0, y0, sigma, I0)`
  - `Sersic(x0, y0, R_e, n, I_e, ell, pa)` (Ciotti & Bertin 1999 para b_n).
- **Plano fuente** [SourcePlane](../scr/common/lens_image.py) en z = -D_LS
  perpendicular al eje óptico.
- **Detector** [image_plane.py](../scr/detectors/image_plane.py) — plano imagen
  con inclinación `iota` (ι=π/2 ⇒ proyección paralela, observador en infinito;
  natural para ver un anillo de Einstein simétrico).

### 2.4 Visualización

En [visual.py](../scr/common/visual.py):
- Stretch asinh, composición RGB, ruido fotográfico, starfield simulado,
  proyección de perfil sobre detector.

### 2.5 Ejemplos funcionando end-to-end

| Script | Qué valida |
|---|---|
| [ex9.einstein_ring.py](../ex9.einstein_ring.py) | Anillo de Einstein con Schwarzschild + Gaussiana. Radio predicho `b_ring = 2√(M·D_LS)`. |
| [ex10.cosmic_horseshoe.py](../ex10.cosmic_horseshoe.py) | "Cosmic Horseshoe" compuesto (SIS + Sérsic desplazado) con estética Hubble, ruido Poisson y starfield. |
| [ex11.lensing_trajectory.py](../ex11.lensing_trajectory.py) | Diagrama de trayectorias de geodésicas nulas a través de un lente SIS, ángulos de deflexión y parámetros de impacto. |
| [ex12.kerr_lensing.py](../ex12.kerr_lensing.py) | Comparativa Kerr (a=0.9) vs Schwarzschild: asimetría por frame-dragging en campo fuerte. |
| [ex13.einstein_cross.py](../ex13.einstein_cross.py) | SIS vs SIE (q=0.55): anillos simétricos vs configuración de imágenes múltiples (cruz de Einstein). |
| [ex14.realistic_lensing.py](../ex14.realistic_lensing.py) | **Realismo observacional**: SIE asimétrico + múltiples fuentes lensadas + galaxias de fondo sin lensing + starfield denso + ruido fotográfico HST-like. Anillo incompleto, irregularidades e inhomogeneidades. |
| [ex15.broken_ring_lensing.py](../ex15.broken_ring_lensing.py) | **Imágenes múltiples asimétricas**: SIE + fuente primaria altamente asimétrica (ell=0.45, I_e=1.4, R_e=0.11×b) para crear arcos asimétricos bien visibles, NO un anillo. Emula Einstein Cross-like configurations. |
| [ex16.deep_field_lensing.py](../ex16.deep_field_lensing.py) | **Deep field Hubble-like**: SIE con dos núcleos de lente (merging) + 35 galaxias de fondo con colores variados (rojo, naranja, amarillo, azul) + 250 estrellas + cielo gris visible. Emula campos profundos reales (SDSS, Abell). |
| [ex17.cheshire_cat.py](../ex17.cheshire_cat.py) | **Cheshire Cat (BinarySIE real)**: lente binario (dos SIEs superpuestos) donde la asimetría del anillo proviene de la geometría física del potencial, NO de post-proceso. Campo de fondo denso (180 galaxias) con luminosidades ley-de-potencias, tamaños log-normal, 20% edge-on y paleta de 5 colores simulando múltiples redshifts. Emula SDSS J1038+4849. |

### 2.6 Diagnósticos ya disponibles

- Curvas críticas de esfera de fotones (Bardeen 1973 en Schwarzschild,
  `photon_sphere_critical_curve()` en [kerr.py](../scr/black_holes/kerr.py)).
- Verificación del constraint Hamiltoniano (`verify_Hamiltonian()` en
  [common.py](../scr/common/common.py) líneas 420-448).
- Comparación de área de sombra (shoelace analítico vs componentes conexas
  numéricas).

---

## 3. Estado y limitaciones para "cualquier métrica con realismo"

El núcleo de ray-tracing **ya es agnóstico a la métrica**: cualquier espaciotiempo
que exponga los hooks Numba puede ser trazado. Las brechas para *realismo*
observacional son:

**Observacional / instrumental**
- No hay PSF (Airy / Gaussiana / Moffat) ni convolución del plano imagen.
- Ruido limitado: solo Poisson + grano gaussiano; falta ruido de lectura,
  background de cielo, cosmic rays.
- Sin *pixelation blur* ni pérdida de resolución coherente.
- Sin calibración fotométrica (magnitudes AB, zero-point, filtros HST/JWST).

**Física de lensing**
- Sin convergencia/shear externos (`κ_ext`, `γ_ext`).
- Un único plano fuente; no hay **multi-plane lensing** ni cosmología FRW
  (D_L, D_S, D_LS a partir de redshifts y H_0).
- Sin perfiles de masa clave: NFW, Hernquist, Jaffe, Power-law elíptico,
  Chameleon; ni potenciales generales no-axisimétricos.
- Lentes débiles (SIS/SIE) no verifican `|Φ| ≪ 1` ni degradan elegantemente
  a campo fuerte.
- Sin conversión entre parámetros físicos (σ_v, M_200, c_NFW) y parámetros
  efectivos (radio de Einstein θ_E, κ).

**Diagnóstico**
- No se calculan curvas críticas en el plano imagen ni cáusticas en el plano
  fuente.
- Sin mapas de magnificación μ(α,β) ni superficies de time-delay
  (función de Fermat) — bloqueante para reproducir cuásares multi-imagen
  con retardos temporales.
- Sin integración de flujo sobre apertura ni fotometría por imagen lensada.

**Fuentes**
- Solo Gaussiana y Sérsic. Falta: disco+bulbo, ley de potencias elíptica,
  fuentes no-paramétricas (tablas/imágenes reales como input), cuásares
  puntuales.

---

## 4. Plan de expansión

Objetivo: **dada cualquier métrica que exponga los hooks estándar, producir
una imagen lensada realista del cielo**. Se prioriza el realismo manteniendo
el backend de trazado actual.

Fases ordenadas por dependencia. Cada fase produce un artefacto demostrable
y usa lo ya existente antes de introducir código nuevo.

### Fase A — Cosmología y unidades físicas

**Meta**: poder especificar un lente por (z_L, z_S) y propiedades físicas
(σ_v, M, r_s) en lugar de M geometrizado.

- Nuevo módulo `scr/common/cosmology.py`:
  - FRW plano Λ-CDM por defecto (H_0, Ω_m, Ω_Λ).
  - Funciones: `angular_diameter_distance(z)`, `D_LS(z_L, z_S)`,
    `critical_surface_density(z_L, z_S)`.
- Factorizar `set_ray_bounds` → `set_geometry(z_L, z_S, cosmology=...)`
  devolviendo `D_L`, `D_S`, `D_LS` en M geometrizados.
- Parámetros efectivos por métrica:
  - SIS/SIE: `theta_E(sigma_v, z_L, z_S)`.
  - Schwarzschild: `theta_E = √(4GM/c² · D_LS/(D_L·D_S))`.
- Archivos afectados: nuevo `cosmology.py`, [common.py](../scr/common/common.py),
  [sis.py](../scr/lens_metrics/sis.py), [sie.py](../scr/lens_metrics/sie.py).

**Verificación**: reproducir θ_E de SDSS J1004+4112 (σ_v≈352 km/s,
z_L=0.68, z_S=1.73) dentro de 1%.

### Fase B — Catálogo de métricas de lensing realistas

Añadir a `scr/lens_metrics/`:

1. **NFW** (`nfw.py`): perfil de halo CDM. Parámetros (M_200, c).
   Deflexión analítica de Wright & Brainerd (2000).
2. **Power-law Ellipsoid (PEMD / EPL)** (`pemd.py`): generalización de SIE
   con índice logarítmico `γ` (γ=2 → SIE). Ampliamente usado en SLACS.
3. **Convergencia + shear externos** (`external.py`): término aditivo κ_ext,
   γ_ext con ángulo φ_γ. Se combina con cualquier lente base.
4. **Composite (galaxy + halo)** (`composite.py`): lente base + `external`
   → suma de deflexiones con misma API.

Cada archivo expone `LensMetric` con `_rhs_nb`, `_metric_nb`, `_omega_nb` y
test unitario en `scr/lens_metrics/test_*.py` siguiendo el patrón de
[test_sis_deflection.py](../scr/lens_metrics/test_sis_deflection.py).

**Verificación**: test por métrica comparando deflexión numérica vs
analítica en 3 radios de impacto (rtol=1e-3).

### Fase C — Diagnóstico de lensing

Nuevo `scr/common/lens_diagnostics.py`:

- `jacobian_map(lens, detector)` — matriz A = ∂β/∂θ por píxel vía
  diferenciación numérica de la grilla ya trazada.
- `magnification_map(lens, detector)` → μ = 1/det(A).
- `critical_curves(lens, detector)` — contornos de det(A)=0 en plano imagen.
- `caustics(lens, detector)` — mapeo de curvas críticas al plano fuente.
- `time_delay(lens, source_plane, z_L)` — función de Fermat
  τ = (1+z_L)/c · D_L·D_S/D_LS · [½(θ-β)² - ψ(θ)].

**Verificación**: para SIS cáustica analítica es un punto; para SIE es una
astroide. Overlay sobre [ex13.einstein_cross.py](../ex13.einstein_cross.py).

### Fase D — Realismo instrumental

Nuevo `scr/detectors/instrument.py` (o extender [image_plane.py](../scr/detectors/image_plane.py)):

- Convolución por PSF (Gaussiana, Moffat, o tabulada desde un FITS).
- Pixelación con *sub-pixel integration* (el trazado actual ya puede
  supersamplear subiendo `x_pixels`; añadir binning configurable).
- Modelo de ruido: Poisson del objeto + sky background + read noise + dark current.
- Calibración fotométrica: zero-point AB, tiempo de exposición, filtros
  (F814W, F160W, etc.) con SED de fuente.
- Salida en FITS (`astropy.io.fits`) además de PNG/NPY.

**Verificación**: recrear visualmente [cosmic_horseshoe.png](../images/cosmic_horseshoe.png)
pero con PSF HST F814W (FWHM ~0.1") y ruido realista; comparar SNR por píxel
del arco vs referencia observacional.

### Fase E — Fuentes extendidas realistas

Extender [light_profiles.py](../scr/sources/light_profiles.py):

- `Bulge + Disk` (Sérsic n=4 + exponencial n=1).
- `PointSource` (para cuásares, delta + PSF).
- `ImageSource`: cargar array 2-D (p.ej. imagen real de una galaxia deep-field
  sin lensing) y usarlo como `I(x,y)` interpolado.
- Múltiples fuentes por `SourcePlane` (lista de perfiles con brillo aditivo).

Registro en `_eval_profile_nb` con nuevos `KIND_*`.

**Verificación**: lensear una imagen HST real de una galaxia de fondo a través
de SIE + shear; visualmente indistinguible de un sistema Einstein-ring real
(p.ej. SDSS J1038+4849 "smiley").

### Fase F — Multi-plane lensing (opcional, avanzado)

- `SourcePlane` → lista ordenada de planos con (z_i, deflector_i).
- Iterar trazado: al cruzar cada plano aplicar deflexión acumulada antes
  de continuar hacia la siguiente.
- Requiere Fase A (cosmología) y Fase B (lentes componibles).

**Verificación**: reproducir un sistema doble conocido (e.g. lente en z=0.3
+ lente en z=0.7 + fuente en z=2.0) con imágenes predichas por glafic/lenstronomy.

---

## 5. Orden sugerido de ejecución

```
A (cosmología)   →   B (NFW, PEMD, shear)   →   C (diagnósticos)
                                 │
                                 ↓
                     D (PSF, ruido, FITS)   →   E (fuentes realistas)
                                                          │
                                                          ↓
                                                  F (multi-plane)
```

Cada fase añade un ejemplo numerado (ex14, ex15, ...) que la demuestra y sirve
como test de regresión visual.

---

## 6. Criterios de "realismo suficiente"

Una imagen generada se considerará realista cuando:

1. **Geometría**: θ_E concuerda con predicción analítica (<1%) para todas las
   métricas de la Fase B.
2. **Fotometría**: magnitudes por imagen lensada consistentes con
   magnificación del `magnification_map` (<5%).
3. **Instrumentación**: SNR por píxel y FWHM efectivo consistentes con la
   configuración HST/JWST emulada.
4. **Visual**: indistinguible por inspección de un sistema real cuando se
   usan fuentes reales (Fase E).

---

## 7. Archivos críticos a tocar

- Núcleo que NO se modifica (estable): [integrator.py](../scr/common/integrator.py),
  [_solvers.py](../scr/common/_solvers.py), [_numba_kernels.py](../scr/common/_numba_kernels.py),
  [parallel.py](../scr/common/parallel.py).
- Núcleo a extender: [common.py](../scr/common/common.py),
  [lens_image.py](../scr/common/lens_image.py),
  [image_plane.py](../scr/detectors/image_plane.py),
  [light_profiles.py](../scr/sources/light_profiles.py),
  [visual.py](../scr/common/visual.py).
- Nuevos módulos: `scr/common/cosmology.py`, `scr/common/lens_diagnostics.py`,
  `scr/detectors/instrument.py`, `scr/lens_metrics/{nfw,pemd,external,composite}.py`.
