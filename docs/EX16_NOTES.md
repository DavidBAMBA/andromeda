# ex16.deep_field_lensing.py — Deep Field Strong Lensing

## Objetivo

Generar una imagen de lensing gravitacional que emule observaciones reales del Hubble en campos profundos,
con:
- Anillo de Einstein difuso y menos prominente
- Múltiples núcleos de lente (sistema merging/binario)
- Fondo poblado de 30+ galaxias con colores y morfologías variadas
- Starfield denso (250 estrellas)
- Cielo de fondo visible (gris oscuro, no negro puro)
- Paleta cromática realista simulating múltiples redshifts

## Cambios respecto a ex15

### 1. **Lente: Dos nuclei en lugar de uno**

**ex15:**
```python
# Un único perfil
lens_light_profile = Sersic(x0=0.0, y0=0.0, R_e=0.28*b_ring, ...)
```

**ex16:**
```python
# Dos núcleos separados (merging/binary system)
lens_nucleus_1 = Sersic(x0=-0.08*b_ring, y0=0.10*b_ring,
                         R_e=0.15*b_ring, n=4.0, I_e=0.9, ...)  # Northernl

lens_nucleus_2 = Sersic(x0=0.08*b_ring, y0=-0.08*b_ring,
                         R_e=0.12*b_ring, n=3.5, I_e=0.75, ...)  # Southern
```

**Efecto:** Simula una galaxia en proceso de merger, como se ve en sistemas reales
(e.g., NGC 6240, Centaurus A).

### 2. **Fondo: 5 galaxias → 35 galaxias con colores variados**

**ex15:**
```python
background_galaxies = [
    # 5 galaxias, todas con colores similares
    Sersic(...), Sersic(...), Gaussian(...), ...
]
```

**ex16:**
```python
def generate_background_galaxies(b_ring, n_galaxies=35):
    """Generate random population with diverse colors"""
    colors_palette = [
        (0.9, 0.4, 0.2),   # Red elliptical (z~1)
        (1.0, 0.6, 0.2),   # Orange elliptical (z~0.8)
        (1.0, 0.8, 0.2),   # Yellow elliptical (z~0.6)
        (0.8, 0.8, 0.8),   # White/blue elliptical (z~0.4)
        (0.3, 0.6, 1.0),   # Blue spiral (z~0.3, starburst)
    ]
    # Genera 35 galaxias con posiciones, tamaños, formas y colores aleatorios
```

**Cambios:**
- **35 galaxias vs 5**: cubrimiento mucho más denso del campo
- **Colores variados**: Red (z~1), Orange (z~0.8), Yellow (z~0.6), White (z~0.4), Blue (z~0.3)
  simulating diferentes redshifts
- **Morfologías aleatorias**: Sérsic n = 1.0, 1.5, 2.5, 4.0 (discos y bulges)
- **Posiciones aleatorias**: distribución uniforme en todo el detector
- **Brightnesses variadas**: I_e ∈ [0.05, 0.25], más faint que en ex15

### 3. **Anillo más difuso**

**ex15:**
```python
primary_source = Sersic(..., R_e=0.09*b_ring, I_e=1.1, ell=0.45, ...)
```

**ex16:**
```python
primary_source = Sersic(..., R_e=0.10*b_ring, I_e=0.8, ell=0.35, ...)  # Menos brillante, menos elíptico
```

**Efecto:**
- Menor brillo (I_e: 1.1 → 0.8) → anillo menos dominante, más "sumergido" en el fondo
- Menor elipticidad (ell: 0.45 → 0.35) → anillo más suave, menos "roto"
- Resultado: anillo visible pero difuso, como en campos profundos reales

### 4. **Composición RGB con capas de color separadas**

**ex15:**
```python
rgb = (lens_img[..., None] * lens_color
       + arc_img[..., None] * arc_color
       + gal_img[..., None] * gal_color
       + stars_img[..., None] * stars_color)
```

**ex16:**
```python
# Fondo cielo (gray oscuro)
rgb[..., :] = bg_sky_color

# Capas separadas por componente
rgb[..., 0] += 0.6 * arc_img * arc_color[0]    # R del arco
rgb[..., 1] += 0.6 * arc_img * arc_color[1]    # G del arco
rgb[..., 2] += 0.6 * arc_img * arc_color[2]    # B del arco

# Nuclei con sus propios colores
rgb[..., 0] += lens_img * nucleus1_color[0] * 0.8
rgb[..., 1] += lens_img * nucleus1_color[1] * 0.8
...

# Galaxias de fondo con capas RGB separadas
rgb[..., 0] += 0.4 * bg_img_red
rgb[..., 1] += 0.4 * bg_img_green
rgb[..., 2] += 0.4 * bg_img_blue
```

**Ventaja:** Permite que cada componente tenga su propia paleta cromática,
simulando correctamente la mezcla de luz de objetos a diferentes redshifts.

### 5. **Cielo de fondo no puro negro**

**ex15:**
```python
# Implícito: fondo negro (0, 0, 0)
```

**ex16:**
```python
bg_sky_color   = (0.03, 0.02, 0.05)       # Dark blue-gray
# Se añade en cada píxel como base
rgb[..., :] = bg_sky_color  # Fondo inicial
```

**Efecto:** El cielo no es puro negro, sino un gris muy oscuro levemente azulado,
como en imágenes reales del Hubble donde hay siempre algo de fondo de cielo.

### 6. **Starfield más denso**

**ex15:**
```python
n_stars = 150
```

**ex16:**
```python
n_stars = 250  # 67% más denso
```

## Parámetros clave de ex16

| Parámetro | Valor | Significado |
|---|---|---|
| **Lente** |
| σ_v | 0.028 c | Dispersión de velocidades |
| q_ax | 0.72 | Aplanamiento del lente |
| Nucleus 1 | R_e=0.15, n=4.0, (x0,y0)=(-0.08, +0.10)×b | Golden, northern |
| Nucleus 2 | R_e=0.12, n=3.5, (x0,y0)=(+0.08, -0.08)×b | Slightly redder, southern |
| **Fuente lensada** |
| I_e primaria | 0.8 | Más diffuse que ex15 |
| ell primaria | 0.35 | Menos asimetría |
| **Fondo** |
| # Galaxias background | 35 | vs 5 en ex15 |
| Colores | 5 tipos | Red, Orange, Yellow, White/Blue, Blue |
| I_e galaxias | 0.05-0.25 | Faint como en deep fields |
| **Ambiente** |
| n_stars | 250 | Starfield denso |
| bg_sky_color | (0.03, 0.02, 0.05) | Gray oscuro, no negro |
| poisson_scale | 300 | Ruido controlado |

## Pipeline visual

```
Base: cielo gris oscuro (0.03, 0.02, 0.05)
  ↓
+ Anillo azul difuso (60% peso)
  ↓
+ Dos núcleos de lente (nuclei 1 y 2 con colores distintos)
  ↓
+ 35 galaxias de fondo (40% peso, colores variados)
  ↓
+ 250 estrellas puntuales
  ↓
+ Ruido fotográfico (Poisson + read noise)
  ↓
+ Stretch asinh (softening=0.015)
```

## Sistemas reales emulados

Esta imagen intenta reproducir características de:

- **SDSS J0029-0055**: Anillo casi completo, fondo poblado, dos lentes
- **Abell 1689**: Cluster lensing, múltiples arcos, fondo profundo
- **Abell 2218**: Strong lensing en cluster, muchas imágenes múltiples
- **Bullet Cluster**: Merging system, estructura visible en el lente

## Uso

```bash
python ex16.deep_field_lensing.py
```

Produce:
- `images_data/deep_field_lensing.npy` (array RGB flotante)
- `images/deep_field_lensing.png` (PNG 8-bit comprimido)

Tiempo estimado: ~2-5 minutos dependiendo de CPU (35 galaxias de fondo × ray-tracing).

## Extensiones futuras

1. **Catálogo de redshifts**: Usar z ∈ [0.2, 2.0] para asignar colores según SED
2. **PSF real de HST**: Convolucionar con PSF punto difracción (star diffraction pattern)
3. **Observational realism**: Pixelización, dithering patterns, cosmic rays
4. **Multi-object spectroscopy**: Indicar qué galaxias son el lente vs fuentes
5. **Substructure lensing**: Pequeños halos perturban el anillo localmente
