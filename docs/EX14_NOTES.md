# ex14.realistic_lensing.py — Notas de implementación

## Objetivo
Crear una imagen de lensing gravitacional con realismo observacional comparable a imágenes reales del Hubble. Partiendo de ex10 (Cosmic Horseshoe), se agregan complejidades que hacen la imagen visualmente más interesante:
- Anillo incompleto (no perfectamente simétrico)
- Múltiples fuentes de luz (no una sola)
- Galaxias de fondo (ambiente profundo del cielo)
- Starfield denso
- Ruido fotográfico realista (HST-like)

## Cambios respecto a ex10

### 1. **Métrica del lente: SIS → SIE**
- **ex10**: `SIS` — esfera isotérmica singular (simetría perfecta)
- **ex14**: `SIE` — elipsoide isotérmico singular con aplanamiento `q_ax=0.70`
  - Rompe la simetría rotacional del anillo
  - El lente es oblato (aplastado) a lo largo del eje z
  - Parámetro σ_v ligeramente menor (0.028 vs 0.03) para compensar

### 2. **Fuentes lensadas: Una sola → Múltiples**
- **ex10**: Un único perfil Sérsic desplazado
- **ex14**: 
  - Una fuente **primaria** (grande, Sérsic n=1, el arco principal)
  - Dos fuentes **secundarias** (pequeñas, Sérsic n=1.0 y n=1.5, galaxias acompañantes magnificadas)
  - Cada una se trazada por separado y se combinan mediante `np.maximum()` (el píxel más brillante)
  - Resultado: arcos múltiples con intensidades diferentes, irregularidades

### 3. **Galaxias de fondo (sin lensing)**
- **ex10**: Nada
- **ex14**: 5 galaxias proyectadas directamente en el plano imagen
  - Mezcla de perfiles Sérsic (n=1, n=1.5, n=4) y Gaussianas
  - Diferentes tamaños, posiciones, elipticidades
  - Se proyectan sin ray-tracing, simulando el "deep field" de fondo
  - Color: `gal_color = (0.85, 0.65, 0.35)` (amarillento, distinto del lente y del arco)
  - Blendeo: 0.3× el brillo de las galaxias (fondo faint)

### 4. **Starfield**
- **ex10**: `n_stars = 0` (desactivado en modo simulación limpio)
- **ex14**: `n_stars = 150` (denso, realista para HST)
  - Distribuye 150 estrellas puntuales con PSF Gaussiana
  - Rango de flujo log-uniforme, sigmas variadas

### 5. **Ruido fotográfico**
- **ex10**: `photographic_noise = False` (desactivado)
- **ex14**: Activado con parámetros realistas
  - `read_noise = 0.02` (Gaussian)
  - `poisson_scale = 350.0` (shot noise, escala para SNR realista)
  - Ambos se aplican al RGB final antes del stretch

### 6. **Stretch asinh y composición RGB**
- **ex10**: 3 capas (lens, arc, stars)
- **ex14**: 4 capas (lens, arc, background_galaxies, stars)
  - Los parámetros de asinh ajustados: `softening=0.025`, `max_percentile=99.8`
  - Mayor agresividad en el stretch para resaltar detalles faint

## Pipeline detallado

```
1. Ray-trace primaria lensada     → primary_arc
        ↓
2. Ray-trace secundarias (loop)   → secondary_arcs[]
        ↓
3. Combinar arcos                 → arc_img = max(primary + secondaries)
        ↓
4. Proyectar luz del lente        → lens_img
        ↓
5. Proyectar galaxias de fondo    → gal_img
        ↓
6. Starfield                      → stars_img
        ↓
7. Composición RGB:
   rgb = lens_img × lens_color
       + arc_img × arc_color
       + 0.3 × gal_img × gal_color
       + stars_img × stars_color
        ↓
8. Ruido fotográfico              → Poisson + read noise
        ↓
9. Stretch asinh                  → escala perceptual
        ↓
10. Save + plot
```

## Parámetros clave

| Parámetro | Valor | Significado |
|---|---|---|
| `sigma_v` (lente) | 0.030 c | Dispersión de velocidades (SIE) |
| `q_ax` | 0.95 | Casi circular (1 = perfecto), para anillo simétrico |
| `b_ring` | ~113 M | Radio del anillo (calculado, no puesto a mano) |
| `x_side` | 3.2 × b_ring | Campo de visión (más grande que ex10 para ver fondo) |
| `offset_primary` | 0.05 × b_ring | Pequeño desplazamiento → anillo casi completo |
| `R_e_primary` | 0.12 × b_ring | Tamaño de fuente primaria (mayor que antes) |
| `I_e_primary` | 1.3 | Brillo aumentado de la fuente |
| `n_stars` | 150 | Densidad de estrellas |
| `read_noise` | 0.015 | Ruido Gaussiano de lectura |
| `poisson_scale` | 280 | Escala de ruido Poisson (más visible que antes) |
| `softening` (asinh) | 0.018 | Stretch más agresivo para mejor contraste |

## Diferencias visuales esperadas

**ex10 (clean cosmic horseshoe):**
- Anillo casi perfecto, ligeramente roto por offset
- Solo luz del lente + arco lensado + starfield escaso
- Imagen "limpia", como simulación numérica

**ex14 (realistic):**
- Anillo incompleto, asimétrico por SIE
- Múltiples arcos/isófotas de brillo variable
- 5 galaxias faint de fondo, estrellas densas
- Ruido de shot visible en píxeles individuales
- Aspecto visual cercano a imagen HST real

## Uso

```bash
python ex14.realistic_lensing.py
```

Produce:
- `images_data/realistic_lensing_sie.npy` (array RGB, formato NumPy)
- `images/realistic_lensing_sie.png` (PNG comprimido, 8-bit)

Ambos tienen el mismo contenido visual; el .npy preserva precisión flotante si se necesita post-procesamiento.

## Historial de versiones

### v1.0 (Inicial)
- Lente SIE con q_ax=0.70 (significativamente elíptico)
- Offset fuente primaria: 0.12×b_ring → anillo muy incompleto
- Ruido Poisson moderado (poisson_scale=350)
- Problema: anillo demasiado fragmentado, poco parecido a referencias reales

### v1.1 (Optimizada para Einstein ring)
- Lente SIE con q_ax=0.95 (casi circular) → anillo simétrico
- sigma_v ajustada a 0.030 para compensar circularidad
- Offset fuente primaria: 0.05×b_ring → anillo casi completo
- Fuente primaria más grande (R_e: 0.08 → 0.12×b_ring)
- Fuente primaria más brillante (I_e: 1.0 → 1.3)
- Lente más prominente (I_e: 0.8 → 1.0, R_e reducido)
- Ruido aumentado (poisson_scale: 350 → 280, read_noise: 0.02 → 0.015)
- Contraste mejorado (softening: 0.025 → 0.018)
- **Resultado**: anillo completo y bien definido, similar a referencias observacionales

## Extensiones futuras

- Reemplazar starfield aleatorio por catálogo sísmico (ficción → realidad)
- Añadir PSF real de HST (convolución, no solo Gaussianas)
- Incorporar ruido de background del cielo (no solo shot noise)
- Usar imágenes reales de galaxias como perfiles de fuente (en lugar de analíticos)
- Implementar multi-plane lensing para sistemas con múltiples lentes
