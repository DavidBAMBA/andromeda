# ex15.broken_ring_lensing.py — Anillo de Einstein "Roto"

## Objetivo

Generar una imagen de lensing gravitacional donde el anillo de Einstein no es completo,
sino que tiene un sector oscuro o "borrado", como en sistemas reales observados con el Hubble.

## El problema: ¿Por qué un sector del anillo desaparece?

En lensing gravitacional, la magnitud del anillo (brillo en el plano imagen) depende de:

1. **El brillo de la fuente** en el plano fuente
2. **La magnificación** (derivada del Jacobiano de la ecuación de lensing)

Si la fuente en el plano fuente es **asimétricamente distribuida** (por ejemplo, una galaxia
elíptica muy alargada, o una galaxia con un "dust lane"), entonces la magnitud del anillo
varía alrededor del azimut. Las regiones del anillo que corresponden a partes oscuras de
la fuente aparecerán más débiles o invisibles — como si el anillo tuviera un "hueco".

## Implementación en ex15

### Cambio clave: Fuente primaria altamente asimétrica

**ex14:**
```python
primary_source = Sersic(x0=0.12*b_ring, y0=0.04*b_ring,
                        R_e=0.08*b_ring, n=1.0, I_e=1.0,
                        ell=0.20, pa=pi/4)  # ell=0.20 (moderadamente elíptica)
```

**ex15 (v1.1 - optimizada):**
```python
primary_source = Sersic(x0=0.10*b_ring, y0=0.06*b_ring,
                        R_e=0.11*b_ring, n=1.0, I_e=1.4,
                        ell=0.45, pa=pi/3.5)  # ell=0.45 (FUERTEMENTE elíptica!)
```

### Parámetros de asimetría

| Parámetro | ex14 | ex15 v1.0 | ex15 v1.1 (actual) | Efecto |
|---|---|---|---|---|
| `ell` (ellipticity) | 0.20 | 0.45 | 0.45 | Fuente muy alargada → arcos asimétricos |
| `R_e` (semeje mayor) | 0.08×b_ring | 0.09×b_ring | **0.11×b_ring** | Fuente más grande → arcos más brillantes |
| `I_e` (brillo) | 1.0 | 1.1 | **1.4** | Fuente más brillante (+27%) |
| `pa` (position angle) | π/4 | π/3.5 | π/3.5 | Ángulo de orientación de la fuente |
| `x0` (offset x) | 0.12×b_ring | 0.10×b_ring | 0.10×b_ring | Offset desde eje |
| `y0` (offset y) | 0.04×b_ring | 0.06×b_ring | 0.06×b_ring | Desplazamiento vertical |

### Resultado esperado (v1.1)

La alta elipticidad (ell=0.45) + mayor brillo (I_e=1.4) + tamaño mayor (R_e=0.11×b_ring)
produce:

- **Arcos brillantes y visibles** en los sectores que mapean a las regiones anchas de la fuente
- **Arcos débiles** donde la fuente es muy delgada
- **Asimetría clara**: no se ve como un anillo continuo, sino como imágenes múltiples separadas
- **Efecto visual**: "Multi-image lensing" asimétrico, no "Einstein ring"
- **Comparación con ref**: Similar a Einstein Cross (SDSS J0924+0219), con imágenes azules
  asimétricamente distribuidas alrededor de un lente central

## Comparación visual esperada

| Sistema | Fuente | Anillo resultante |
|---|---|---|
| **ex9** (Einstein ring) | Gaussiana circular | Anillo perfecto, completo |
| **ex10** (Cosmic horseshoe) | Sérsic, offset grande | Anillo casi completo, ligeramente quebrado |
| **ex14** (Realistic) | Sérsic asimétrica moderada (ell=0.20) | Anillo incompleto, múltiples arcos |
| **ex15** (Broken ring) | Sérsic ALTAMENTE asimétrica (ell=0.45) | Anillo con sector claro oscuro/faltante |

## Física subyacente

En observaciones reales, hay varias razones por las que los anillos se ven "rotos":

1. **Fuentes elípticas**: galaxias de fondo con forma de disco o barrada
2. **Dust lanes**: polvo interestelar en la fuente que absorbe luz
3. **Substructura**: clúmps o regiones discretas de formación estelar
4. **Magnificación variable**: el Jacobiano del lensing cambia con posición

**ex15 modela el caso 1**: una fuente fuertemente elíptica cuya luz no se distribuye uniformemente.

## Nombre de la imagen

**v1.0:** `broken_ring_lensing` — sugiere un anillo roto/incompleto

**v1.1:** `asymmetric_multi_image_lensing` — refleja mejor que son imágenes múltiples
asimétricamente distribuidas, NO un anillo continuo. Este nombre es más preciso
físicamente y evita la confusión con "Einstein ring" (que implica simetría).

## Uso

```bash
python ex15.broken_ring_lensing.py
```

Produce:
- `images_data/asymmetric_multi_image_lensing.npy`
- `images/asymmetric_multi_image_lensing.png`

## Extensiones futuras

Para hacer el efecto aún más realista:

1. **Agregar dust**: modelar una región oscura explícita en el plano fuente
   - Restar una componente Gaussiana oscura en la posición del dust lane
2. **Substructura**: agregar pequeñas regiones brillantes asimétricamente distribuidas
3. **Lentes compuestos**: combinar con lentes secundarios débiles ("subhaloes")
   que perturben el anillo en sectores específicos
4. **PSF real**: convolucionar con PSF del Hubble para mayor realismo

## Parámetros ajustables para experimentar

Si quieres más o menos "rotura" del anillo:

- **Aumentar `ell` fuente** (e.g., 0.55): anillo aún más asimétrico
- **Cambiar `pa` fuente**: rota el eje de asimetría → sector oscuro gira
- **Cambiar offset (x0, y0)**: mueve la posición del sector oscuro
- **Cambiar `q_ax` del lente**: de 0.70 a más cercano a 1 (menos elipticidad del lente,
  toda la asimetría viene de la fuente)
