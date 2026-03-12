# Contexto — Replicación Kaggle PANDA 1st Place Solution (para TFG)

## Situación
Estoy desarrollando mi TFG sobre redes neuronales convolucionales.
Como parte del TFG, quiero **replicar en local el trabajo de terceros**: la solución ganadora (#1) del Kaggle PANDA Challenge, desarrollada por el equipo PND (yukkyo + kentaroy47).
El código y los modelos son de sus autores originales (licencia CC-BY-NC 4.0).
**Ahora mismo estoy en Linux** y quiero proceder con la replicación completa.

---

## Repositorio
- **Ruta local**: el repositorio ya está clonado en el portátil
- **GitHub original**: https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution
- **Ramas**: solo hay una rama relevante (`master`). La rama `yukkyo-patch-1` solo añade un fix de README y ya está mergeada.
- **Licencia**: CC-BY-NC 4.0

---

## Hardware disponible
- GPU NVIDIA con **<8GB VRAM** (RTX 3060 o inferior)
- Sistema operativo: **Linux**
- Preferencia de entorno: **pip + venv**
- Cuenta Kaggle con API key configurada

---

## Sobre el problema (PANDA Challenge)
- **Tarea**: clasificar imágenes histopatológicas de biopsias de próstata en 6 grados ISUP (0–5)
- **Métrica**: Quadratic Weighted Kappa (QWK)
- **Score ganador**: público 0.904, privado **0.940** (1st place)
- **Score reproducible** (seed fijo): público 0.894, privado 0.939

---

## Arquitectura: Ensemble de 2 modelos

### Modelo 1 — "arutema47"
| Parámetro | Valor |
|---|---|
| Backbone | EfficientNet-B0 |
| Tiles | 36 tiles de 256×256 (cuadrícula 6×6) |
| Imagen entrada | 256×256 |
| Canales salida | 5 |
| Pooling | AvgPool |
| Loss | BCEWithLogitsLoss |
| LR | 1e-4 con warmup 10× en epoch 0 |
| Epochs | 30 |
| Augmentations | Flip, Transpose, ShiftScaleRotate, ElasticTransform, **Mixup** |
| Script | `train_famdata-kfolds.py` |

### Modelo 2 — "fam_taro"
| Parámetro | Valor |
|---|---|
| Backbone | EfficientNet-B1 |
| Tiles | 64 tiles de 192×192 (cuadrícula 8×8, imagen 1536×1536) |
| Imagen entrada | 1536×1536 |
| Canales salida | 10 |
| Pooling | **GeM** (Generalized Mean, p learnable) |
| Loss | BCEWithLogitsLoss |
| LR | 3e-5 con warmup 10× en epoch 0 |
| Epochs | 30 |
| FP16 | Sí (O1), grad_acc=2 |
| Augmentations | Flip, Rotate90, ShiftScaleRotate, **Cutout**, BrightnessContrast |
| Script | `src/train.py --config configs/final_2.yaml` |

---

## Idea clave: Salida Ordinal Multi-Label
En vez de clasificación directa a 6 clases, usan multi-label ordinal:
```
Si ISUP grade = 3  →  target = [1, 1, 1, 0, 0]
Predicción sigmoid: [0.95, 0.88, 0.76, 0.21, 0.09]
grade_pred = pred.sum() = 2.89 ≈ 3  ✓
```
El Modelo 2 tiene 10 canales: los 5 primeros predicen ISUP, los 5 últimos el primer componente del Gleason score.

---

## Pipeline completo paso a paso

```
FASE 1 — Preparación de datos
  train_images/*.tiff
    → s07_simple_tile.py --mode 0   (tiles normales)
    → s07_simple_tile.py --mode 2   (tiles con offset)
    → a00_save_tiles.py             (descomprimir ZIPs)
  s00_make_k_fold.py                (ya hecho, CSV ya existe en input/)

FASE 2 — Entrenamiento modelo base (para limpiar labels)
  train.py --config configs/final_1.yaml --kfold 1
  train.py --config configs/final_1.yaml --kfold 2
  train.py --config configs/final_1.yaml --kfold 3
  train.py --config configs/final_1.yaml --kfold 4
  train.py --config configs/final_1.yaml --kfold 5
    → ~18h por fold en Titan RTX (ajustar batch_size para <8GB VRAM)
  kernel.py --kfold 1..5            (predicciones sobre validación)

FASE 3 — Label cleaning
  data_process/s12_remove_noise_by_local_preds.py
    → detecta ~20% de etiquetas ruidosas

FASE 4 — Entrenamiento modelos finales
  Modelo 1: python train_famdata-kfolds.py   (desde raíz del repo)
  Modelo 2: train.py --config configs/final_2.yaml --kfold 1
            train.py --config configs/final_2.yaml --kfold 4
            train.py --config configs/final_2.yaml --kfold 5
            (solo folds 1, 4, 5 — los mejores en LB)

FASE 5 — Ensemble e inferencia
  submitted_notebook.ipynb
```

---

## Estructura de ficheros clave

```
Kaggle-PANDA-1st-place-solution/
├── src/
│   ├── train.py                   # Entrenamiento principal (Modelo 2)
│   ├── trainer.py                 # LightningModule: forward, val, metrics
│   ├── dataset.py                 # DataLoader con tiles y augmentaciones
│   ├── factory.py                 # Factory: modelos, losses, optimizers
│   ├── kernel.py                  # Inferencia / predicciones locales
│   ├── configs/
│   │   ├── final_1.yaml           # Config modelo base (limpieza labels)
│   │   └── final_2.yaml           # Config modelo final fam_taro
│   ├── models/
│   │   ├── efficientnet.py        # CustomEfficientNet (backbone principal)
│   │   └── layer.py               # GeM, SEBlock, SCSE
│   ├── myloss/loss.py             # BCEWithLogitsLoss, FocalLoss, QWKLoss...
│   ├── utils/
│   │   ├── tile.py                # Procesamiento de tiles
│   │   └── trainer.py             # Mixup, Cutmix
│   └── data_process/
│       ├── s07_simple_tile.py     # Extracción de tiles desde TIFF
│       ├── s00_make_k_fold.py     # K-Fold estratificado
│       └── s12_remove_noise_by_local_preds.py  # Label cleaning
├── train_famdata-kfolds.py        # Entrenamiento Modelo 1 (arutema)
├── submitted_notebook.ipynb       # Ensemble final
├── final_models/                  # Pesos pre-entrenados ya incluidos
│   ├── efficientnet-b0*_fold*.pth # Modelo 1 (5 folds)
│   └── final_2_efficientnet-b1_kfold_*.pt  # Modelo 2 (3 folds)
├── input/
│   ├── train-5kfold.csv           # Ya incluido (no hace falta regenerar)
│   ├── train-5kfold_remove_noisy_*.csv  # Ya incluido
│   ├── duplicate_imgids_*.csv     # Ya incluido
│   ├── train_images/              # HAY QUE DESCARGAR DE KAGGLE
│   └── train_label_masks/         # HAY QUE DESCARGAR DE KAGGLE
└── docker/
    ├── Dockerfile                 # Ubuntu 18.04 + CUDA 10.2 + Python 3.7.2
    └── requirements.txt           # Dependencias exactas
```

---

## Dependencias críticas

```
torch==1.5.0
torchvision==0.6.0
pytorch-lightning==0.8.5
efficientnet-pytorch==0.6.3
albumentations==0.4.1
scikit-image==0.17.2
opencv-python==4.2.0.34
numpy==1.18.5
pandas==1.0.4
scikit-learn==0.23.1
iterative-stratification==0.1.6
tqdm==4.46.1
PyYAML==5.3.1
ttach==0.0.2

# Instalación especial:
nvidia/apex              # compilar desde fuente (para FP16)
warmup-scheduler         # pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
efficientnet_pytorch     # pip install efficientnet_pytorch
```

---

## Ajustes necesarios para <8GB VRAM

Los autores usaban **Titan RTX (24GB)**. Con menos VRAM hay que reducir:

- **Modelo 1**: `batch_size: 3 → 1` (ya usaban 3, quizás funcione)
- **Modelo 2**: `batch_size: 6 → 2 o 3` + mantener `grad_acc: 2` (o subir a 4)
- El `gradient_accumulation` ya está implementado, simula batches más grandes

Ficheros a modificar:
- `src/configs/final_1.yaml` → línea `batch_size: 6`
- `src/configs/final_2.yaml` → línea `batch_size: 6`
- `train_famdata-kfolds.py` → variable `batch_size`

---

## Técnicas avanzadas implementadas

| Técnica | Dónde | Para qué |
|---|---|---|
| GeM Pooling | `models/layer.py` | Pooling adaptativo learnable |
| SEBlock | `models/layer.py` | Atención por canal |
| Mixup (por proveedor) | `utils/trainer.py` | Regularización respetando dominio |
| Cutout | config augmentation | Regularización espacial |
| Label Cleaning | `s12_remove_noise.py` | Elimina ~20% de labels ruidosos |
| Ordinal multi-label | `trainer.py` | Captura orden natural de grados ISUP |
| Warmup LR | scheduler | Estabiliza inicio del entrenamiento |
| FP16 + grad_acc | `trainer.py` | Simula batches grandes con menos VRAM |
| K-Fold estratificado | `s00_make_k_fold.py` | Cross-val por provider+grade |

---

## Lo que YA está listo (no hace falta regenerar)

- `input/train-5kfold.csv` — splits ya calculados (seed 1222)
- `input/train-5kfold_remove_noisy_by_0622_*.csv` — labels limpios de la competición
- `input/duplicate_imgids_imghash_thres_090.csv` — duplicados detectados
- `final_models/*.pth / *.pt` — pesos entrenados listos para inferencia

---

## Lo que HAY que hacer desde cero

1. Descargar `train_images/` y `train_label_masks/` de Kaggle (~400GB)
2. Configurar entorno Python con dependencias exactas
3. Generar tiles (s07_simple_tile.py) — proceso largo pero sin GPU
4. Entrenar modelos (o usar los pesos de `final_models/` para solo inferencia)

---

## Primer paso al llegar a Linux

```bash
# 1. Verificar entorno
nvidia-smi
nvcc --version
python --version

# 2. Crear entorno virtual
python -m venv panda_env
source panda_env/bin/activate

# 3. Instalar dependencias
pip install -r docker/requirements.txt
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install efficientnet_pytorch

# 4. Compilar apex (para FP16)
git clone https://github.com/NVIDIA/apex /tmp/apex
cd /tmp/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# 5. Descargar datos
cd input/
kaggle competitions download -c prostate-cancer-grade-assessment
unzip prostate-cancer-grade-assessment.zip
```

---

## Nota para Claude en Linux

Cuando retomes esta conversación, el siguiente paso es:
1. Verificar que el entorno está bien instalado
2. Ajustar los `batch_size` en los YAML según la VRAM disponible
3. Arrancar la generación de tiles (paso más largo, sin GPU)
4. Proceder con el entrenamiento por fases

El usuario tiene todo el tiempo necesario y quiere replicar el proceso completo,
no solo hacer inferencia con los pesos pre-entrenados.
