# Preparando um Dataset de Lixo para o fastai

Este guia mostra como transformar uma coleção de imagens de diferentes tipos de lixo em um dataset organizado que pode ser carregado facilmente em um notebook Jupyter usando o `fastai` e o modelo `resnet18`.

## 1. Organize as imagens brutas

1. Crie uma pasta em seu computador, por exemplo `~/raw_garbage_images`.
2. Dentro dela crie uma subpasta para cada categoria de lixo que deseja classificar, por exemplo:

```
raw_garbage_images/
├── metal/
├── paper/
├── plastic/
├── glass/
└── organic/
```

3. Copie as imagens correspondentes para dentro das subpastas corretas. Os nomes dos arquivos podem ser livres.

> 💡 **Dica:** mantenha entre 20 e 30 imagens por classe para um primeiro experimento. Quanto mais dados, melhor o modelo.

## 2. Gere as pastas `train/` e `valid/`

Este repositório inclui o script `scripts/prepare_garbage_dataset.py` para criar automaticamente a estrutura esperada pelo fastai (`train/` e `valid/` com subpastas por classe).

Execute no terminal (ajuste os caminhos de acordo com o seu ambiente):

```bash
python scripts/prepare_garbage_dataset.py \
  --source ~/raw_garbage_images \
  --dest   ~/datasets/garbage \
  --valid-pct 0.2
```

O script irá:

- Separar aleatoriamente 20% das imagens para validação (`--valid-pct`).
- Copiar os arquivos para `~/datasets/garbage/train/<classe>` e `~/datasets/garbage/valid/<classe>`.
- Garantir que todas as classes tenham pelo menos uma imagem em `train` e `valid`.

> 📁 Você pode mudar `--dest` para o diretório onde deseja manter o dataset preparado. Use `--seed` para reprodutibilidade do sorteio.

## 3. Carregue o dataset no Jupyter Notebook

Dentro do seu notebook (por exemplo, `fastbook` ou qualquer notebook fastai), rode o seguinte código para treinar uma `resnet18`:

```python
from fastai.vision.all import *
from pathlib import Path

path = Path.home()/"datasets"/"garbage"

# Carrega as imagens usando o DataBlock padrão de classificação

dls = ImageDataLoaders.from_folder(
    path,
    train="train",
    valid="valid",
    seed=42,
    item_tfms=Resize(460),
    batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)],
)

# Inspecione algumas imagens para confirmar os rótulos

dls.show_batch(max_n=9, figsize=(6, 6))

# Treine uma resnet18 com fine-tuning

learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(5)

# Avalie o desempenho

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(5, 5))
```

## 4. Próximos passos

- Adicione mais imagens para melhorar a qualidade do modelo.
- Use `learn.export("garbage-classifier.pkl")` para salvar o modelo e reutilizar depois.
- Experimente `learn.lr_find()` para selecionar uma taxa de aprendizado ideal.
- Caso tenha classes desbalanceadas, ajuste o parâmetro `valid_pct` ou aplique técnicas de data augmentation específicas.

Com esse fluxo você terá um dataset pronto para treinar um classificador de tipos de lixo com `fastai` e `resnet18`.
