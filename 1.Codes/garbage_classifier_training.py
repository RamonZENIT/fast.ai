
from fastai.vision.all import *  # Importa todos os módulos da biblioteca FastAI relacionados a visão computacional (classificação, data loaders, modelos, etc.)
from pathlib import Path          # Importa a classe Path para manipulação de caminhos de arquivos e diretórios de forma prática e multiplataforma
from PIL import Image             # Importa a classe Image da biblioteca Pillow para abrir e manipular imagens
import matplotlib.pyplot as plt   # Importa o módulo pyplot do Matplotlib para exibir gráficos e imagens

# Define o caminho base do dataset no sistema local
path = Path(r"C:\Users\ramon\OneDrive\Documentos\Python Scripts\Fast.ai\garbage classifier\dataset")

# Verifica se o caminho definido realmente existe no sistema e imprime True ou False
print(path.exists())  # Deve retornar True se o diretório existir

print(list(path.iterdir()))  # Mostra os itens dentro da pasta

# Verificar imagens

# Caminho da pasta que contém as imagens da classe "metal"
folder = Path(r"C:/Users/ramon/OneDrive/Documentos/Python Scripts/Fast.ai/garbage classifier/dataset/metal")

# Busca todas as imagens com extensão .jpg e .png dentro da pasta especificada
img_paths = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

# Limita a lista às 5 primeiras imagens para visualização
img_paths = img_paths[:5]  # só as 5 primeiras imagens

# Cria uma grade (grid) de 1 linha e 5 colunas para exibir as imagens lado a lado
fig, axes = plt.subplots(1, 5, figsize=(15, 6))

# Itera sobre cada eixo (posição do grid) e caminho de imagem correspondente
for ax, img_path in zip(axes.flat, img_paths):
    img = Image.open(img_path)              # Abre a imagem a partir do caminho
    ax.imshow(img)                          # Exibe a imagem no eixo correspondente
    ax.set_title(img_path.name, fontsize=8) # Define o nome do arquivo como título da imagem
    ax.axis("off")                          # Remove os eixos e rótulos para uma visualização mais limpa

# 3) DataBlock: cria uma estrutura para carregar e preparar os dados de imagens.
# Cada imagem é rotulada automaticamente pelo nome da pasta em que se encontra.

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),           # Define os tipos de dados: imagens e categorias (rótulos)
    get_items=get_image_files,                    # Função que busca todas as imagens recursivamente no diretório base
    splitter=RandomSplitter(valid_pct=0.2, seed=42),  # Divide os dados em treino (80%) e validação (20%) de forma aleatória, com semente fixa
    get_y=parent_label,                           # Define o rótulo de cada imagem como o nome da pasta-mãe
    item_tfms=Resize(192, method='squish'),       # Redimensiona cada imagem para 192x192 antes do treino (mantendo proporção "esticada")
    batch_tfms=[*aug_transforms(size=224),        # Aplica transformações de aumento de dados (rotação, corte, flip, etc.) no batch
                Normalize.from_stats(*imagenet_stats)]  # Normaliza as imagens com as médias e desvios padrão do ImageNet
).dataloaders(path, bs=32, num_workers=0)         # Cria os DataLoaders com batch size de 32; num_workers=0 evita travamentos no Windows/Jupyter

# 4) Inspecionar um batch
dls.show_batch(max_n=12, figsize=(8, 8))          # Exibe 12 imagens aleatórias do batch com seus rótulos para verificação visual

# Cria um modelo de aprendizado profundo (CNN) usando a arquitetura ResNet18 pré-treinada no ImageNet
# 'dls' fornece os dados de treino e validação, e 'error_rate' será usada como métrica de desempenho
learn = vision_learner(dls, resnet18, metrics=error_rate)

# Realiza o fine-tuning do modelo por 10 épocas, ajustando os pesos da rede para o novo conjunto de dados
learn.fine_tune(10)

# Define o caminho completo de uma imagem específica dentro do dataset (classe "glass")
img_path = Path(r"C:/Users/ramon/OneDrive/Documentos/Python Scripts/Fast.ai/garbage classifier/dataset/glass/glass_001.jpg")

# (opcional) verifica se o arquivo realmente existe no caminho especificado
print(img_path.exists())

# Cria um objeto PILImage (formato compatível com o FastAI) a partir do caminho da imagem
im = PILImage.create(img_path)

# Exibe uma miniatura da imagem com tamanho 256x256 pixels (útil para visualização no Jupyter Notebook)
im.to_thumb(256, 256)

# Obtém a lista de classes (rótulos) aprendidas a partir do DataLoader do modelo
# 'vocab' contém todos os nomes de categorias identificadas no dataset
classes = list(learn.dls.vocab)

# Exibe na tela a lista completa de classes detectadas
print(classes)

# Usa o modelo treinado para fazer uma previsão sobre a imagem 'im'
# Retorna: a classe prevista, o índice interno (ignoramos com '_') e as probabilidades de cada classe
predicted_class, _, probs = learn.predict(im)

# (Comentado) Exemplo alternativo: prever outra imagem chamada 'forest000001.png'
# is_bird,_,probs = learn.predict(PILImage.create('forest000001.png'))

# Imprime no console qual classe o modelo acredita que a imagem pertence
print(f"This is a: {predicted_class}.")

# Encontra o índice da classe prevista dentro da lista de classes (vocab)
predicted_idx = classes.index(predicted_class)

# (Comentado) Exemplo: imprimir apenas a probabilidade da classe prevista
# print(f"Probability it's a {predicted_class}: {probs[predicted_idx]:.4f}")

# Percorre todas as classes e imprime a probabilidade associada a cada uma
for idx, ele in enumerate(classes):
    print(f"Probability it's a {ele}: {probs[idx]:.4f}")

# Exporta o modelo treinado (incluindo pesos, arquitetura e transformações) para um arquivo chamado 'model.pkl'
# Esse arquivo pode ser recarregado posteriormente com load_learner() para fazer previsões sem precisar treinar novamente
learn.export('model.pkl')
