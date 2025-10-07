from pathlib import Path
from fastai.vision.all import *  

learn = load_learner("model.pkl")

# Obtém a lista de classes do modelo (nome das categorias)
classes = list(learn.dls.vocab)                    # vocabulário com as classes em ordem consistente

img_path = Path(r"C:/Users/ramon/OneDrive/Documentos/Python Scripts/Fast.ai/garbage classifier/dataset/paper/paper_001.jpg")

if not img_path.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {img_path}")

im = PILImage.create(img_path)
predicted_class, _, probs = learn.predict(im)
print(f"Classe prevista: {predicted_class}")

# Verifica se o arquivo existe
if not img_path.exists():                          # checa existência do arquivo
    print(f"Arquivo não encontrado: {img_path}")   # avisa que não encontrou
    sys.exit(1)                                    # encerra com erro
  
# Cria um objeto de imagem compatível com o FastAI
im = PILImage.create(img_path)                     # carrega a imagem em formato aceito pelo modelo

# Faz a predição com o modelo
predicted_class, _, probs = learn.predict(im)      # retorna classe prevista, índice (ignorado) e vetor de probabilidades

# Imprime a classe prevista
print(f"Classe prevista: {predicted_class}")       # mostra a categoria que o modelo escolheu

# Imprime a probabilidade de cada classe
for i, cls in enumerate(classes):                  # percorre as classes pelo índice
    print(f"Probabilidade de {cls}: {probs[i]:.4f}")  # mostra a probabilidade formatada com 4 casas
