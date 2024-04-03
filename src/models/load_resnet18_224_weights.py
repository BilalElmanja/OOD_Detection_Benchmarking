import torch
from resnet18_224x224 import ResNet18_224x224

def load_pretrained_weights(model, checkpoint_path):
    # Charger les poids pré-entraînés
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # Charger le dictionnaire d'état dans le modèle
    model.load_state_dict(state_dict)
    print("Modèle chargé avec succès avec les poids pré-entraînés.")


model = ResNet18_224x224(num_classes=200)

# Chemin vers votre fichier .ckpt
checkpoint_path = '../../models/ImageNet-200/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s2/best.ckpt'
# Charger les poids pré-entraînés
load_pretrained_weights(model, checkpoint_path)

# Vérifier que le modèle est correctement chargé (optionnel)
print(model)










