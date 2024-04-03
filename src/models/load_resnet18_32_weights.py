import torch
from resnet18_32x32 import ResNet18_32x32

def load_pretrained_weights(model, checkpoint_path):
    # Charger les poids pré-entraînés
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # Charger le dictionnaire d'état dans le modèle
    model.load_state_dict(state_dict)
    print("Modèle chargé avec succès avec les poids pré-entraînés.")


model = ResNet18_32x32()

# Chemin vers votre fichier .ckpt
checkpoint_path = '../../models/CIFAR-10/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt'

# Charger les poids pré-entraînés
load_pretrained_weights(model, checkpoint_path)

# Vérifier que le modèle est correctement chargé (optionnel)
print(model)
