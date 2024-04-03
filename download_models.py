import gdown
import os
import zipfile

# Dictionnaire des IDs de téléchargement pour les checkpoints
download_id_dict = {
    # 'imagenet_res50_v1.5': '15PdDMNRfnJ7f2oxW6lI-Ge4QJJH3Z0Fy',
    # 'imagenet200_res18_v1.5': '1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs',
    # 'cifar100_res18_v1.5': '1s-1oNrRtmA0pGefxXJOUVRYpaoAML0C-',
    # 'cifar10_res18_v1.5': '1byGeYxM_PlLjT72wZsMQvP6popJeWBgt',
    # 'mnist_lenet': '13mEvYF9rVIuch8u0RVDLf_JMOk3PAYCj',
    'imagenet200_res18_v1.5': '1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs',
}

# Dossier de sauvegarde pour les checkpoints
save_dir = '../data/'

# Vérifiez si le dossier existe, sinon créez-le
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def download_checkpoint(checkpoint_name, save_dir):
    """ Télécharge et extrait le checkpoint spécifié. """
    file_id = download_id_dict[checkpoint_name]
    output_path = os.path.join(save_dir, checkpoint_name + '.zip')
    
    # Télécharger le fichier
    gdown.download(id=file_id, output=output_path, quiet=False)
    
    # Extraire le fichier zip
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    
    # Supprimer le fichier zip
    os.remove(output_path)
    print(f"{checkpoint_name} téléchargé et extrait dans {save_dir}")

# Liste des checkpoints à télécharger
checkpoints_to_download = [
    # 'imagenet_res50_v1.5',
    # 'imagenet200_res18_v1.5',
    # 'cifar100_res18_v1.5',
    # 'cifar10_res18_v1.5',
    'imagenet200_res18_v1.5'
]

# Télécharger les checkpoints
for checkpoint in checkpoints_to_download:
    download_checkpoint(checkpoint, save_dir)
