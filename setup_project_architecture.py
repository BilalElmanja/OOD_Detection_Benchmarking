import os

# Define the project structure again after reset
project_structure = {
    "OOD-Detection-Benchmarking": {
        "datasets": {
            "MNIST": ["Near-OOD/NotMNIST/__init__.py", "Near-OOD/FashionMNIST/__init__.py", "Far-OOD/Texture/__init__.py", "Far-OOD/CIFAR-10/__init__.py", "Far-OOD/TinyImageNet/__init__.py", "Far-OOD/Places365/__init__.py" , "MNIST/__init__.py"],
            "CIFAR-10": ["CIFAR-10/__init__.py", "Near-OOD/CIFAR-100/__init__.py", "Near-OOD/TinyImageNet/__init__.py", "Far-OOD/MNIST/__init__.py", "Far-OOD/SVHN/__init__.py", "Far-OOD/Texture/__init__.py", "Far-OOD/Places365/__init__.py"],
            "CIFAR-100": ["CIFAR-100/__init__.py" ,"Near-OOD/CIFAR-10/__init__.py", "Near-OOD/TinyImageNet/__init__.py", "Far-OOD/MNIST/__init__.py", "Far-OOD/SVHN/__init__.py", "Far-OOD/Texture/__init__.py", "Far-OOD/Places365/__init__.py"],
            "ImageNet-200": ["ImageNet-200/__init__.py","Near-OOD/SSB-hard/__init__.py", "Near-OOD/NINCO/__init__.py", "Far-OOD/iNaturalist/__init__.py", "Far-OOD/Texture/__init__.py", "Far-OOD/OpenImage-O/__init__.py", "Covariate-Shifted ID/ImageNet-C/__init__.py", "Covariate-Shifted ID/ImageNet-R/__init__.py", "Covariate-Shifted ID/ImageNet-v2/__init__.py"],
            "ImageNet-1K": [ "ImageNet-1K/__init__.py", "Near-OOD/SSB-hard/__init__.py", "Near-OOD/NINCO", "Far-OOD/iNaturalist/__init__.py", "Far-OOD/Texture/__init__.py", "Far-OOD/OpenImage-O/__init__.py", "Covariate-Shifted ID/ImageNet-C/__init__.py", "Covariate-Shifted ID/ImageNet-R/__init__.py", "Covariate-Shifted ID/ImageNet-v2/__init__.py"]
        },
        "models":  {
            "MNIST": ["__init__.py"],
            "CIFAR-10": ["__init__.py"],
            "CIFAR-100": ["__init__.py"],
            "ImageNet-200": ["__init__.py"],
            "ImageNet-1K": ["__init__.py"],
        },
        "src": {
            "data_preprocessing": ["__init__.py"],
            "models": ["__init__.py"],
            "nmf_mahalanobis": ["__init__.py"],
            "pca_mahalanobis": ["__init__.py"]
        },
        "results": ["__init__.py"],
        "notebooks": ["__init__.py"],
    }
}

# Function to create the directory structure
def create_dir_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        os.makedirs(path, exist_ok=True)
        if isinstance(content, dict):  # If the content is a dictionary, recurse
            create_dir_structure(path, content)
        elif isinstance(content, list):  # If the content is a list, create subdirectories
            for subname in content:
                subpath = os.path.join(path, subname)
                os.makedirs(subpath, exist_ok=True)

# Specify the base path for the project structure
base_path = './'

# Create the project structure
create_dir_structure(base_path, project_structure["OOD-Detection-Benchmarking"])

print("Project structure created successfully.")

