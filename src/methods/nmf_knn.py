
from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
from scipy.optimize import minimize


def reconstruction_loss(W_flat, A_test, H_base):
    """Calculer la perte de reconstruction ||A_test - W_test * H_base||_2."""
    W_test = W_flat.reshape(A_test.shape[0], -1)
    reconstruction = np.dot(W_test, H_base)
    return np.linalg.norm(A_test - reconstruction)


class NMF_KNN(OODBaseDetector):
    def __init__(
        self,

    ):
      super().__init__()

      self.A_train = None
      self.W_train = None
      self.H_Base = None
      self.NMF = None
      self.Scaler = None

    def _fit_to_dataset(self, fit_dataset):

      # Calculate the activations_matrix A_train for the training dataset, to calculate the PCs
      training_features = self.feature_extractor.predict(fit_dataset)
      # The activations_matrix A_train
      A_train = training_features[0][0]
      A_train = self.op.convert_to_numpy(A_train)

      self.A_in = A_train - np.min(A_train) + 1e-5
      
      # The training labels
      labels_train = training_features[1]["labels"]
      
      # Appliquer NMF
      nmf = NMF(n_components=8, init='random', random_state=42)
      self.W_train = nmf.fit_transform(self.A_in)  # La matrice des coefficients (ou des caractéristiques latentes)
      self.H_Base = nmf.components_  # La matrice des composantes (ou la base)
      print("the shape of H_base is : ", self.H_Base.shape)
      print("the shape of W_train is  : ", self.W_train.shape)



      # plt.figure(figsize=(14, 7))
      # # Visualization for feature 1 and feature 2
      # plt.subplot(2, 3, 1)
      # # plt.scatter(self.A_in[:, 0], self.A_in[:, 1], c=labels_train, cmap='viridis', alpha=0.5, edgecolor='k')
      # plt.scatter(W[:, 0], W[:, 1], c=labels_train, s=200, alpha=0.75, marker='X')
      # plt.scatter(self.CAVs[:, 0], self.CAVs[:, 1], c='blue', s=200, alpha=0.75, marker='X')
      # plt.title('NMF with : Feature 1 and Feature 2' )
      # plt.xlabel('Feature 1')
      # plt.ylabel('Feature 2')
      # plt.tight_layout()
      # plt.show()
  

      return

    def _score_tensor(self, inputs):

      features, logits = self.feature_extractor.predict_tensor(inputs)
      A_test = features[0].cpu()
      A_test = self.op.convert_to_numpy(A_test) # la matrice des données de test A_test
      A_test = A_test - np.min(A_test) + 1e-5

      # Initialisation de W_test comme une matrice aplatie (pour l'optimisation)
      initial_W_test_flat = np.random.rand(A_test.shape[0] * self.W_train.shape[1])

      # Minimiser la perte de reconstruction
      result = minimize(reconstruction_loss, initial_W_test_flat, args=(A_test, self.H_Base), method='L-BFGS-B')

      # Remodeler W_test dans sa forme originale (M, K)
      W_test_optimized = result.x.reshape(A_test.shape[0], self.W_train.shape[1])

      # Définir le nombre de voisins à considérer
      k = 10

      # Créer et ajuster le modèle kNN
      neigh = NearestNeighbors(n_neighbors=k)
      neigh.fit(self.W_train)

      # Trouver les k plus proches voisins de W_test
      distances, indices = neigh.kneighbors(W_test_optimized)

      min_distance = np.min(distances, axis=1)

      return min_distance

    @property
    def requires_to_fit_dataset(self) -> bool:
        """
        Whether an OOD detector needs a `fit_dataset` argument in the fit function.


        Returns:
            bool: True if `fit_dataset` is required else False.
        """
        return True

    @property
    def requires_internal_features(self) -> bool:
        """
        Whether an OOD detector acts on internal model features.

        Returns:
            bool: True if the detector perform computations on an intermediate layer
            else False.
        """
        return True


nmf_knn = NMF_KNN()

