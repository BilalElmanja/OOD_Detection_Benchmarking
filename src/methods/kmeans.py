
from oodeel.datasets import OODDataset
from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.spatial.distance import cdist



class K_Means(OODBaseDetector):
    def __init__(
        self,
    ):
      super().__init__()

      self.CAVs = None
      self.k = None
      self.A_train = None
      self.A_in = None
      self.A_out = None

    def _fit_to_dataset(self, fit_dataset):
      # we calculate the activations_matrix A_train for the training dataset, in order to calculate the CAVs Matrix
      training_features = self.feature_extractor.predict(fit_dataset)
      # the activations_matrix A_train
      A_train = training_features[0][0]
      A_train = self.op.convert_to_numpy(A_train)
      # the training labels
      labels_train = training_features[1]["labels"]
      # the training logits
      logits_train = training_features[1]["logits"]
      logits_train = self.op.convert_to_numpy(logits_train)
      self.A_in = A_train
      if len(self.A_in.shape) > 2:
         self.A_in = self.A_in[:,:, 0, 0]
      # finding the best number of Clusters K in order to best represent the data
      # Elbow Method
    #   inertias = []
    #   # Silhouette Method
    #   silhouette_scores = []
    #   K_range = range(10, 11)  # range for number of clusters k
    #   print("finding the optimal number of clusters k...")
    #   for k in K_range:
    #     print("number of clusters : ", k)
    #     kmeans = KMeans(n_clusters=k, random_state=42, max_iter=100).fit(self.A_in)
    #     inertias.append(kmeans.inertia_)
    #     silhouette_scores.append(silhouette_score(self.A_in, kmeans.labels_))

    #   print("optimal number of clusters has been found...")
    #   # # find the highest silhouette score
    #   highest_silhouette_score = max(silhouette_scores)
    #   best_silhouette_k = list(K_range)[silhouette_scores.index(highest_silhouette_score)]
      self.k = 10 # best_silhouette_k
    #   print("optimal k = ", self.k)
    #   print("#------------------------------------------------------------")
      # perform the k-means with optimal clusters number
      print("Performing K-means clustering...")
      print("shape of A_in is : ", self.A_in.shape)
      kmeans = KMeans(n_clusters=self.k, random_state=42, max_iter=100).fit(self.A_in)
      print("K-means clustering Done...")
      print("#------------------------------------------------------------")
      # get the centroids coordinates in the feature space with shape (10, 10) k*p
      centroids = kmeans.cluster_centers_
      self.CAVs = centroids
      # get the labels of the centroids
      labels_centroids = kmeans.labels_

    #   # Plotting the Elbow Method
    #   plt.figure(figsize=(12, 6))
    #   plt.subplot(1, 2, 1)
    #   plt.plot(K_range, inertias, '-o')
    #   plt.xlabel('Number of clusters, k')
    #   plt.ylabel('Inertia')
    #   plt.title('Elbow Method')

    #   # # Plotting Silhouette Scores
    #   plt.subplot(1, 2, 2)
    #   plt.plot(K_range, silhouette_scores, '-o')
    #   plt.xlabel('Number of clusters, k')
    #   plt.ylabel('Silhouette Score')
    #   plt.title('Silhouette Score Method')
    #   plt.tight_layout()
    #   plt.show()

    #   # print("\n \n \n")

    #   plt.figure(figsize=(14, 7))

    #   # Visualization for feature 1 and feature 2
    #   plt.subplot(1, 2, 1)
    #   plt.scatter(self.A_in[:, 0], self.A_in[:, 1], c=labels_train, cmap='viridis', alpha=0.5, edgecolor='k')
    #   plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
    #   plt.title('K-Means Clustering with : Feature 1 and Feature 2' )
    #   plt.xlabel('Feature 1')
    #   plt.ylabel('Feature 2')

    #   # Visualization for feature 3 and feature 4
    #   plt.subplot(1, 2, 2)
    #   plt.scatter(self.A_in[:, 2], self.A_in[:, 3], c=labels_train, cmap='viridis', alpha=0.5, edgecolor='k')
    #   plt.scatter(centroids[:, 2], centroids[:, 3], c='red', s=200, alpha=0.75, marker='X')
    #   plt.title('K-Means Clustering with : Feature 3 and Feature 4')
    #   plt.xlabel('Feature 3')
    #   plt.ylabel('Feature 4')
      
    #   plt.tight_layout()
    #   plt.show()

      return

    def _score_tensor(self, inputs):

      features, logits = self.feature_extractor.predict_tensor(inputs)
      
      if len(features[0].shape) > 2:
         features[0] = features[0][:,:, 0, 0]
      # Calculate the Euclidean distance between each sample and the centroids
      distances = cdist(features[0].cpu(), self.CAVs, 'euclidean')
      # Calculate the average distance for each sample to all centroids
    #   squared_distances = distances ** 2
    #   frobenius_norms = np.sqrt(squared_distances.sum(axis=1))
    #   average_distances = distances.mean(axis=1)
    #   max_distances = distances.max(axis=1)
      min_distances = distances.min(axis=1)

      return min_distances

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



kmeans = K_Means()