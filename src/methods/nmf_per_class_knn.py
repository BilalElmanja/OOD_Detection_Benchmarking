from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors

class NMF_Unique_Classes_KNN(OODBaseDetector):
    def __init__(self):
        super().__init__()
        self.n_basis = 5 # 9 for all scripts except yannick's script !!!!!!!!!!!
        self.NMFs = {}  # Remplacer par NMFs pour clarté
        self.H_Bases = {}
        self.W_trains = {}
        # self.Scaler = None
        self.labels_train = None

    def _fit_to_dataset(self, fit_dataset):
        # Extraction des caractéristiques et des étiquettes
        training_features = self.feature_extractor.predict(fit_dataset)
        A_train = self.op.convert_to_numpy(training_features[0][0])
        self.labels_train = self.op.convert_to_numpy(training_features[1]["labels"])

        if len(A_train.shape) > 2:
            A_train = A_train[:,:, 0, 0]
        
        # S'assurer que les données sont positives pour NMF
        self.A_in = A_train - np.min(A_train) + 1e-5
        
        # self.Scaler = StandardScaler()
        # A_train_scaled = self.Scaler.fit_transform(self.A_in)
        self.classes_ = np.unique(self.labels_train)
        
        for class_label in self.classes_:
            A_train_class = self.A_in[self.labels_train == class_label]
            
            nmf = NMF(n_components=self.n_basis, init='random', random_state=42)
            W_train_class = nmf.fit_transform(A_train_class)
            H_Base_class = nmf.components_
            
            self.NMFs[class_label] = nmf  # Stocker l'objet NMF
            self.H_Bases[class_label] = H_Base_class
            self.W_trains[class_label] = W_train_class

        return

    def _score_tensor(self, inputs):
        features, _ = self.feature_extractor.predict_tensor(inputs)
        if len(features[0].shape) > 2:
         features[0] = features[0][:,:, 0, 0]
         
        A_test = self.op.convert_to_numpy(features[0].cpu())
        A_test = A_test - np.min(A_test) + 1e-5
        
        min_distances = np.inf * np.ones(A_test.shape[0])
        
        for class_label in self.classes_:
            nmf = self.NMFs[class_label]
            W_test_class = nmf.transform(A_test)
            
            neigh = NearestNeighbors(n_neighbors=20)
            W_train_class = self.W_trains[class_label]
            neigh.fit(W_train_class)
            distances, _ = neigh.kneighbors(W_test_class)
            min_distance = np.min(distances, axis=1)
            
            min_distances = np.minimum(min_distances, min_distance)
        
        return min_distances

    @property
    def requires_to_fit_dataset(self) -> bool:
        return True

    @property
    def requires_internal_features(self) -> bool:
        return True

# Note: Assurez-vous que clear_output() est appelé à l'endroit approprié si nécessaire, par exemple :
# from IPython.display import clear_output
# clear_output()



nmf_per_class = NMF_Unique_Classes_KNN()

