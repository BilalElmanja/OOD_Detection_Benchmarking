from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.decomposition import NMF
import torch
from sklearn.neighbors import NearestNeighbors

class CRAFT_PER_CLASS_NMF(OODBaseDetector):
    def __init__(self, 
                 n_components=16

             ):
        super().__init__()
        self.n_components=n_components
        self.NMFs = {}  # Remplacer par NMFs pour clarté
        self.H_Bases = {}
        self.W_trains = {}
        # self.Scaler = None
        self.labels_train = None

    def _fit_to_dataset(self, fit_dataset):
        # Extraction des caractéristiques et des étiquettes

        training_features = self.feature_extractor.predict(fit_dataset)
        if len(training_features[0][0].shape) > 2:
            shape = training_features[0][0].shape
            A_train = training_features[0][0].reshape(shape[0], -1)

        A_train = self.op.convert_to_numpy(A_train)
        self.logits_train = training_features[1]["logits"]
        self.labels_train = self.op.convert_to_numpy(training_features[1]["labels"]).tolist()
        print("example of labels : ", self.labels_train[:10])
        print("logits shape : ", self.logits_train.shape)
        # print("labels shape : ", self.labels_train.shape)
        # Apply softmax to the logits across the last dimension
        probabilities = torch.softmax(self.logits_train, dim=1)
        # Extract the indices of the maximum value in each row, which are the predicted classes
        predicted_classes = torch.argmax(probabilities, dim=1)
        print("shape of predicted classes : ", predicted_classes.shape)
        predicted_classes = self.op.convert_to_numpy(predicted_classes).tolist()
        print("example of predicted classes : ", predicted_classes[:10])
        # Calculate accuracy
        correct_predictions = sum(1 for true, pred in zip(self.labels_train, predicted_classes) if true == pred)
        accuracy = correct_predictions / len(self.labels_train)
        print("accuracy is : ", accuracy)

        

        
       

        return

    def _score_tensor(self, inputs):
        features, _ = self.feature_extractor.predict_tensor(inputs)
        if len(features[0].shape) > 2:
         features[0] = features[0][:,:, 0, 0]
         
        A_test = self.op.convert_to_numpy(features[0].cpu())

        min_distances = np.inf * np.ones(A_test.shape[0])
        
        
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



craft = CRAFT_PER_CLASS_NMF()

