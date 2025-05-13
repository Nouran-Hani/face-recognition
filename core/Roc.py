import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

class ROC_Curve:
    def __init__(self, model):
        self.model = model

    def compute_roc_curve(self, y_true, y_scores):
        """
        Compute ROC curve (FPR, TPR) from scratch given true labels and decision function scores.
        """
        # Sort decision function scores and labels in descending order
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]
        y_scores_sorted = y_scores[sorted_indices]

        # Initialize variables
        tpr_list = []
        fpr_list = []
        thresholds = np.unique(y_scores_sorted)

        # Total number of positives and negatives
        total_positive = np.sum(y_true == 1)
        total_negative = len(y_true) - total_positive

        # Initialize true positive and false positive counts
        tp = 0
        fp = 0
        fn = total_positive
        tn = total_negative

        for threshold in thresholds:
            # Update tp, fp, fn, tn based on threshold
            tp = np.sum(y_true_sorted[y_scores_sorted >= threshold] == 1)
            fp = np.sum(y_true_sorted[y_scores_sorted >= threshold] == 0)
            fn = total_positive - tp
            tn = total_negative - fp

            # Calculate TPR (True Positive Rate) and FPR (False Positive Rate)
            tpr = tp / total_positive if total_positive > 0 else 0
            fpr = fp / total_negative if total_negative > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        return np.array(fpr_list), np.array(tpr_list)

    def plot_roc(self, y_true, y_scores):
        """
        Plot the ROC curve from scratch.
        """
        fpr, tpr = self.compute_roc_curve(y_true, y_scores)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Chance level (diagonal line)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()

    def get_decision_function_scores(self, X):
        """
        Get the decision function scores for SVM.
        """
        return self.model.decision_function(X)




# Assuming you've already loaded and split the data
# Train the SVM model
model = FaceRecognitionModel(n_components=100)
model.train(X_train, y_train)

# Get the decision function scores from SVM
roc = ROC_Curve(model.model)  # Pass the trained SVM model
X_test_pca = PCA(n_components=100, whiten=True).fit_transform(X_test)
y_scores = roc.get_decision_function_scores(X_test_pca)

# Plot the ROC curve
roc.plot_roc(y_test, y_scores)
