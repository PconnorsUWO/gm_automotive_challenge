import numpy as np
import polars as pl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



class RFModelWrapper:
    def __init__(self, training_data):
        """
        Initialize and train the RandomForest model.

        Parameters:
        -----------
        X_train : array-like or DataFrame of shape (n_samples, 4)
            The training features corresponding to x1, y1, x2, y2.
        y_train : array-like or Series of shape (n_samples,)
            The target labels.
        """
        df = pl.read_csv(
        training_data,
        separator=" ",
        has_header=False,
        )

        df.columns = ["image_file", "class_id", "x1", "y1", "x2", "y2"]

        df = df.drop("image_file")
        X = df.drop("class_id")
        y = df["class_id"]

        self.model = RandomForestClassifier(
            n_estimators=30,            # Number of trees
            criterion='entropy',        # Splitting criterion
            max_depth=None,             # Maximum depth of trees
            min_samples_split=2,        # Minimum samples required to split
            min_samples_leaf=0.01,      # Minimum samples required at a leaf (fractional)
            min_weight_fraction_leaf=0.0,
            max_features='sqrt',        # Maximum features per split
            max_leaf_nodes=None,        # Maximum leaf nodes
            min_impurity_decrease=0.0001, # Minimum impurity decrease to split
            bootstrap=True,             # Bootstrapping for sampling
            oob_score=True,             # Out-of-bag score
            n_jobs=-1,                  # Use all available cores
            random_state=20190305,      # Seed for reproducibility
            verbose=1,                  # Verbosity during training
            warm_start=False,           # Do not reuse previous training
            class_weight='balanced'     # Handle imbalanced classes
        )
        # Train the model on initialization
        self.model.fit(X, y)
    
    def predict(self, x1, y1, x2, y2) -> list[int, float]:
        """
        Predict the class label and confidence score for the given input features.

        Parameters:
        -----------
        x1 : float
        y1 : float
        x2 : float
        y2 : float

        Returns:
        --------
        prediction : The predicted class label.
        confidence : The confidence score (probability) of the prediction.
        """
        sample = pd.DataFrame([[x1, y1, x2, y2]], columns=["x1", "y1", "x2", "y2"])
        prediction = self.model.predict(sample)[0]
        probabilities = self.model.predict_proba(sample)[0]
        confidence = round(max(probabilities), 2)  # Round confidence to 1 decimal place
        return prediction, confidence

if __name__ == "__main__":
    rf = RFModelWrapper("labeled_data.txt")
    print(rf.predict(85, 210, 160, 266))