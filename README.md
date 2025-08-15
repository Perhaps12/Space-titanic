First attempt learning/using XGBoost to train an AI model  
highest scoring attempt provided in the file along with the code

**Process:**
- Splits the cabin category into 3 different columns to allow the model to be trained on it
- Splits the training data into 80% training, 20% validation
- Identifies the low cardinality categorical and numerical columns
- One hot encodes the categorical columns
- Rounds the final predictions to the nearest integer (1: True, 0: False)
- Uses XGBoost to train the model on the data & checks the MAE using the validation set
