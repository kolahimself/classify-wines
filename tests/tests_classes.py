import numpy as np
import os
import joblib


# load the saved model pipeline
root = os.path.dirname(os.path.abspath(os.getcwd()))
filename = os.path.join(os.path.join(root, 'models'), 'wine_multi_model.pkl')

multi_model = joblib.load(filename)

# define new wine samples
X_new = np.array([[13.72, 1.43, 2.5, 16.7, 108, 3.4, 3.67, 0.19, 2.04, 6.8, 0.89, 2.87, 1285],
                  [12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520]])
print('New samples:\n{}'.format(X_new))

# Recall the wine classes
classes = ['Variety 0', 'Variety 1', 'Variety 2']

# Call the web service, passing the input data
predictions = multi_model.predict(X_new)

# Get the predicted classes.
for prediction in predictions:
    print(prediction, '(' + classes[prediction] + ')')
