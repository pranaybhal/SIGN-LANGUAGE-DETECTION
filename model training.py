import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Feature Scaling (important for some algorithms)
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Hyperparameter grid for tuning RandomForest
param_grid = {
    'n_estimators': [100, 200, 300],   # Number of trees
    'max_depth': [5, 10],              # Maximum depth
    'min_samples_split': [10, 15, 20], # Minimum samples to split a node
    'min_samples_leaf': [4, 5, 6],     # Minimum samples at a leaf node
    'bootstrap': [True, False]         # Bootstrap sampling
}

# Initialize and perform GridSearchCV
model = RandomForestClassifier()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# Use the best estimator
best_model = grid_search.best_estimator_

# Perform cross-validation
cv_scores = cross_val_score(best_model, x_train, y_train, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation accuracy: {np.mean(cv_scores) * 100:.2f}%')

# Test on test set
y_predict = best_model.predict(x_test)
test_score = accuracy_score(y_predict, y_test)
print(f'{test_score * 100}% of test samples were classified correctly!')

# Save the trained model (best model)
with open('model.p', 'wb') as f:
    pickle.dump({'model': best_model}, f)  # Save the trained model







