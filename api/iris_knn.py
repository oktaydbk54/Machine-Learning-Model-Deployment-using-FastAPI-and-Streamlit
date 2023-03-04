from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Iris veri setini yükleyelim
iris_df = pd.read_csv('IRIS.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    iris_df.iloc[:, :-1], iris_df.iloc[:, -1], test_size=0.2, random_state=42)
# Modeli eğitelim
n_neighbors = 3
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train, y_train)

# Oluşturduğumuz modeli model1.pkl adlı dosyaya kaydedelim
with open('KNN.pkl', 'wb') as f:
    pickle.dump(model, f)
