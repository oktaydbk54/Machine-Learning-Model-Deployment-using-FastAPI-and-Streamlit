from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import pickle

# Iris veri setini yükleyelim
iris_df = pd.read_csv('iris.csv')

# X ve y verilerini ayıralım
X = iris_df.iloc[:, :-1]
y = iris_df.iloc[:, -1]

# Modeli eğitelim
n_neighbors = 3
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X, y)

# Oluşturduğumuz modeli model1.pkl adlı dosyaya kaydedelim
with open('KNN.pkl', 'wb') as f:
    pickle.dump(model, f)
