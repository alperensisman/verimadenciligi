import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# pip install pandas
# pip install scikit-learn
# pip install matplotlib

km = KMeans(n_clusters=3, init='k-means++',random_state=0)
data = pd.read_csv('csv\iris.csv')
v = data.iloc[:,1:-1].values
km.fit(v)
predict = km.predict(v)
print(km.cluster_centers_)
 
plt.scatter(v[predict==0,0],v[predict==0,1],s=50,color='red')
plt.scatter(v[predict==1,0],v[predict==1,1],s=50,color='blue')
plt.scatter(v[predict==2,0],v[predict==2,1],s=50,color='green')
plt.title('K-Means Iris Dataset')
plt.show()

plt.scatter(v[predict==0,0],v[predict==0,1],s=50,color='red')
plt.scatter(v[predict==1,0],v[predict==1,1],s=50,color='blue')
plt.scatter(v[predict==2,0],v[predict==2,1],s=50,color='green')
plt.scatter(v[predict==3,0],v[predict==3,2],s=50,color='black')
plt.title('K-Means Iris Dataset')
plt.show()