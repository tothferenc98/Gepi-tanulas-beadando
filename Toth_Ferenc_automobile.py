# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:23:58 2021

@author: Tóth Ferenc
"""

import numpy as np;  # importing numerical computing package
from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework
import pandas as pd
from sklearn.feature_selection import SelectKBest; # importing feature selection 
from sklearn.decomposition import PCA;  # importing PCA# importing dimension reductionfrom sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score; # importing crossvalidation
from sklearn.linear_model import LinearRegression,LogisticRegression;  # importing linear regression class
from sklearn.utils.random import sample_without_replacement;  #  importing sampling
from sklearn.metrics import r2_score,confusion_matrix, plot_confusion_matrix
import scipy as sp;  # Scientific Python library
from sklearn.cluster import KMeans; # Class for K-means clustering
from sklearn.metrics import davies_bouldin_score;  # function for Davies-Bouldin goodness-of-fit 
from sklearn.ensemble import GradientBoostingRegressor


data = pd.read_csv('https://raw.githubusercontent.com/tothferenc98/Gepi-tanulas-beadando/master/automobile.csv');  #https://archive.ics.uci.edu/ml/datasets/Automobile


# adatok átalakítása

# horsepower kérdőjelek törlése
horsepower = data['horsepower'].loc[data['horsepower'] != '?'];
hpmean = horsepower.astype(str).astype(int).mean();
data['horsepower'] = data['horsepower'].replace('?', hpmean).astype(int);

# normalized-losses oszlop 'nan' értékű elemeinek cseréje az átlag normalized-losses értékével 
data['normalized-losses'] = data['normalized-losses'].fillna(data['normalized-losses'].mean())

# ajtók string értékének átalakítása intre
data['num-of-doors']=data['num-of-doors'].map({'two':2,'four':4})

# szelepek számának string értékének átalaítása intre
data['num-of-cylinders']=data['num-of-cylinders'].map({'two':2,'three':3,'four':4,'five':5,'six':6,'eight':8,'twelve':12})

# num-of-doors nan helyettesítése
data['num-of-doors'] = data['num-of-doors'].fillna(4)


# plot1
plt.figure();
data.make.value_counts().nlargest(10).plot(kind='bar');
plt.title("Autók száma márkánként");
plt.ylabel('Darabszám');
plt.xlabel('Gyártó');
plt.show();

# plot2
plt.figure();
data['fuel-type'].value_counts().plot(kind='bar');
plt.title("Üzemanyag típusa szerint az autók száma");
plt.ylabel('Autók száma');
plt.xlabel('Üzemanyag típusa');
plt.show();

# plot3
plt.figure();
data.horsepower[np.abs(data.horsepower-data.horsepower.mean())<=(3*data.horsepower.std())].hist(bins=5);
plt.title("Lőerő autónként")
plt.ylabel('Autók száma')
plt.xlabel('Lóerő');
plt.show();

# plot4
sns.lmplot('price',"horsepower", data);

# plot5
sns.lmplot('engine-size',"horsepower", data, hue="make",fit_reg=False);


# dataframe másolása
data2=data.copy();

# autók márkáinak átalakítása számmá
data2['make']=data['make'].map({'alfa-romero':0,'audi':1,'bmw':2,'chevrolet':3,'dodge':4,'honda':5,'isuzu':6,'jaguar':7,'mazda':8,'mercedes-benz':9,'mercury':10,'mitsubishi':11,'nissan':12,'peugot':13,'plymouth':14,'porsche':15,'renault':16,'saab':17,'subaru':18,'toyota':19,'volkswagen':20,'volvo':21})


data2 = data2.drop(['fuel-type', 'aspiration', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'fuel-system', 'bore', 'stroke', 'peak-rpm'], axis=1)



data_without_price=data2.copy();
data_without_price = data2.drop(['price'], axis=1)
price=data2['price'];

p = data_without_price.shape[1]; # number of attributes

# PCA
feature_selection = SelectKBest(k=2);
feature_selection.fit(data_without_price,price);
scores = feature_selection.scores_;
features = feature_selection.transform(data_without_price);
mask = feature_selection.get_support();
feature_indices = [];
for i in range(p):
    if mask[i] == True : feature_indices.append(i);
x_axis, y_axis = feature_indices;

print('A bemeneti attribútumok fontossági súlya')
for i in range(p):
    print(scores[i]);
data_without_price.plot.scatter(x_axis,y_axis,s=50,c=price)

# Full PCA using scikit-learn
pca = PCA();
pca.fit(data_without_price);

# Visualizing the variance ratio which measures the importance of PCs
fig = plt.figure();
plt.title('Explained variance ratio plot');
var_ratio = pca.explained_variance_ratio_;
x_pos = np.arange(len(var_ratio));
plt.xticks(x_pos,x_pos+1);
plt.xlabel('Főkomponensek');
plt.ylabel('Változás');
plt.bar(x_pos,var_ratio, align='center', alpha=0.5);
plt.show(); 


# Tanítás 
df_train_x = data_without_price
df_train_x.describe()

df_train_y = price
df_train_y.describe

x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.15, random_state=42)

# LogisticRegression
fig = plt.figure();
reg = LogisticRegression().fit(x_train, y_train)
predictions = reg.predict(x_test)
print("LogisticRegression r2_score is : " , r2_score(y_test, predictions))
sns.regplot(x = y_test, y = predictions)

# LinearRegression
fig = plt.figure();
reg = LinearRegression().fit(x_train, y_train)
predictions = reg.predict(x_test)
print("LinearRegression r2_score is : " , r2_score(y_test, predictions))
sns.regplot(x = y_test, y = predictions)

# GradientBoostingRegressor
fig = plt.figure();
reg = GradientBoostingRegressor().fit(x_train, y_train)
predictions = reg.predict(x_test)
print("GradientBoostingRegressor r2_score is : " , r2_score(y_test, predictions))
sns.regplot(x = y_test, y = predictions)

""" ""TypeError: '(31, slice(None, None, None))' is an invalid key" félkész--hibát dob
-------------------------------------------------------------

# Fitting logistic regression for whole dataset
logreg = LogisticRegression(solver='liblinear');  # instance of the class
logreg.fit(data_without_price,price);  #  fitting the model to data
intercept = logreg.intercept_[0]; #  intecept (constant) parameter
weight = logreg.coef_[0,:];   #  regression coefficients (weights)
score = logreg.score(data_without_price,price);  # accuracy of the model

# Prediction by scikit-learn
target_pred = logreg.predict(data_without_price);  
p_pred = logreg.predict_proba(data_without_price)[:,1];
# Prediction by numpy
z = np.dot(data_without_price,weight)+intercept;
p_pred1 = sp.special.expit(z);

# Visualizing the prediction
fig = plt.figure(1);
plt.title('Comparing various prediction methods');
plt.xlabel('Sklearn prediction');
plt.ylabel('Numpy prediction');
plt.scatter(p_pred,p_pred1,s=50,c=price);
plt.show();

# Partitioning for train/test dataset
test_rate = 0.2;
X_train, X_test, y_train, y_test = train_test_split(data_without_price,
        price, test_size=test_rate, random_state=2020);
n_train = X_train.shape[0];
n_test = X_test.shape[0];
# Printing the basic parameters
print(f'Number of training records:{n_train}');
print(f'Number of test records:{n_test}');


# Fitting logistic regression
logreg1 = LogisticRegression(solver='liblinear');
logreg1.fit(X_train,y_train);
intercept1 = logreg1.intercept_[0];
weight = logreg1.coef_[0,:];
score_train = logreg1.score(X_train,y_train);
score_test = logreg1.score(X_test,y_test);

# Prediction of a random test record
ind = np.random.randint(0,n_test);
test_record = X_test[ind,:].reshape(1, -1);
pred_class = logreg1.predict(test_record)[0];
pred_distr = logreg1.predict_proba(test_record);
print('Prediction of test record with index',ind,':',pred_class,'/true: ',y_test[ind]);
print('Prediction of positive class probability:',pred_distr[0,1]);

# Replication analysis of logistic regression model
rep = 1000;
score = [];
logreg = LogisticRegression(solver='liblinear');
for i in range(rep):
    X_train, X_test, y_train, y_test = train_test_split(data_without_price,
        price, test_size=test_rate);
    logreg.fit(X_train,y_train);
    score.append(logreg.score(X_test,y_test));

score_mean = np.mean(score);
score_std = np.std(score);
# Printing the results
print(f'Mean of accuracy:{score_mean}');
print(f'Standard deviation of accuracy:{score_std}');
# Histogram for the accuracy
plt.figure(2);   
count, bins, ignored = plt.hist(np.array(score),10,density=True);
plt.plot(bins,1/(score_std*np.sqrt(2*np.pi))*np.exp(-(bins-score_mean)**2/(2*score_std**2)),linewidth=2,color='red');   
plt.show();
-------------------------------------------------------------
"""


# Klaszterezés k-közép módszerrel

# Default parameters
n_c = 2; # number of clusters


# Kmeans clustering
kmeans = KMeans(n_clusters=n_c, random_state=2020);  # instance of KMeans class
kmeans.fit(data_without_price);   #  fitting the model to data
automobile_labels = kmeans.labels_;  # cluster labels
automobile_centers = kmeans.cluster_centers_;  # centroid of clusters
sse = kmeans.inertia_;  # sum of squares of error (within sum of squares)
score = kmeans.score(data_without_price);  # negative error
# both sse and score measure the goodness of clustering

# Davies-Bouldin goodness-of-fit
DB = davies_bouldin_score(data_without_price,automobile_labels);

# Printing the results
print(f'Number of cluster: {n_c}');
print(f'Within SSE: {sse}');
print(f'Davies-Bouldin index: {DB}');


# PCA with limited components
pca = PCA(n_components=2);
pca.fit(data_without_price);
iris_pc = pca.transform(data_without_price);  #  data coordinates in the PC space
centers_pc = pca.transform(automobile_centers);  # the cluster centroids in the PC space

# Visualizing of clustering in the principal components space
fig = plt.figure();
plt.title('Clustering of the Automobile data after PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(iris_pc[:,0],iris_pc[:,1],s=50,c=automobile_labels);  # data
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');  # centroids
plt.legend();
plt.show();

# Kmeans clustering with K=2
kmeans = KMeans(n_clusters=2, random_state=2020);  # instance of KMeans class
kmeans.fit(data_without_price);   #  fitting the model to data
automobile_labels = kmeans.labels_;  # cluster labels
automobile_centers = kmeans.cluster_centers_;  # centroid of clusters
distX = kmeans.transform(data_without_price);
dist_center = kmeans.transform(automobile_centers);


# Visualizing of clustering in the distance space
fig = plt.figure();
plt.title('Automobile dataset clustering');
plt.xlabel('Cluster 1');
plt.ylabel('Cluster 2');
plt.scatter(distX[:,0],distX[:,1],s=50,c=automobile_labels);  # data
plt.scatter(dist_center[:,0],dist_center[:,1],s=200,c='red',marker='X');  # centroids
plt.legend();
plt.show();



# Finding optimal cluster number
Max_K = 31;  # maximum cluster number
SSE = np.zeros((Max_K-2));  #  array for sum of squares errors
DB = np.zeros((Max_K-2));  # array for Davies Bouldin indeces
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(data_without_price);
    automobile_labels = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(data_without_price,automobile_labels);

# Visualization of SSE values    
fig = plt.figure();
plt.title('Sum of squares of error curve');
plt.xlabel('Number of clusters');
plt.ylabel('SSE');
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show();

# Visualization of DB scores
fig = plt.figure();
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();

# The local minimum of Davies Bouldin curve gives the optimal cluster number

# End of code

