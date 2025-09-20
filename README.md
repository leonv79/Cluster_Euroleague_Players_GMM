# Cluster_Euroleague_Players_GMM
Identifying the main archetypes of players for the Euroleague season 2024-25, initially with k-means clustering and then with GMM

#Step 1: Data Preprocessing 
To begin with all the data came from the free source of Hack a stat website. 
The main challenge there was to preprocess a large amount of data, as I initially began to work wth 5 years of data, even though in the end we end up with only last year's stats. 
Each year had a problem of its own, as many column names were different and had to be tracked every time. 

Also steps used:
-Removing and replacing NAs
-Removing unnecessary columns 
-Scaling numeric values

More details can be found in the Data_prep file. 

#Step 2: Cluster Analysis - Kmeans Clustering
In order for the analysis to be trustworthy, players had to meet the following requirements:
-Play at least 15 games throughout the season
-Play at least 10mpg.

The first method used was K-means clustering, a relatively standard and safe approach. It is also the most common method for basketball clustering. 
To decide the number of clusters, I initially used silhouette score and elbow method and then also deployed my domain knowledge. The number of clusters used was 11.

<img width="389" height="278" alt="image" src="https://github.com/user-attachments/assets/bd51b48b-48cc-4850-bf36-5a8ff02e9e03" />

<img width="395" height="278" alt="image" src="https://github.com/user-attachments/assets/55d9d39b-fbd6-42c1-9f09-f1c9a39c45e2" />

<img width="392" height="278" alt="image" src="https://github.com/user-attachments/assets/7d864d7b-948e-4a9b-898e-ba286613fec5" />

