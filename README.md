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
To decide the number of clusters, I initially used silhouette score and elbow method and then also deployed my domain knowledge. The number of clusters used was 11. It is important though to mention that ideally we want a number of clusters larger than 5, as the clusters resemble more or less to the original 5 positions in basketball. 

<img width="389" height="278" alt="image" src="https://github.com/user-attachments/assets/bd51b48b-48cc-4850-bf36-5a8ff02e9e03" />

<img width="395" height="278" alt="image" src="https://github.com/user-attachments/assets/55d9d39b-fbd6-42c1-9f09-f1c9a39c45e2" />

<img width="392" height="278" alt="image" src="https://github.com/user-attachments/assets/7d864d7b-948e-4a9b-898e-ba286613fec5" />

After settling down on the number of clusters, we move to the clustering part. The challenge here is to actually pick the features that can give the most accuarate deiction. Some researchs focus on more basic stats (PTS,REB,AST), others take advantage of more profound metrics (BPM,VORP,WIN SHARES) or sometimes take into account the antromometric characteristics like height/weight and vertical. 

In this project I used tried different approaches to test the results, but the most satisfactory one was the combination of these metrics. (Important note, that the anthropometric info was not available)
The predictors chosen were: 'Wins', '3PT%', 'FT%', 'USG%', 'OR%','AST%','eFG%','PTS','2PTA','3PTA','FTA','OFF WIN SHARE','DEF WIN SHARE','OBPM','DBPM','VORP'

The above features are an attempt for the model to take into account the defensive part of the game, soemthing that most researchs fail to grasp accurately. It must be noted though, that Blocks and steals were excluded for they seemed to mostly confuse the clusters rather than fixing them. 

The addition also of variables like win shares and wins was an attempt to distinct the players that play well and actually help their team to win. In essence I didn't want 'empty stat' players getting more credit than players who contribute to winning.

Below you can see the cluster centers that came up after the analysis:
<img width="1498" height="916" alt="image" src="https://github.com/user-attachments/assets/fe0a924b-d1c1-4bf8-a00f-4662f5afc813" />

Also how the positions were distributed for each cluster:

<img width="424" height="424" alt="image" src="https://github.com/user-attachments/assets/4e8ed81f-2685-4db8-910c-1e73a89fb6e5" />

In general, K-means does not work well in high dimensions, so the selection of many predictors can mess up the final clustering.

One way to overcome this problem would be Principal Component Analysis (PCA), as it reduces dimensionality while keeping the necessary information. Το keep the necessary information we need to pick the number of components that explain an important part of the original variation. In the following scree plot we see that a 90% variance is covered after 7 components, so that will be the number to use.
<img width="386" height="279" alt="image" src="https://github.com/user-attachments/assets/55f27c98-2f21-429e-80cb-ce54eff9652e" />

After that we apply the K-means model.

#Step 3: The results
The results of the clusters along with their visualization (through PCA) are the following:

K-means with the original data
<img width="801" height="577" alt="image" src="https://github.com/user-attachments/assets/ec416880-8898-44b2-bb07-31b3aa49d66a" />


<img width="1520" height="889" alt="image" src="https://github.com/user-attachments/assets/823f216d-b272-488e-82e6-30001cefed5c" />

K-means results after applying PCA
<img width="801" height="577" alt="image" src="https://github.com/user-attachments/assets/242181ca-36ec-41c9-b42d-0b64c4f9689f" />


<img width="1522" height="624" alt="image" src="https://github.com/user-attachments/assets/3adcd392-2165-4f6d-8902-7daeb82cbef3" />

What seems surprising is that the original data seem to give a clearer distinction for the clusters, despite the high dimensionality. 

However both matrices have clusters that don't seem to make much sense (the marked ones in the pictures), even thoough we can't expect the clusters to be perfect. More speicfically for the PCA results, clusters 1,3,4,7,10 are not easy to interpret, as they have clustered players that seem to have really different characteristics. A notable example is cluster 4, where Darius Thompson (a playmaking PG) and Jan Vesely (a heavier center) are grouped together. 

The original data results are a bit clearer, as already mentioned, but they too present some conflicting clusters with clusters 5 and 10. I think it is pretty evident that even for someone who has been barely following last year's Euroleague, can see that Alberto Abalde and Papagiannis for cluster 5, and Voigtmann and Llull for cluster 10 shouldn't be in the same type of cluster.

#Step 4: Gaussian Mixture Models
So after observing that K-means offered some insight, but not what was expected we will try a different model. A more complex method is Gaussian Mixture Models (GMM). GMM can capture more flexible shapes of clusters, as it assumes that each cluster follows the normal distribution with its own mean and covariance. 

To select the number of clusters in GMMs we will use the AIC/BIC criteria. The results for components 1-15 are the following:
Components: 1, BIC: 4493.11, AIC: 3865.07
Components: 2, BIC: 4534.56, AIC: 3275.13
Components: 3, BIC: 5116.99, AIC: 3226.19
Components: 4, BIC: 5637.82, AIC: 3115.66
Components: 5, BIC: 6247.01, AIC: 3093.47
Components: 6, BIC: 6762.31, AIC: 2977.40
Components: 7, BIC: 6398.83, AIC: 1982.55
Components: 8, BIC: 6731.54, AIC: 1683.89
Components: 9, BIC: 7056.30, AIC: 1377.28
Components: 10, BIC: 7086.95, AIC: 776.55
Components: 11, BIC: 7113.75, AIC: 171.98
Components: 12, BIC: 6531.15, AIC: -1041.99
Components: 13, BIC: 6539.15, AIC: -1665.36
Components: 14, BIC: 7340.04, AIC: -1495.84

The goal here is to find the correct balance of clusters, while keeping a low BIC/AIC. As we mentioned earlier components 1-5 are not taken into consideration, so we will check for clusters wit 6+ components. 
There is a significant drop for AIC with 11 clusters and as it coincides with the number we used in K-means, we will proceed with that number. 
