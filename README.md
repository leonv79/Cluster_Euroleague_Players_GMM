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

Also how the positions were distributed for each cluster
<img width="424" height="424" alt="image" src="https://github.com/user-attachments/assets/4e8ed81f-2685-4db8-910c-1e73a89fb6e5" />
