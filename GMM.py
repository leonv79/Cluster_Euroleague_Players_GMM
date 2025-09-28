%reset -f
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score


os.chdir("D:/Downloads/Project_Basketball/Players/")

#player_stats_21_22 = pd.read_excel("Players stats_21_22.xlsx",sheet_name='PLAYERS_STATS')
player_stats_21_22 = pd.read_csv("player_stats_21_22_final.csv")
player_stats_22_23 = pd.read_csv("player_stats_22_23_final.csv")
player_stats_23_24 = pd.read_csv("player_stats_23_24_final.csv")
player_stats_24_25 = pd.read_csv("player_stats_24_25_final.csv")


corr_21_22 = player_stats_21_22.corr(numeric_only=True)
corr_22_23 = player_stats_22_23.corr(numeric_only=True)
corr_23_24 = player_stats_23_24.corr(numeric_only=True)
corr_24_25 = player_stats_23_24.corr(numeric_only=True)


from sklearn.preprocessing import StandardScaler
features_to_exclude = ['RNK','NAME','ROLE','NAT','HEIGHT','AGE','TM NAME','MIN','GP']
features_to_scale = player_stats_24_25.columns.difference(features_to_exclude)

#scale all the numeric values
scaler = StandardScaler()
player_stats_24_25[features_to_scale] = scaler.fit_transform(player_stats_24_25[features_to_scale])
player_stats_23_24[features_to_scale] = scaler.fit_transform(player_stats_23_24[features_to_scale])

features_to_exclude = ['RNK','NAME','ROLE','NAT','HEIGHT','AGE','TM NAME','MIN','GP']
features_to_scale2 = player_stats_22_23.columns.difference(features_to_exclude)
player_stats_22_23[features_to_scale2] = scaler.fit_transform(player_stats_22_23[features_to_scale2])

features_to_scale3 = player_stats_21_22.columns.difference(features_to_exclude)
player_stats_21_22[features_to_scale3] = scaler.fit_transform(player_stats_21_22[features_to_scale3])


#At first we will keep players with more than 15 games played
player_stats_21_22 = player_stats_21_22[player_stats_21_22['GP']>15]
player_stats_22_23 = player_stats_22_23[player_stats_22_23['GP']>15]
player_stats_23_24 = player_stats_23_24[player_stats_23_24['GP']>15]
player_stats_24_25 = player_stats_24_25[player_stats_24_25['GP']>15]


sns.histplot(player_stats_21_22,x='MIN',kde=True)
plt.show()
sns.histplot(player_stats_22_23,x='MIN',kde=True)
plt.show()
sns.histplot(player_stats_23_24,x='MIN',kde=True)
plt.show()
sns.histplot(player_stats_24_25,x='MIN',kde=True)
plt.show()

#Judging from the plot and from intuition we want players that contribute in a meaningful way. So we will exclude players with less than 10 mpg.
player_stats_21_22 = player_stats_21_22[player_stats_21_22['MIN']>10]
player_stats_22_23 = player_stats_22_23[player_stats_22_23['MIN']>10]
player_stats_23_24 = player_stats_23_24[player_stats_23_24['MIN']>10]
player_stats_24_25 = player_stats_24_25[player_stats_24_25['MIN']>10]

#Since our data are ready we will proceed with clustering
#What we need at first is to determnine the predictors we need to keep, as using k means doesn't work great on high dimensions + the lack of interpretability

selected_features = ['FT%','USG%','OR%','TR%','AST%','eFG%','PTS','2PTA','3PTA','FTA','TO%','ST%','BLK%','PF','WIN SHARE','DEF WIN SHARE','OBPM','BPM']


#After K-means let's try Gasussian Mixture models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

#To choose the right number of clusters we can check BIC/AIC metrics. 
for k in range(1, 15):
    gmm = GaussianMixture(n_components=k, random_state=42)
    X = player_stats_24_25[selected_features]
    gmm.fit(player_stats_24_25[selected_features])
    print(f"Components: {k}, BIC: {gmm.bic(X):.2f}, AIC: {gmm.aic(X):.2f}")
    
#K=11 seems like a good chjoice as it coincides with what we did for Kmeans and also fits with what the criteria are telling us
gmm = GaussianMixture(n_components=11, random_state=42, covariance_type='full')
player_stats_24_25['cluster_GMM'] = gmm.fit_predict(player_stats_24_25[selected_features])
players_GMM = player_stats_24_25[['NAME','cluster_GMM']]
cluster_centers = pd.DataFrame(gmm.means_,columns=selected_features)
cluster_centers_2 = cluster_centers.melt(var_name="Feature", value_name="Value")
n_clusters = cluster_centers.shape[0]
n_features = cluster_centers.shape[1]

cluster_centers_2['cluster'] = np.tile(np.arange(0,n_clusters),n_features)



import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'




for i in range(0,n_clusters):
    cluster_1 = cluster_centers_2[(cluster_centers_2['cluster']==i) ]
    fig = px.strip(data_frame=cluster_centers_2,x="Feature",
                   y="Value",template='plotly_dark',stripmode="group")

    if not cluster_1.empty:
        fig.add_trace(go.Scatter(
            x=cluster_1["Feature"],
            y=cluster_1["Value"],
            mode='markers+text',
            marker=dict(
                size=12,
                color='red',
                symbol='circle'),
            text=f'cluster {i}',
            textposition='top center'
            ))
        
    fig.show()    


for i in range(0,n_clusters):
    plt.figure(figsize=(16,6))
    sns.stripplot(data=cluster_centers_2,x="Feature",y="Value",jitter=False)
    
    cluster = cluster_centers_2[cluster_centers_2['cluster']==i]
    plt.scatter(
        cluster["Feature"], 
        cluster["Value"], 
        s=120,           # marker size
        color="red",     # marker color
        marker="o",      # marker shape
        label=f"cluster {i}"
        )
    
    
    plt.title(f'PCA Coefficients for cluster {i}')
   
    
    for j, row in cluster.iterrows():
        plt.text(row["Feature"], row["Value"]+0.1, f"cluster {i}",
                 ha="center", va="bottom")
    plt.legend()
    plt.show()


    



for team in player_stats_24_25['TM NAME'].unique():
    print(f'{team} clusters')
    print(player_stats_24_25['TM NAME'][player_stats_24_25['TM NAME']==team].groupby(player_stats_24_25['cluster_GMM']).count().reset_index())
    df = player_stats_24_25['TM NAME'][player_stats_24_25['TM NAME']==team].groupby(player_stats_24_25['cluster_GMM']).count().reset_index()
    sns.barplot(data=df,x='cluster_GMM',y='TM NAME')
    plt.title(f'{team} cluster distribution')
    plt.show()


#name the clusters
cluster_names = {
    0: "Starting pg caliber",
    1: "The good (but not great) Big Men",
    2: "The one step before superstar Guards/Forwards",
    3: "The superstar cluster",
    4: "The 'OK' Guards",
    5: "The(mostly) empty stats players",
    6: "The fill the roster cluster (F/Cs)",
    7: "3nD cluster",
    8: "The fill the roster cluster (G/Fs)",
    9: "Elite Cs",
    10: "The necessary tools cluster"
}

player_stats_24_25['cluster_names'] = player_stats_24_25['cluster_GMM'].map(cluster_names)
players_GMM['cluster_names'] = players_GMM['cluster_GMM'].map(cluster_names)

cluster_colors = {
    'The necessary tools cluster': "#1f77b4",             # blue
    "The 'OK' Guards": "#ff7f0e",                        # orange
    'The fill the roster cluster (G/Fs)': "#2ca02c",     # green
    'The good (but not great) Big Men': "#d62728",       # red
    'Starting pg caliber': "#9467bd",                    # purple
    'The one step before superstar Guards/Forwards': "#8c564b",  # brown
    'The(mostly) empty stats players': "#e377c2",        # pink
    'Elite Cs': "#7f7f7f",                               # gray
    'The fill the roster cluster (F/Cs)': "#bcbd22",     # olive
    '3nD cluster': "#17becf",                            # cyan
    'The superstar cluster': "#FFD700"                   # gold
}


player_stats_24_25["color"] = player_stats_24_25["cluster_names"].map(cluster_colors)




for team in player_stats_24_25['TM NAME'].unique():
    plt.figure(figsize=(17,6))
      # rotation makes them angled, fontsize controls size
    fig, ax = plt.subplots(figsize=(20,6))
    
    bg_image = plt.imread(f'{team}.png')
    ax.imshow(bg_image, extent=[ax.get_xlim()[0],ax.get_xlim()[0]+0.7, 3.5, 5], aspect='auto', zorder=-1)
    plt.xticks(rotation=45, fontsize=10)
    print(f'{team} clusters')
    print(player_stats_24_25['TM NAME'][player_stats_24_25['TM NAME']==team].groupby(player_stats_24_25['cluster_names']).count().reset_index())
    df = player_stats_24_25['TM NAME'][player_stats_24_25['TM NAME']==team].groupby(player_stats_24_25['cluster_names']).count().reset_index()
    sns.barplot(data=df,x='cluster_names',y='TM NAME',palette=cluster_colors)
    plt.ylabel(f'{team}')
    plt.title(f'{team} cluster distribution')
    plt.show()






#players_GMM.to_csv('players_cluster.csv')
players_GMM.head()
#pd.crosstab(players_GMM["NAME"], players_GMM["cluster_names"]).to_csv('players_cluster.csv')






from sklearn.decomposition import PCA

#to find the smallest suitable number of components we need to use a scree plot
pca = PCA(n_components=2)
pca.fit(player_stats_24_25[selected_features])
pca.explained_variance_ratio_.sum()
pca.explained_variance_
plt.plot(range(1,len(pca.explained_variance_ratio_)+1),np.cumsum(pca.explained_variance_ratio_),marker='o')
plt.xlabel('n_components')
plt.ylabel('expaind_variance')
plt.title('scree plot')
plt.show()

pca.components_




X_pca = pca.fit_transform(player_stats_24_25[selected_features])
plt.figure( dpi=150)
#Visualize the results with PCA
plt.scatter(X_pca[:,0], X_pca[:,1], c=player_stats_24_25['cluster_GMM'], cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KMeans Clusters (PCA-reduced data)')
plt.show()