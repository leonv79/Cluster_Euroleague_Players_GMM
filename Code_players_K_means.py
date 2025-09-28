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
#We will use K means clustering as a simple yet effective method
#What we need at first is to determnine the predictors we need to keep, as using k means doesn't work great on high dimensions + the lack of interpretability

selected_features = ['W','3PT%','FT%','USG%','OR%','TR%','AST%','eFG%','PTS','2PTA','3PTA','FTA','OFF WIN SHARE','DEF WIN SHARE','OBPM','DBPM','VORP']

#After slecting our features we need to decide the optimal number clusters. Some of the main workarounds are silhouette score and elbow method

#silhouette score
silhouette_scores = []
k_values = range(2, len(selected_features))

for k in k_values:
    kmeans = KMeans(n_clusters=k,init='k-means++',random_state=42)
    cluster_labels = kmeans.fit_predict(player_stats_24_25[selected_features])
    silhouette_avg = silhouette_score(player_stats_24_25[selected_features], cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
plt.plot(k_values,silhouette_scores,marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.show()


#elbow method
inertia_values = []


for k in k_values:
    kmeans = KMeans(n_clusters=k,init='k-means++',random_state=42)
    kmeans.fit(player_stats_24_25[selected_features])
    inertia_values.append(kmeans.inertia_)
    
plt.plot(k_values,inertia_values,marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia values')
plt.title('Elbow Method for Optimal k')
plt.show()


#Let's see the inertia values from a closer look
inertia_diff = []
diff = []
for k in range(1, len(inertia_values)):
    diff =inertia_values[k-1]- inertia_values[k]
    inertia_diff.append(diff)

k_values2 = range(1, 15)
plt.plot(k_values2,inertia_diff,marker='o')
plt.xlabel('Inertia difference')
plt.ylabel('Inertia values')
plt.title('Rolling difference for k')
plt.show()

pd.DataFrame(k_values2,inertia_diff)

#It needs to be noted that values of k closer to 5 are not really useful, as the give us results that resemble mostly the 5 common positions basketball. A larger k can give a greater and more intriguing insight



final_k = 11 

kmeans = KMeans(n_clusters=final_k,init='k-means++',random_state=42)
player_stats_24_25['cluster']= kmeans.fit_predict(player_stats_24_25[selected_features])

cluster_centers = pd.DataFrame(kmeans.cluster_centers_,columns=selected_features)
sns.scatterplot(cluster_centers)
plt.show()
#pyplot.plot(cluster_centers)
import matplotlib.pyplot as plt

plt.plot(cluster_centers['W'],range(len(cluster_centers['W'])),marker='o')
plt.show()



#check the counts in each cluster
Value_counts = pd.DataFrame(player_stats_24_25['cluster'].value_counts())
Value_counts = Value_counts.reset_index()
sns.barplot(data=Value_counts,x='cluster',y='count')
plt.title('Counts for each cluster')
plt.show()

'''
sns.stripplot(data=cluster_centers, y='W', color="black", size=3, jitter=False)
plt.xlabel('W')
plt.ylabel('')
plt.show()
sns.boxplot(data=player_stats_24_25,y='W')
plt.boxplot(player_stats_24_25['W'],widths=0.1)
'''




import pandas as pd

# if cluster_centers is a NumPy array, make it a DataFrame first
cluster_centers.reset_index()

long_centers = cluster_centers.melt(var_name="Feature", value_name="Value")





#So let's check the cluster centers

n_clusters = cluster_centers.shape[0]      # number of clusters, e.g. 9
n_features = cluster_centers.shape[1]      # number of features

long_centers = cluster_centers.melt(var_name="Feature", value_name="Value")
long_centers["cluster"] = np.tile(np.arange(0,n_clusters),len(long_centers))[:len(long_centers)]

plt.figure(figsize=(12,6),dpi=150)
sns.stripplot(x="Feature", y="Value", data=long_centers,  size=5, jitter=False,hue='cluster',palette='deep')
plt.xticks(rotation=45)
plt.title("Cluster Centers Across Features")
plt.show()


plt.figure(figsize=(16,6))
sns.stripplot(
    x="Feature", y="Value", data=long_centers,
    jitter=False, s=20, marker="D", linewidth=1, alpha=.1,palette='deep'
)
plt.xticks(rotation=45)
plt.title("Cluster Centers Across Features")
plt.show()





import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
for i in long_centers['cluster'].unique():
    fig = px.strip(data_frame=long_centers,x="Feature",
                   y="Value",template='plotly_dark',stripmode="group",title=f'Cluster {i}')

    #fig.show()

    cluster_1 = long_centers[(long_centers['cluster']==i) ]
    

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
    


    
    

    








'''
import plotly.express as px

fig = px.strip(
    data_frame=long_centers,
    x="Feature",
    y="Value",
    color=long_centers["cluster"].eq(1),   # True/False: cluster==1?
    color_discrete_map={True: "red", False: "gray"},
    stripmode="overlay",
    template="plotly_dark"
)

fig.show()
'''










# Base stripplot
plt.figure(figsize=(16,6))
sns.stripplot(data=long_centers, x="Feature", y="Value", jitter=False)

vb=1
# Highlight cluster 1 points
cluster_1 = long_centers[long_centers["cluster"]==vb]

plt.scatter(
    cluster_1["Feature"], 
    cluster_1["Value"], 
    s=120,           # marker size
    color="red",     # marker color
    marker="o",      # marker shape
    label=f"cluster {vb}"
)

# Add text labels if you want
for i, row in cluster_1.iterrows():
    plt.text(row["Feature"], row["Value"]+0.1, f"cluster {vb}",
             ha="center", va="bottom")

plt.legend()
plt.show()







#Let's check how positions are distributed in the clusters
cluster_role = player_stats_24_25[['cluster','ROLE']]
cluster_role = cluster_role[cluster_role['ROLE'].isin(['PG','SG','SF','PF','C'])]
cluster_role.value_counts()
cluster_role.groupby('ROLE').count().reset_index()
sns.barplot(data=cluster_role.groupby('cluster').count().reset_index(),y='ROLE',x='cluster')

cluster_role.groupby('ROLE').count().reset_index()
cluster_role[cluster_role['cluster']==1].groupby('ROLE').count().reset_index()

cluster_role.groupby(['cluster', 'ROLE']).size()



    
    
for k in range(len(cluster_role['cluster'].unique())):
    counts = cluster_role[cluster_role['cluster'] == k].groupby('ROLE').size().reset_index(name=f'Role for cluster {k}')
    print(counts)





clusters = cluster_role['cluster'].unique()
clusters = np.sort(clusters)
fig, axes = plt.subplots(4, 3, figsize=(6, 6))
axes = axes.flatten()

for i, k in enumerate(clusters):
    counts = cluster_role[cluster_role['cluster'] == k].groupby('ROLE').count().reset_index()
    sns.barplot(data=counts, x='ROLE', y='cluster', ax=axes[i])
    axes[i].set_title(f"Cluster {k}")

plt.tight_layout()
plt.show()


#So now let's see what the actual players are in each cluster

players_clusters = player_stats_24_25[['NAME','cluster']]

players_clusters.head()


# Assuming your dataframe is called players_clusters
clusters_dict = players_clusters.groupby("cluster")["NAME"].apply(list).to_dict()

#clusters_dict[7]


#Results are satisfying but still not great. One reason is the realtively large number of variables and k means is not really great at handling large dimensions.
#One solution is to try Principal component anaylsis (PCA)
from sklearn.decomposition import PCA
selected_features = ['3PT%','FT%','USG%','OR%','TR%','AST%','eFG%','PTS','2PTA','3PTA','FTA','ST%','OBPM','DBPM','VORP']
#to find the smallest suitable number of components we need to use a scree plot
pca = PCA(n_components=7)
pca.fit(player_stats_24_25[selected_features])
pca.explained_variance_ratio_.sum()
pca.explained_variance_
plt.plot(range(1,len(pca.explained_variance_ratio_)+1),np.cumsum(pca.explained_variance_ratio_),marker='o')
plt.xlabel('n_components')
plt.ylabel('expaind_variance')
plt.title('scree plot')
plt.show()


#A safe choice is when the explained variance is around 90%. So we will proceed with 6 n_components


X_pca = pca.fit_transform(player_stats_24_25[selected_features])
kmeans_pca = KMeans(n_clusters=11,init='k-means++',random_state=42)
player_stats_24_25['cluster_pca'] = kmeans_pca.fit_predict(X_pca)



cluster_role_pca = player_stats_24_25[['cluster_pca','ROLE']]
cluster_role_pca = cluster_role_pca[cluster_role_pca['ROLE'].isin(['PG','SG','SF','PF','C'])]
cluster_role_pca.value_counts()
cluster_role_pca.groupby('ROLE').count().reset_index()
sns.barplot(data=cluster_role_pca.groupby('cluster_pca').count().reset_index(),y='ROLE',x='cluster_pca')

cluster_role_pca.groupby('ROLE').count().reset_index()
cluster_role_pca[cluster_role_pca['cluster_pca']==1].groupby('ROLE').count().reset_index()

cluster_role_pca.groupby(['cluster_pca', 'ROLE']).size()



    
    
for k in range(len(cluster_role_pca['cluster_pca'].unique())):
    counts = cluster_role_pca[cluster_role_pca['cluster_pca'] == k].groupby('ROLE').size().reset_index(name=f'Role for cluster {k}')
    print(counts)




cluster_pca = cluster_role_pca['cluster_pca'].unique()
cluster_pca = np.sort(cluster_pca)
fig, axes = plt.subplots(3, 4, figsize=(6, 6))
axes = axes.flatten()

for i, k in enumerate(cluster_pca):
    counts = cluster_role_pca[cluster_role_pca['cluster_pca'] == k].groupby('ROLE').count().reset_index()
    sns.barplot(data=counts, x='ROLE', y='cluster_pca', ax=axes[i])
    axes[i].set_title(f"Cluster {k}")

plt.tight_layout()
plt.show()


players_clusters_pca = player_stats_24_25[['NAME','cluster_pca','cluster']]

players_clusters_pca.head()
clusters_dict_pca = players_clusters_pca.groupby("cluster_pca")["NAME"].apply(list).to_dict()


#Visualize the results with PCA
plt.scatter(X_pca[:,0], X_pca[:,1], c=player_stats_24_25['cluster_pca'], cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KMeans Clusters (PCA-reduced data)')
plt.show()




df_pca_coef = pd.DataFrame(pca.components_,columns=selected_features)
n_clusters_pca = df_pca_coef.shape[0]
n_features_pca = df_pca_coef.shape[1]


df_pca_coef=df_pca_coef.melt(var_name="Feature", value_name="Value")
df_pca_coef['cluster'] = np.tile(np.arange(0,n_clusters_pca),n_features_pca)




# Let's see the centers for the PCA clusters


for i in range(0,n_clusters_pca):
    plt.figure(figsize=(16,6))
    sns.stripplot(data=df_pca_coef, x="Feature", y="Value", jitter=False)
    
    # Highlight cluster 1 points
    cluster_1 = df_pca_coef[df_pca_coef["cluster"]==i]

    plt.scatter(
        cluster_1["Feature"], 
        cluster_1["Value"], 
        s=120,           # marker size
        color="red",     # marker color
        marker="o",      # marker shape
        label=f"cluster {i}"
    )
    plt.title(f'PCA Coefficients for cluster {i}')
    

    # Add text labels if you want
    for j, row in cluster_1.iterrows():
        plt.text(row["Feature"], row["Value"]+0.1, f"cluster {i}",
                 ha="center", va="bottom")

    plt.legend()
    plt.show()






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



#full
X_pca = pca.fit_transform(player_stats_24_25[selected_features])
plt.figure( dpi=150)
#Visualize the results with PCA
plt.scatter(X_pca[:,0], X_pca[:,1], c=player_stats_24_25['cluster'], cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KMeans Clusters')
plt.show()

#reduced PCA
X_pca = pca.fit_transform(player_stats_24_25[selected_features])
plt.figure( dpi=150)
#Visualize the results with PCA
plt.scatter(X_pca[:,0], X_pca[:,1], c=player_stats_24_25['cluster_pca'], cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KMeans Clusters (reduced data)')
plt.show()