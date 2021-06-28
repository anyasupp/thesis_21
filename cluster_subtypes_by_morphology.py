# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 21:03:58 2020
Cluster 
@author: AnyaS
"""


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%config InlineBackend.figure_format='retina'
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
#%%C:\Users\AnyaS\Documents\Python Scripts\20210219_PK2old_freeRunning\morphology_cluster_combined
#load extracted data
dt = pd.read_excel("20210427_all_condition_cluster_feature.xlsx",sheet_name='Sheet1')
#%%
print(dt.isna().sum())
#fill Nan with 0 if neurons have no proximal arbour
dt['PA_loc'].fillna(0, inplace = True)

#%%
#keep fil length sum / AP span/ distance skin/ darbour thickness/ darbour_loc/ PA location for clustering 
#drop other parameters measured
X = np.array(dt.drop(['Blinded File','Fish_ID','Lam_Profile','total','Body_to_arbour','darbour_to_skin','GFP','RFP_distance',
                      'RFP_Start','RFP_Stop','Full_width','Max_RFP_distance','PA_Distance'], 1).astype(float))

# Standardize the data to have a mean of ~0 and a variance of 1
X_std = StandardScaler().fit_transform(X)
#%%
# Create a PCA instance: pca
pca = PCA()
principalComponents = pca.fit_transform(X_std)
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)

#%%
#cumulative variance plot
#showing how much variance is explained by each of the 6individual components
pca.explained_variance_ratio_
plt.figure(figsize= (10,8))
plt.plot(range(0,6),pca.explained_variance_ratio_.cumsum(), marker ='o',linestyle='--')
plt.title('Explained Variance by Components')
plt.xlabel('No of components')
plt.ylabel('Cumulative Explained Variance')
#
#%%
#find most important features
#get index of the most important feature on EACH compnent
print(abs(pca.components_))
check_feature = abs(pca.components_)
n_pcs=pca.n_components_ #get number of components
most_important = [abs(pca.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = dt.columns
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

#%% Looks like PCA 4 
#perform PCA with the chosen number of components
pca= PCA(n_components= 4)
pca.fit(X_std)
#calculate resulting components scores for the elements in our dataset 
pca.transform(X_std)
scores_pca_4= pca.transform(X_std)
#%%
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12),locate_elbow=True,timing=False)
#green line shows time to train clustering model per K (to hide it timings=False)
visualizer.fit(scores_pca_4)
visualizer.poof()
# gives optimal 4 k means 

#use calinski harabasz
model = KMeans()
visualizer = KElbowVisualizer(model, k=(3,12),metric='calinski_harabasz',timings=False,locate_elbow=True)
visualizer.fit(scores_pca_4)
#gives optimal 4 k means
#%%
#use silhouette
#Silhouette analysis use to evaluate density and separation between clusters - average silhouette coefficient for each sample
#difference between intra-cluster distance and the mean nearest-cluster distance -- then normalized
#scores near +1 = high separation 
#scores near -1 indicate samples may be assigned to wrong cluster
#put in number of clusters and see distribution
fig, ax = plt.subplots(1,4)
plt.subplot(1,4,1)
model = KMeans(3)
visualizer = SilhouetteVisualizer(model)
visualizer.fit(scores_pca_4)
visualizer.poof()

plt.subplot(1,4,2)
model = KMeans(4)
visualizer = SilhouetteVisualizer(model)
visualizer.fit(scores_pca_4)
visualizer.poof()

plt.subplot(1,4,3)
model = KMeans(5)
visualizer = SilhouetteVisualizer(model)
visualizer.fit(scores_pca_4)
visualizer.poof()

plt.subplot(1,4,4)
model = KMeans(6)
visualizer = SilhouetteVisualizer(model)
visualizer.fit(scores_pca_4)
visualizer.poof()
#plt.savefig('20200507_silhouette_2_for_withoutoutliers.jpg', format='jpg', dpi=1200)

# Looks like PCA4 and k means 4 gives the best! 
# Proceed to do clustering
#%%
#use initializer and random state as before 
kmeans_pca = KMeans(n_clusters =4, init ='k-means++',random_state = 22)
#we fir our data with k-means pca model (4)
kmeans_pca.fit(scores_pca_4)


#create a new dataframe with the orginical features and add the pCA scores and assigned clusters
dt_segm_pca_kmeans = pd.concat([dt.reset_index(drop=True),pd.DataFrame(scores_pca_4)], axis=1)
dt_segm_pca_kmeans.columns.values[-4:] = ['Component1','Component2','Component3','Component4']
#last column add pca k-means cluster labels
dt_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_

#see head of our new dataframe
dt_segm_pca_kmeans.head()

#add names of segments to the labels
dt_segm_pca_kmeans['Segment'] = dt_segm_pca_kmeans['Segment K-means PCA'].map({0:'first',1:'second',2:'third',3:'fourth'})
#%%
# dt_segm_pca_kmeans.to_excel("20201127_segmentation_cluster_all_condition_PCA4keamsn4.xlsx")
# %%
#how to visualize cluster by components
#plot data by PCA components - y axis is first, x is second
x_axis = dt_segm_pca_kmeans['Component2']
y_axis = dt_segm_pca_kmeans['Component1']
plt.figure(figsize=(10,8))
# palette = ['lightseagreen','orangered', 'dodgerblue','gold','mediumvioletred']
palette = ['lightseagreen','orangered', 'dodgerblue','gold']

sns.scatterplot(x_axis,y_axis,hue=dt_segm_pca_kmeans['Segment'],palette=palette)
plt.title('Cluster by PCA Components')
plt.show()


#%%
#THIS IS FOR 5
#fig, ax = plt.subplots(4, 4)
C2 = dt_segm_pca_kmeans['Component2']
C1 = dt_segm_pca_kmeans['Component1']
C3 = dt_segm_pca_kmeans['Component3']
C4 = dt_segm_pca_kmeans['Component4']
# C5 = dt_segm_pca_kmeans['Component5']
plt.figure()
#plt.title('K Means segmentation using PCA')
ax = plt.subplot(3,3,1)
ax = sns.scatterplot(C2,C1,hue=dt_segm_pca_kmeans['Segment'],palette=palette)
ax.get_legend().remove()
#plt.xlim(-3.75,5.5 )
#plt.ylim(-2.75,3.5 )
sns.despine()

ax = plt.subplot(3,3,2)
ax = sns.scatterplot(C3,C1,hue=dt_segm_pca_kmeans['Segment'],palette=palette)
ax.get_legend().remove()
#plt.xlim(-3.75,5.5 ) 
#plt.ylim(-2.75,3.5 )
ax.set_xlabel('') 
ax.set_ylabel('')  
ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off

sns.despine()

ax = plt.subplot(3,3,3)
ax = sns.scatterplot(C4,C1,hue=dt_segm_pca_kmeans['Segment'],palette=palette)
ax.get_legend().remove()
#plt.xlim(-2,3 ) 
#plt.ylim(-2.75,3.5 )
ax.set_xlabel('') 
ax.set_ylabel('')
ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
sns.despine()



ax = plt.subplot(3,3,5)
ax = sns.scatterplot(C3,C2,hue=dt_segm_pca_kmeans['Segment'],palette=palette)
ax.get_legend().remove()
#ax.set_xlabel('') 
#ax.set_ylabel('')  
#plt.xlim(-2,3)

sns.despine()

ax = plt.subplot(3,3,6)
ax = sns.scatterplot(C4,C2,hue=dt_segm_pca_kmeans['Segment'],palette=palette)
#plt.xlim(-2,3 )
ax.set_xlabel('') 
ax.set_ylabel('')
ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
sns.despine()
ax.get_legend().remove()
sns.despine()


ax = plt.subplot(3,3,9)
ax = sns.scatterplot(C4,C3,hue=dt_segm_pca_kmeans['Segment'],palette=palette)
#plt.xlim(-2,3 )
ax.get_legend().remove()
sns.despine()


# plt.savefig('20201121_combinedLD-LL_cluster_morphology_PCA5-kmeans5.jpg', format='jpg', dpi=1200)
#%%

# See what percentage if each cluster in conditon 
#cluster_name_change_to match the rest
df = pd.read_excel("20210428_segmentation_cluster_all_condition_PCA4keamsn4_change_cluster_name.xlsx",sheet_name='Sheet1')

#%%
# data = df[['Fish_ID','Conditions','Segment K-means PCA']]
data = df[['Fish_ID','Conditions','Segment Cluster']]
#%%prepare data for Plot stack bar compare cluster
count=data.groupby(['Conditions','Segment Cluster']).size() #group everything by label and segmenta and count
count=pd.DataFrame(count)
count = count.rename({0:'count'},axis=1)
count=count.reset_index() #reset index so can move things around
#%%
count_table=count.pivot(index='Segment Cluster',columns='Conditions')['count']
count_table= count_table.fillna(0)#fill nan to zero
count_table=count_table.reset_index()
#remove decimals in plot 
count_table=count_table.astype(int)
count_table=count_table.round()

#%%
from matplotlib.colors import ListedColormap
palette = ['lightseagreen','orangered', 'dodgerblue','gold']
sns.set_context("talk")
sns.set_style('ticks')
ax=count_table.set_index('Segment Cluster').T.plot(kind='bar', stacked=True,colormap=ListedColormap(palette))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_xlabel('Conditions')
ax.set_ylabel('Count')
sns.despine()
# ax.get_legend().remove()
# plt.legend(fontsize='x-small',fancybox=True,title='Segmentation cluster',title_fontsize='12')

# plt.legend(fontsize='x-small',fancybox=True,title='Segmentation cluster',title_fontsize='12',ncol=2)
ax.legend(loc='center left', bbox_to_anchor=(0.70,0.95),title='Segmentation cluster')

#%%
# plt.savefig('20210427_conditions_vs_cluster.png', transparent=True, dpi=1200)
#%% Percentrage distribution 
per = count_table.set_index('Segment Cluster')

per['LD_per']= (per['LD']/per['LD'].sum())*100
per['LL_per']= (per['LL']/per['LL'].sum())*100
per['FR_per']= (per['FR']/per['FR'].sum())*100
#%%
percent_df = per[['LD_per','LL_per','FR_per']]
percent_df = percent_df.rename(columns={'LD_per':'LD','LL_per':'LL','FR_per':'FR'})
percent_df = percent_df.reset_index()
#%%
from matplotlib.colors import ListedColormap
palette = ['lightseagreen','orangered', 'dodgerblue','gold']
sns.set_context("talk")
sns.set_style('ticks')
ax=percent_df.set_index('Segment Cluster').T.plot(kind='bar', stacked=True,colormap=ListedColormap(palette))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_xlabel('Conditions')
ax.set_ylabel('Percentage %')
sns.despine()
# ax.get_legend().remove()

legend = ax.legend(loc='center left', bbox_to_anchor=(0.88,0.95),title='Morphology cluster', fontsize=12)
plt.setp(legend.get_title(),fontsize=12)
#%%
# plt.savefig('20210427_conditions_vs_cluster_percentage_n.png', transparent=True, dpi=1200)
#%% STATS
#use chi square with corrections 
#prep data for stats
# have conditions as index and segment cluster as columns
percent_df = percent_df.set_index('Segment Cluster')
percent_df = percent_df.T
#%%
count_df = count_table.set_index('Segment Cluster')
count_df = count_df.T 
#%% use count for stats -- as percentage ditribution have no weight on n number
from scipy.stats import chi2_contingency
chi2, p, dof, ex = chi2_contingency(count_df, correction=True)
print(f"Chi2 result of the contingency table: {chi2}, p-value: {p}")
# Chi2 result of the contingency table: 25.3690095606157, p-value: 0.0002916961582076556
#%% Chi Square post-hoc tests
#look into further which is different?
import itertools
# gathering all combinations for post-hoc chi2
all_combinations = list(itertools.combinations(count_df.index, 2))
print("Significance results:")
for comb in all_combinations:
    # subset df into a dataframe containing only the pair "comb"
    new_df = count_df[(count_df.index == comb[0]) | (count_df.index == comb[1])]
    # running chi2 test
    chi2, p, dof, ex = chi2_contingency(new_df, correction=False)
    print(f"Chi2 result for pair {comb}: {chi2}, p-value: {p}")
# Significance results:
# Chi2 result for pair ('FR', 'LD'): 6.345596133190117, p-value: 0.09595479995380139
# Chi2 result for pair ('FR', 'LL'): 3.6190476190476186, p-value: 0.30564709186087763
# Chi2 result for pair ('LD', 'LL'): 24.660878143021, p-value: 1.817812430660561e-05
#%% correct our results for multiple comparisons. 
# gathering all combinations for post-hoc chi2
all_combinations = list(itertools.combinations(count_df.index, 2))
p_vals = []
for comb in all_combinations:
    # subset df into a dataframe containing only the pair "comb"
    new_df = count_df[(count_df.index == comb[0]) | (count_df.index == comb[1])]
    # running chi2 test
    chi2, p, dof, ex = chi2_contingency(new_df, correction=True)
    p_vals.append(p)
#%%
from statsmodels.sandbox.stats.multicomp import multipletests

reject_list, corrected_p_vals = multipletests(p_vals, method='fdr_bh')[:2]
#%%
#for p value threshold = 0.01, use multipletests(p_vals, method='fdr_bh', alpha=0.01)).
print("original p-value\tcorrected p-value\treject?")
for p_val, corr_p_val, reject in zip(p_vals, corrected_p_vals, reject_list):
    print(p_val, "\t", corr_p_val, "\t", reject)
    
# original p-value	corrected p-value	reject?
# 0.09595479995380139 	 0.1439321999307021 	 False
# 0.30564709186087763 	 0.30564709186087763 	 False
# 1.817812430660561e-05 	 5.453437291981683e-05 	 True


#%% use p value asteriks

def get_asterisks_for_pval(p_val):
    """Receives the p-value and returns asterisks string."""
    if p_val > 0.05:
        p_text = "ns"  # above threshold => not significant
    elif p_val < 1e-4:  
        p_text = '****'
    elif p_val < 1e-3:
        p_text = '***'
    elif p_val < 1e-2:
        p_text = '**'
    else:
        p_text = '*'
    
    return p_text
#%%
def chisq_and_posthoc_corrected(count_df):
    """Receives a dataframe and performs chi2 test and then post hoc.
    Prints the p-values and corrected p-values (after FDR correction)"""
    # start by running chi2 test on the matrix
    chi2, p, dof, ex = chi2_contingency(count_df, correction=True)
    print(f"Chi2 result of the contingency table: {chi2}, p-value: {p}")
    
    # post-hoc
    all_combinations = list(itertools.combinations(count_df.index, 2))  # gathering all combinations for post-hoc chi2
    p_vals = []
    print("Significance results:")
    for comb in all_combinations:
        new_df = count_df[(count_df.index == comb[0]) | (count_df.index == comb[1])]
        chi2, p, dof, ex = chi2_contingency(new_df, correction=True)
        p_vals.append(p)
        # print(f"For {comb}: {p}")  # uncorrected

    # checking significance
    # correction for multiple testing
    reject_list, corrected_p_vals = multipletests(p_vals, method='fdr_bh')[:2]
    for p_val, corr_p_val, reject, comb in zip(p_vals, corrected_p_vals, reject_list, all_combinations):
        print(f"{comb}: p_value: {p_val:5f}; corrected: {corr_p_val:5f} ({get_asterisks_for_pval(p_val)}) reject: {reject}")
        
#%%chisq_and_posthoc_corrected(percent_df)
# Chi2 result of the contingency table: 25.3690095606157, p-value: 0.0002916961582076556
# Significance results:
# ('FR', 'LD'): p_value: 0.095955; corrected: 0.143932 (ns) reject: False
# ('FR', 'LL'): p_value: 0.305647; corrected: 0.305647 (ns) reject: False
# ('LD', 'LL'): p_value: 0.000018; corrected: 0.000055 (****) reject: True