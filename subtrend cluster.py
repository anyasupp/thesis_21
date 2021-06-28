# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:40:39 2020
Find subtrend using heirarchical clustering 
@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import pingouin as pg # for stats

data2 = pd.read_excel("20200430_combined_PK_airy.xlsx",sheet_name='allpoints_tracked3points')
#%%
#prep data to have only change, fish id, and time
data = data2.drop(['Puncta','Segment K-means PCA','Comments'],1)
data['time'] = data['Time'].map({'dpf7_0':1,'dpf7_10':2,'dpf8_0':3,'dpf8_10':4,'dpf9_0':5,'dpf9_10':6})
data_new= data.drop(['Time'],1)

#rearrange data to have 'wide format'
original_data=data_new.pivot(index='Fish_ID',columns='time')['Change']
newdata=original_data.dropna() ##drop nan in rows

#%% check Cophenetic Correlation Coefficient
#breifly compare (correlates)the actual pairwaise distance of all your samples to those implied by the hierachical clustering
#closer the value to 1 is the better clusterting presences the original distances 

Z = linkage(newdata,'ward',metric='euclidean')
c, coph_dists = cophenet(Z, pdist(newdata))
print(c) #0.6444079916636003

Z_single = linkage(newdata,'single',metric='euclidean') 
c, coph_dists = cophenet(Z_single, pdist(newdata))
print(c) #0.7141331512962678 #but dendrogram looks odd

Z_complete = linkage(newdata,'complete',metric='euclidean') 
c, coph_dists = cophenet(Z_complete, pdist(newdata))
print(c)#0.7228442394110496

# linkage average coph = 0.7284880321415844
Z_average = linkage(newdata,'average',metric='euclidean') 
c, coph_dists = cophenet(Z_average, pdist(newdata))
print(c)

Z_weighted = linkage(newdata,'weighted',metric='euclidean') 
c, coph_dists = cophenet(Z_weighted, pdist(newdata))
print(c)#0.7235432841957433

Z_cen = linkage(newdata,'centroid',metric='euclidean') 
c, coph_dists = cophenet(Z_cen, pdist(newdata))
print(c)#0.727741979392638

Z_median = linkage(newdata,'median',metric='euclidean') 
c, coph_dists = cophenet(Z_median, pdist(newdata))
print(c)#0.7191738747754349

# Average linkage has highest coph. Use Average. 
#%% look at dendrogram to work out cluster number 
ax = plt.subplot()
plt.title('Hierarchical Cluster Dendrogram_3 time points')
plt.xlabel('Fish_ID')
plt.ylabel('distance')
dendrogram(
    Z_average,
    leaf_rotation=90.,
    leaf_font_size=8.,
    labels=newdata.index,#put Fish_ID back into
    color_threshold=50,above_threshold_color='black' 
)
sns.despine(ax=None,top=True,bottom=True,left=True)
plt.tight_layout()
# plt.savefig('20200522_cluster_trends_3Tpoint_linkagecen.jpg', format='jpg', dpi=1200)
#%% found two cluster is best - extract cluster labels 
k=2
from scipy.cluster.hierarchy import fcluster

clusterslabels_zav= fcluster(Z_average,k,criterion='maxclust')
#Cluster labels for avearge to newdata
newdata.insert(3,'Label_linkaverage',clusterslabels_zav,True)
data_clustered=newdata.reset_index()

#%%Make the right kind of dataframe with Puncta in it as well 

#rearrange data to have 'wide format' so to get the same FishID arrangement as data_clustered_new with correct corresponding puncta and time
data2_new=data2.pivot(index='Fish_ID',columns='Time')['Puncta']
data2_new=data2_new.dropna() ##drop nan in rows
data2_new=data2_new.reset_index() #reset index so can move things around
data2_new= pd.melt(data2_new,id_vars='Fish_ID',value_vars=['dpf7_0','dpf7_10','dpf8_0']) #melt it so to have same arrangement
data2_new=data2_new.rename(columns={'value':'Puncta'}) #cchange name to puncta so not confusing


data_clustered_new= pd.melt(data_clustered,id_vars=['Fish_ID','Label_linkaverage'],value_vars=[1,2,3])
data_clustered_new.insert(4,'Puncta',data2_new.Puncta,True)
data_clustered_new.insert(5,'Time',data2_new.Time,True)
data_clustered_new=data_clustered_new.rename(columns={'value':'Change'}) #cchange name to ROCchange so not confusing

#group everything by cluster again!!
data_clustered_new_grouped=data_clustered_new.groupby('Label_linkaverage')
df_c1 = data_clustered_new_grouped.get_group(1)
df_c2 = data_clustered_new_grouped.get_group(2)

#%%palette/colors
cluster1=sns.color_palette('GnBu_r',20)
cluster2=sns.color_palette('OrRd_r',30)
#%%
# sns.set_context("paper")
sns.set_style('ticks')
fig, ax = plt.subplots(2, 3, sharex=True)

ax = plt.subplot(2,3,4)
ax = sns.pointplot(x="Time", y="Change", data=df_c1, ci=68,color='steelblue') 
ax = sns.pointplot(x="Time", y="Change", data=df_c2, ci=68,color='orangered') 

ax.axvspan(-1, -0.1, alpha=0.3, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.3, color='grey')

plt.xlim(-0.5,2.5)
plt.ylim(-35,75) 
sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('RoC (%)')    
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#
ax = plt.subplot(2,3,5)
ax = sns.pointplot(x="Time", y="Change",hue='Fish_ID', data=df_c1, ci=68,palette=cluster1) 
# ax = sns.pointplot(x="Time", y="value", data=data_cluster_1, ci=68,color='#769c7b') 

ax.axvspan(-1, -0.1, alpha=0.3, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.3, color='grey')

plt.xlim(-0.5,2.5)
plt.ylim(-35,75) 
sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.get_legend().remove()
plt.text(1.5,65,'Cluster 1')
#
ax = plt.subplot(2,3,6)
ax = sns.pointplot(x="Time", y="Change",hue='Fish_ID', data=df_c2, ci=68,palette=cluster2) 

ax.axvspan(-1, -0.1, alpha=0.3, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.3, color='grey')

plt.xlim(-0.5,2.5)
plt.ylim(-35,75) 
sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.get_legend().remove()
plt.text(1.5,65,'Cluster 2')
#
ax = plt.subplot(2,3,1)

ax = sns.pointplot(x="Time", y="Puncta", data=df_c1, ci=68,color='steelblue') 
ax = sns.pointplot(x="Time", y="Puncta", data=df_c2, ci=68,color='orangered') 

ax.axvspan(-1, -0.1, alpha=0.3, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.3, color='grey')

plt.xlim(-0.5,2.5) 
plt.ylim(45,290)
plt.tick_params(labelsize=8)

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

sns.despine()
# ADDED: Remove labels.
#ax.set_ylabel('')    
ax.set_xlabel('')
#
ax = plt.subplot(2,3,2)
ax = sns.pointplot(x="Time", y="Puncta",hue='Fish_ID', data=df_c1, ci=68,palette=cluster1) 
ax.get_legend().remove()
ax.axvspan(-1, -0.1, alpha=0.3, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.3, color='grey')

plt.xlim(-0.5,2.5) 
plt.ylim(45,290)


ax.set_ylabel('')    
ax.set_xlabel('')
#ax1.legend(loc="upper left", markerscale=0.5, fontsize=15)
sns.despine()

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.tick_params(labelsize=8)
plt.text(1.5,265,'Cluster 1')
#
ax = plt.subplot(2,3,3)
ax = sns.pointplot(x="Time", y="Puncta",hue='Fish_ID', data=df_c2, ci=68,palette=cluster2) 
ax.get_legend().remove()
ax.axvspan(-1, -0.1, alpha=0.3, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.3, color='grey')

plt.xlim(-0.5,2.5) 
plt.ylim(45,290)


ax.set_ylabel('')    
ax.set_xlabel('')
#ax1.legend(loc="upper left", markerscale=0.5, fontsize=15)
sns.despine()

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.tick_params(labelsize=8)
plt.text(1.5,265,'Cluster 2')
#%%
# plt.savefig('20210429_3T_average_correct.png', transparent = True, dpi=1200)
#%%stats

sns.set()
sns.pointplot(data=data_clustered_new, x='Time', y='Change', hue='Label_linkaverage', dodge=True, markers=['o', 's'],
	      capsize=.1, errwidth=1, palette='colorblind')
#
print(data_clustered_new[data_clustered_new.Time !='dpf7_0'].groupby(['Time', 'Label_linkaverage'])['Change'].agg(['mean', 'std']).round(4))
#%%
aov= pg.mixed_anova(data=data_clustered_new[data_clustered_new.Time !='dpf7_0'],
                    dv='Change',within='Time',between='Label_linkaverage',subject='Fish_ID')
pg.print_table(aov)
#=============
# ANOVA SUMMARY
# =============

# Source                    SS    DF1    DF2         MS       F    p-unc    np2      eps
# -----------------  ---------  -----  -----  ---------  ------  -------  -----  -------
# Label_linkaverage    346.032      1     28    346.032   1.777    0.193  0.060  nan
# Time                3991.088      1     28   3991.088  18.427    0.000  0.397    1.000
# Interaction        19399.042      1     28  19399.042  89.566    0.000  0.762  nan
#%%
posthocs = pg.pairwise_ttests(dv='Change', within='Time', between='Label_linkaverage',
                              subject='Fish_ID', data=data_clustered_new[data_clustered_new.Time !='dpf7_0'])
pg.print_table(posthocs)
# ==============
# POST HOC TESTS
# ==============

# Contrast                  Time     A        B       Paired    Parametric         T     dof  Tail         p-unc        BF10    hedges
# ------------------------  -------  -------  ------  --------  ------------  ------  ------  ---------  -------  ----------  --------
# Time                      -        dpf7_10  dpf8_0  True      True           2.132  29.000  two-sided    0.042       1.386     0.693
# Label_linkaverage         -        1        2       False     True          -1.375  27.930  two-sided    0.180       0.701    -0.478
# Time * Label_linkaverage  dpf7_10  1        2       False     True          -6.682  27.997  two-sided    0.000   35080.000    -2.311
# Time * Label_linkaverage  dpf8_0   1        2       False     True           7.754  21.154  two-sided    0.000  431500.000     2.899