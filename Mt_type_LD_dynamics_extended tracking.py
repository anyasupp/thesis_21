# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:56:32 2021

@author: AnyaS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:52:39 2021

@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%C:\Users\AnyaS\Documents\Python Scripts\20210219_PK2old_freeRunning
data = pd.read_excel("20210310_combined_Dataframe_ALL.xlsx",sheet_name='Sheet1')

#%% seperate all into conditions
groups=data.groupby('Condition')

#make new dataframe with cyan,green,red with correct fish ID
LD_group = [groups.get_group('LD')]
LD_group = pd.concat(LD_group)

LL_group = [groups.get_group('LL')]
LL_group = pd.concat(LL_group)


FR_group = [groups.get_group('FR')]
FR_group = pd.concat(FR_group)
#%%
cluster = LD_group.groupby('Morphology_cluster')
# cluster = LL_group.groupby('Cluster_morph')
# cluster = FR_group.groupby('Cluster_morph')

c_zero = cluster.get_group(0)
c_one = cluster.get_group(1)
c_two = cluster.get_group(2)
c_three = cluster.get_group(3)

#%%
palette = ['lightseagreen','orangered', 'dodgerblue','gold']
c0 =palette[0]
c1=palette[1]
c2=palette[2]
c3=palette[3]
color='gray'
# color='khaki'

rows=3
cols=5
#%%
sns.set_style('ticks')
sns.set_context('notebook')
fig, ax = plt.subplots(rows,cols, sharex=True)
fig.suptitle('Day/Night FingR.PSD95 dynamics by morphology cluster LL', fontsize=16)
fig.set_size_inches(15,8)
#baseline at zero
x1,y1 = [-0.5,6.5],[0,0]


ylim=40
ymax=350
ax = plt.subplot(rows,cols,1)
ax = sns.pointplot(x="Time", y="Puncta", data=LD_group,hue="Morphology_cluster",ci=68,dodge=True, palette = palette)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax.tick_params(axis='both', which='major', labelsize=14)  

plt.xlim(-0.5,5.5 )
# plt.ylim(ylim,ymax)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax.set_xlabel('')
ax.set_ylabel('Puncta')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,2)
# ax = sns.pointplot(x="Time", y="Puncta", data=c_zero,hue="Fish_ID",ci=68,dodge=True,color=c0)
# ax.get_legend().remove()
ax = sns.pointplot(x="Time", y="Puncta", data=c_zero,ci=68,dodge=True,color=c0)

ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax)
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    

sns.despine()

ax = plt.subplot(rows,cols,3)
ax = sns.pointplot(x="Time", y="Puncta", data=c_one,hue="Fish_ID",ci=68,dodge=True,color=c1)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,4)
# ax = sns.pointplot(x="Time", y="Puncta", data=c_two,hue="Fish_ID",ci=68,dodge=True,color=c2)
# ax.get_legend().remove()
ax = sns.pointplot(x="Time", y="Puncta", data=c_two,ci=68,dodge=True,color=c2)

ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    

sns.despine()

ax = plt.subplot(rows,cols,5)
ax = sns.pointplot(x="Time", y="Puncta", data=c_three,hue="Fish_ID",ci=68,dodge=True,color=c3)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()


#% ROC 

ylim=-75
ymax=75
ax = plt.subplot(rows,cols,6)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=LD_group,hue="Morphology_cluster",ci=68,dodge=True, palette = palette)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax.tick_params(axis='both', which='major', labelsize=14)  

plt.xlim(-0.5,5.5 )
plt.ylim(-50,50)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax.set_xlabel('')
ax.set_ylabel('RoC(%)')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,7)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=c_zero,ci=68,dodge=True,color=c0)

ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax)
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    

sns.despine()

ax = plt.subplot(rows,cols,8)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=c_one,hue="Fish_ID",ci=68,dodge=True,color=c1)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,9)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=c_two,ci=68,dodge=True,color=c2)

ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    

sns.despine()

ax = plt.subplot(rows,cols,10)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=c_three,hue="Fish_ID",ci=68,dodge=True,color=c3)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
#%pdiff
ylim=-125
ymax=125
ax = plt.subplot(rows,cols,11)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=LD_group,hue="Morphology_cluster",ci=68,dodge=True, palette = palette)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax.tick_params(axis='both', which='major', labelsize=14)  

plt.xlim(-0.5,5.5 )
plt.ylim(-50,50)
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel('')
ax.set_ylabel('Δ Puncta')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,12)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_zero,ci=68,dodge=True,color=c0)
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax)


# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
sns.despine()

ax = plt.subplot(rows,cols,13)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_one,hue="Fish_ID",ci=68,dodge=True,color=c1)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax = plt.subplot(rows,cols,14)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_two,ci=68,dodge=True,color=c2)

ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel('')
ax.set_ylabel('')    
sns.despine()

ax = plt.subplot(rows,cols,15)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_three,hue="Fish_ID",ci=68,dodge=True,color=c3)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()


#%%
fig.tight_layout()
#%%
# plt.savefig('20210503_6t_morphology_subtypes_examined_LD_LINE_big.png', transparent=True, dpi=1200)
#%%

LD_24 =  LD_group[LD_group.Morphology_cluster !=0]
LD_24 =  LD_24[LD_24.Morphology_cluster !=2]
#%%
table = LD_24.groupby(['Time','Condition','Morphology_cluster'])['RoC'].agg(['mean','std']).round(2)
print(table)
#                                       mean    std
# Time    Condition Morphology_cluster              
# dpf7_0  LD        1                     NaN    NaN
#                   3                     NaN    NaN
# dpf7_10 LD        1                   17.25  21.37
#                   3                   13.72  29.31
# dpf8_0  LD        1                   -1.31  24.75
#                   3                   -1.11  19.85
# dpf8_10 LD        1                   12.50  23.37
#                   3                    0.47  10.89
# dpf9_0  LD        1                  -16.06  23.63
#                   3                   -1.55  16.58
# dpf9_10 LD        1                   14.23  11.98
#                   3                    0.56  18.17
table = LD_24.groupby(['time','Condition','Morphology_cluster'])['RoC'].agg(['mean','std']).round(2)
print(table)
#                                        mean    std
# time    Condition Morphology_cluster              
# evening LD        1                   14.84  19.69
#                   3                    5.35  21.59
# morning LD        1                   -8.69  24.59
#                   3                   -1.33  17.80
#%%
table = LD_24.groupby(['Time','Condition','Morphology_cluster'])['Puncta_diff'].agg(['mean','std']).round(2)
print(table)
#    mean    std
# Time    Condition Morphology_cluster              
# dpf7_0  LD        1                     NaN    NaN
#                   3                     NaN    NaN
# dpf7_10 LD        1                   29.22  37.02
#                   3                    7.45  26.99
# dpf8_0  LD        1                  -11.12  43.43
#                   3                   -2.30  27.71
# dpf8_10 LD        1                   21.75  43.66
#                   3                   -3.30  14.19
# dpf9_0  LD        1                  -29.38  48.68
#                   3                   -4.20  21.00
# dpf9_10 LD        1                   17.40  11.57
#                   3                   -1.78  26.44
                  
#   mean    std
# time    Condition Morphology_cluster              
# evening LD        1                   23.82  34.74
#                   3                    1.10  23.05
# morning LD        1                  -20.25  45.55
#                   3                   -3.25  23.95
#%%
palette = ['orangered','gold']

#%%
rows=3
cols=3
sns.set_style('ticks')
sns.set_context('notebook')
fig, ax = plt.subplots(rows,cols, sharex=True)
fig.suptitle('Day/Night FingR.PSD95 dynamics by morphology cluster LD', fontsize=16)
fig.set_size_inches(15,8)
#baseline at zero
x1,y1 = [-0.5,6.5],[0,0]


ylim=40
ymax=350
ax = plt.subplot(rows,cols,1)
ax = sns.pointplot(x="Time", y="Puncta", data=LD_24,hue="Morphology_cluster",ci=68,dodge=True, palette = palette)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax.tick_params(axis='both', which='major', labelsize=14)  

plt.xlim(-0.5,5.5 )
# plt.ylim(ylim,ymax)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax.set_xlabel('')
ax.set_ylabel('Puncta')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,2)
ax = sns.pointplot(x="Time", y="Puncta", data=c_one,hue="Fish_ID",ci=68,dodge=True,color=c1)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,3)
ax = sns.pointplot(x="Time", y="Puncta", data=c_three,hue="Fish_ID",ci=68,dodge=True,color=c3)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()


#% ROC 

ylim=-75
ymax=75
ax = plt.subplot(rows,cols,4)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=LD_24,hue="Morphology_cluster",ci=68,dodge=True, palette = palette)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax.tick_params(axis='both', which='major', labelsize=14)  

plt.xlim(-0.5,5.5 )
plt.ylim(-50,50)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax.set_xlabel('')
ax.set_ylabel('RoC(%)')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,5)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=c_one,hue="Fish_ID",ci=68,dodge=True,color=c1)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,6)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=c_three,hue="Fish_ID",ci=68,dodge=True,color=c3)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
#%pdiff
ylim=-125
ymax=125
ax = plt.subplot(rows,cols,7)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=LD_24,hue="Morphology_cluster",ci=68,dodge=True, palette = palette)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax.tick_params(axis='both', which='major', labelsize=14)  

plt.xlim(-0.5,5.5 )
plt.ylim(-50,50)
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel('')
ax.set_ylabel('Δ Puncta')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,8)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_one,hue="Fish_ID",ci=68,dodge=True,color=c1)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax = plt.subplot(rows,cols,9)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_three,hue="Fish_ID",ci=68,dodge=True,color=c3)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
#%%
# plt.savefig('20210602_example_6T_Types Foxp2.png', transparent=True, dpi=1200)
#%%stats
import pingouin as pg

#%%
sns.set()
sns.pointplot(data=LD_24, x='Time', y='Puncta_diff', hue='Morphology_cluster', dodge=True, markers=['o', 's'],
	      capsize=.1, errwidth=1, palette='colorblind')

aov= pg.mixed_anova(data=LD_24,dv='Puncta_diff',within='time',between='Morphology_cluster',subject='Fish_ID')
pg.print_table(aov)
#%%
=============
ANOVA SUMMARY
=============

Source                    SS    DF1    DF2        MS       F    p-unc    np2      eps
------------------  --------  -----  -----  --------  ------  -------  -----  -------
Morphology_cluster    79.098      1     18    79.098   0.347    0.563  0.019  nan
time                5085.025      1     18  5085.025  13.001    0.002  0.419    1.000
Interaction         4157.742      1     18  4157.742  10.630    0.004  0.371  nan
#%%
posthocs = pg.pairwise_ttests(dv='Puncta_diff', within='time', between='Morphology_cluster',
                              subject='Fish_ID', data=LD_24)
pg.print_table(posthocs)
# ==============
# POST HOC TESTS
# ==============

# Contrast                   time     A        B        Paired    Parametric         T     dof  Tail         p-unc    BF10    hedges
# -------------------------  -------  -------  -------  --------  ------------  ------  ------  ---------  -------  ------  --------
# time                       -        evening  morning  True      True           2.937  19.000  two-sided    0.008   5.894     1.098
# Morphology_cluster         -        1        3        False     True           1.272  14.972  two-sided    0.223   0.697     0.562
# time * Morphology_cluster  evening  1        3        False     True           2.667  34.164  two-sided    0.012   4.713     0.784
# time * Morphology_cluster  morning  1        3        False     True          -1.351  21.538  two-sided    0.191   0.654    -0.473

#%%
df = LD_24
#%%
import statsmodels.api as sm
import statsmodels.formula.api as smf
%matplotlib inline

md = smf.mixedlm('Puncta_diff ~ Time', data=df, groups=df['Morphology_cluster'])
mdf = md.fit()
print(mdf.summary())
#%%

