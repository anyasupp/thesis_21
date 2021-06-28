# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:56:53 2021
FingR intensity ratio GFP:mKate2f 
@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%% load data 
mean_in = pd.read_excel("20210422_puncta_intensity_all.xlsx",sheet_name='3T_mean_intensity')
max_stack = pd.read_excel("20210422_puncta_intensity_all.xlsx",sheet_name='3T_max_stack')
min_stack = pd.read_excel("20210422_puncta_intensity_all.xlsx",sheet_name='3T_min_stack')

mkate_mean_in = pd.read_excel("20210422_puncta_intensity_all.xlsx",sheet_name='Check_same_laser')
# mkate_mean_in = pd.read_excel("20210422_puncta_intensity_all.xlsx",sheet_name='mkate2f_mean')
mkate_max_stack = pd.read_excel("20210422_puncta_intensity_all.xlsx",sheet_name='mkate2f_max')

get_cond = pd.read_excel("20210422_puncta_intensity_all.xlsx",sheet_name='3T_mean_intensity')


#%%
mean_in = mean_in.set_index(['Fish_ID'])
max_stack = max_stack.set_index(['Fish_ID'])
min_stack = min_stack.set_index(['Fish_ID'])
mkate_mean_in = mkate_mean_in.set_index(['Fish_ID'])
mkate_max_stack = mkate_max_stack.set_index(['Fish_ID'])

mean_in=mean_in.drop('Condition', axis=1)
max_stack=max_stack.drop('Condition', axis=1)
min_stack=min_stack.drop('Condition', axis=1)
mkate_mean_in=mkate_mean_in.drop('Condition', axis=1)
mkate_max_stack=mkate_max_stack.drop('Condition', axis=1)

normalized_mean_in = (mean_in-min_stack)/(max_stack-min_stack)
mkate_normalized_mean_in = (mkate_mean_in-min_stack)/(mkate_max_stack-min_stack)
#%% calculate mkate/gfp ratio 
puncta_int_ratio = normalized_mean_in/mkate_normalized_mean_in

#%% calculate RoC

dataprep = puncta_int_ratio.T
# do RoC
roc = dataprep.pct_change(limit=1)*100
#%%melt 
puncta_int_ratio =pd.melt(puncta_int_ratio.reset_index(),id_vars='Fish_ID',value_vars=['7dpf_0','7dpf_10','8dpf_0'],
                          value_name='int_ratio',var_name='Time')
#%%
roc = roc.T
roc.replace(0, np.nan,inplace=True)
#%%
ratio_roc =pd.melt(roc.reset_index(),id_vars='Fish_ID',value_vars=['7dpf_0','7dpf_10','8dpf_0'],
                          value_name='ratio_roc',var_name='Time')
#get_cond
condition =pd.melt(get_cond,id_vars=['Fish_ID','Condition'],value_vars=['7dpf_0','7dpf_10','8dpf_0'],
                          value_name='raw',var_name='Time')
#%%combine into one df
puncta_int_ratio.insert(2,'ratio_roc',ratio_roc.ratio_roc)
puncta_int_ratio.insert(3,'Condition',condition.Condition)

#%%take out neurons using diff laser power *check lab book notes*
#decided to take out whole set of neurons even most of the time only last time point had diff laser power.
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F3_1210']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F10_0310']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F11_0310']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F2_0310']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F3_0310']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F4_0310']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F8_20200915_LL']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F2_20200915_LL']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F9_20200915_LL']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F7_20201006_LL']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F9_20201006_LL']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F5_20201006_LL']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F10_20210113_FR']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F8_20210209_FR']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F4_20210209_FR']
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F5_20210209_FR']
#remove empty cell without intensity
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F5_1112']
#%% save to excel 
# puncta_int_ratio.to_excel("20210527_intensity_ratio.xlsx")
#%% seperate all into conditions
groups=puncta_int_ratio.groupby('Condition')

#make new dataframe with cyan,green,red with correct fish ID
LD_group = [groups.get_group('LD')]
LD_group = pd.concat(LD_group)

LL_group = [groups.get_group('LL')]
LL_group = pd.concat(LL_group)


FR_group = [groups.get_group('FR')]
FR_group = pd.concat(FR_group)
#%%check ns
len(LD_group[LD_group.Time=='7dpf_0'])

#%%
palette = ['#5ec0eb' ,'#dd90b5','#00ac87']
# ll = ['#dd90b5']
# ld= ['#5ec0eb']
# fr = ['#00ac87']
#%%#baseline at zero
x1,y1 = [-0.5,3],[0,0]
#%%
sns.set_context("paper")
# sns.set_style('whitegrid')
rows = 2
cols = 4
fig, ax = plt.subplots(rows,cols)
fig.set_size_inches(12,6)

plt.subplot(rows,cols,1)
ax = sns.pointplot(x='Time', y='int_ratio',data=puncta_int_ratio,hue='Condition',ci=68,linestyles='--',dodge=True,palette=palette)
ax.axvspan(-1, -0.1, alpha=0.15, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.15, color='grey')
plt.xlim(-0.5,2.5 ) 
ax.get_legend().remove()
sns.despine()
ax.set_xlabel('')
ax.tick_params(axis='both',which='major',labelsize=12)

plt.subplot(rows,cols,2)
ax = sns.pointplot(x='Time', y='int_ratio',data=LD_group,hue='Fish_ID',linestyles='--',dodge=False,color='#5ec0eb')
ax.axvspan(-1, -0.1, alpha=0.3, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.3, color='grey')
ax.get_legend().remove()
plt.xlim(-0.5,2.5 ) 
sns.despine()
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(axis='both',which='major',labelsize=12)

plt.subplot(rows,cols,3)
ax = sns.pointplot(x='Time', y='int_ratio',data=LL_group,hue='Fish_ID',linestyles='--',dodge=False,color='#dd90b5')
ax.axvspan(-1, -0.1, alpha=0.3, color='khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='khaki')
ax.get_legend().remove()
plt.xlim(-0.5,2.5 ) 
sns.despine()
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(axis='both',which='major',labelsize=12)

plt.subplot(rows,cols,4)
ax = sns.pointplot(x='Time', y='int_ratio',data=FR_group,hue='Fish_ID',linestyles='--',dodge=False,color='#00ac87')
ax.axvspan(-1, -0.1, alpha=0.3, color='khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='khaki')
ax.get_legend().remove()
plt.xlim(-0.5,2.5 ) 
sns.despine()
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(axis='both',which='major',labelsize=12)
#
ymin= -85
ymax= 320
plt.subplot(rows,cols,5)
plt.plot(x1,y1,'k--',alpha=0.35)
ax = sns.pointplot(x='Time', y='ratio_roc',data=puncta_int_ratio[puncta_int_ratio.Time!='7dpf_0'],
                   palette = palette,hue='Condition',ci=68,linestyles='--',dodge=True,
                   join=False)
ax.axvspan(0.175, 0.95, alpha=0.15, color='grey')
ax.tick_params(axis='both',which='major',labelsize=12)
    
plt.xlim(-0.5,1.5 )    
sns.despine()
ax.set_ylabel(' RoC')
ax.get_legend().remove()
ax.set_xlabel('')
# ax.set_xticks([])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.tick_params(axis='both',which='major',labelsize=12)

plt.subplot(rows,cols,6)
# plt.plot(x1,y1,'k--',alpha=0.35)
ax = sns.pointplot(x='Time', y='ratio_roc',data=LD_group[LD_group.Time!='7dpf_0'],hue='Fish_ID',color=palette[0],linestyles='--')
ax.axvspan(0.175, 0.95, alpha=0.55, color='grey')
sns.despine()
plt.xlim(-0.5,1.5 )  
# plt.ylim(ymin,ymax ) 
ax.set_xlabel('')
ax.get_legend().remove()
ax.set_ylabel('')
plt.ylim(ymin,ymax )  
# # ax.set_yticks([])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.tick_params(axis='both',which='major',labelsize=12)

plt.subplot(rows,cols,7)
# plt.plot(x1,y1,'k--',alpha=0.35)
ax = sns.pointplot(x='Time', y='ratio_roc',data=LL_group[LL_group.Time!='7dpf_0'],hue='Fish_ID',color=palette[1],linestyles='--')
ax.axvspan(0.175, 0.95, alpha=0.3, color='khaki')
sns.despine()
plt.xlim(-0.5,1.5 )  
ax.set_ylabel('')
plt.ylim(ymin,ymax )  
ax.get_legend().remove()
ax.set_xlabel('')
# ax.set_xticks([])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.tick_params(axis='both',which='major',labelsize=12)

plt.subplot(rows,cols,8)
# plt.plot(x1,y1,'k--',alpha=0.35)
ax = sns.pointplot(x='Time', y='ratio_roc',data=FR_group[FR_group.Time!='7dpf_0'],hue='Fish_ID',color=palette[2],linestyles='--')
ax.axvspan(0.175, 0.95, alpha=0.3, color='khaki')
sns.despine()
plt.xlim(-0.5,1.5 )  
ax.tick_params(axis='both',which='major',labelsize=12)
plt.ylim(ymin,ymax )  
ax.get_legend().remove()
ax.set_ylabel('')
# ax.set_yticks([])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel('')

ax.tick_params(axis='both',which='major',labelsize=12)
#%%
# fig.tight_layout()

plt.savefig('20210527_intensity_ratio_noddoge_fixaxes_laser_co rrected_correct.png', transparent=True, dpi=1200)

# %% STATS
import pingouin as pg

sns.set()
sns.pointplot(data=puncta_int_ratio, x='Time', y='int_ratio', hue='Condition', dodge=True, markers=['o', 's','x'],
	      capsize=.1, errwidth=1, palette='colorblind')
#
table = puncta_int_ratio.groupby(['Time','Condition'])['int_ratio'].agg(['mean','std']).round(2)
print(table)
#
aov= pg.mixed_anova(data=puncta_int_ratio,dv='int_ratio',within='Time',between='Condition',subject='Fish_ID')
pg.print_table(aov)


#%% post hocs for RAWas conditions were signifiant 
posthocs = pg.pairwise_ttests(dv='int_ratio', within='Time', between='Condition',
                              subject='Fish_ID', data=puncta_int_ratio)
pg.print_table(posthocs)


#%% FOR RATIO ROC
sns.set()
sns.pointplot(data=puncta_int_ratio[puncta_int_ratio.Time!='7dpf_0'], x='Time', y='ratio_roc', hue='Condition', dodge=True, markers=['o', 's','x'],
	      capsize=.1, errwidth=1, palette='colorblind')
#%%
table = puncta_int_ratio.groupby(['Time','Condition'])['ratio_roc'].agg(['mean','std']).round(2)
print(table)

#%%
aov= pg.mixed_anova(data=puncta_int_ratio[puncta_int_ratio.Time!='7dpf_0'],dv='ratio_roc',within='Time',between='Condition',subject='Fish_ID')
pg.print_table(aov)
#%%

#%%
posthocs = pg.pairwise_ttests(dv='ratio_roc', within='Time', between='Condition',
                              subject='Fish_ID', data=puncta_int_ratio[puncta_int_ratio.Time!='7dpf_0'])
pg.print_table(posthocs)

# ==============
# POST HOC TESTS
# ==============

# Contrast          Time     A        B       Paired    Parametric         T     dof  Tail         p-unc    BF10    hedges
# ----------------  -------  -------  ------  --------  ------------  ------  ------  ---------  -------  ------  --------
# Time              -        7dpf_10  8dpf_0  True      True           2.046  52.000  two-sided    0.046   1.024     0.467
# Condition         -        FR       LD      False     True           0.425  25.966  two-sided    0.675   0.370     0.137
# Condition         -        FR       LL      False     True           1.826  19.438  two-sided    0.083   1.190     0.691
# Condition         -        LD       LL      False     True           1.415  38.655  two-sided    0.165   0.672     0.415
# Time * Condition  7dpf_10  FR       LD      False     True          -0.719  31.613  two-sided    0.478   0.420    -0.209
# Time * Condition  7dpf_10  FR       LL      False     True           2.435  18.424  two-sided    0.025   2.894     0.937
# Time * Condition  7dpf_10  LD       LL      False     True           2.779  31.671  two-sided    0.009   5.728     0.792
# Time * Condition  8dpf_0   FR       LD      False     True           1.352  15.505  two-sided    0.196   0.684     0.538
# Time * Condition  8dpf_0   FR       LL      False     True          -0.028  21.177  two-sided    0.978   0.354    -0.010
# Time * Condition  8dpf_0   LD       LL      False     True          -1.683  32.450  two-sided    0.102   0.927    -0.527

#%%

roc = puncta_int_ratio[puncta_int_ratio.Time!='7dpf_0']

roc_noFR = roc[roc.Condition!='FR']
#%%between LL and LD only
aov= pg.mixed_anova(data=roc_noFR,dv='ratio_roc',within='Time',between='Condition',subject='Fish_ID')
pg.print_table(aov)

#%% repeated ANOVA within each conditions to see if time has diff for RoC 

from statsmodels.stats.anova import AnovaRM
aovrm2way = AnovaRM(LD_group[LD_group.Time!='7dpf_0'],'ratio_roc','Fish_ID',within=['Time'])
res2way = aovrm2way.fit()
print(res2way)

#  Anova
# ==================================
#      F Value Num DF  Den DF Pr > F
# ----------------------------------
# Time  8.3421 1.0000 22.0000 0.0085
# ==================================
#%%
aovrm2way = AnovaRM(LL_group[LL_group.Time!='7dpf_0'],'ratio_roc','Fish_ID',within=['Time'])
res2way = aovrm2way.fit()
print(res2way)


#      Anova
# ==================================
#      F Value Num DF  Den DF Pr > F
# ----------------------------------
# Time  0.6608 1.0000 18.0000 0.4269
# ==================================
#%%
aovrm2way = AnovaRM(FR_group[FR_group.Time!='7dpf_0'],'ratio_roc','Fish_ID',within=['Time'])
res2way = aovrm2way.fit()
print(res2way)
# Anova
# ==================================
#      F Value Num DF  Den DF Pr > F
# ----------------------------------
# Time  0.9115 1.0000 10.0000 0.3622
# ==================================
#%% Cluster morphology 
puncta_int_ratio = pd.read_excel("20210527_intensity_ratio.xlsx")

#%%
puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F5_1112']
groups=puncta_int_ratio.groupby('Condition')

#make new dataframe with cyan,green,red with correct fish ID
LD_group = [groups.get_group('LD')]
LD_group = pd.concat(LD_group)
LL_group = [groups.get_group('LL')]
LL_group = pd.concat(LL_group)
FR_group = [groups.get_group('FR')]
FR_group = pd.concat(FR_group)
#%%
data_combined_3T_no7dpf =  puncta_int_ratio[puncta_int_ratio.Time !='7dpf_0']
LD_group_no7= LD_group[LD_group.Time !='7dpf_0']
LL_group_no7= LL_group[LL_group.Time !='7dpf_0']
FR_group_no7= FR_group[FR_group.Time !='7dpf_0']
#%% take nan out of no7 

#%%
palette = ['lightseagreen','orangered', 'dodgerblue','gold']
palette_notype1 = ['orangered', 'dodgerblue','gold']
rows=2
cols=3
ymin= -75
ymax=120
#%% Figure set up
sns.set_context("talk")
# sns.set_style('ticks')
fig, ax = plt.subplots(rows,cols, sharex=True)
fig.suptitle('Day/Night FingR.PSD95 Intensity dynamics by morphology cluster', fontsize=16)
fig.set_size_inches(11,9.5)
#fig.text(0.5, 0.04, 'Time', ha='center', va='center', fontsize=12)

#  puncta 
ax = plt.subplot(rows,cols,1)
ax = sns.pointplot(x="Time", y="int_ratio", data=LD_group,hue="Cluster_morph",ci=68,dodge=True, 
                   palette = palette_notype1,linestyles='--')
#g = sns.catplot(x="Time", y="Change", data=data2,hue="Fish_ID",ci=68,dodge=True, col = 'Segment K-means PCA',kind='point')
ax.axvspan(-1, -0.1, alpha=0.3, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.3, color='grey')
plt.ylim(0.1,0.7)
plt.xlim(-0.5,2.5) 
ax.get_legend().remove()

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('ratio_int')    

sns.despine()
#
ax = plt.subplot(rows,cols,2)
ax = sns.pointplot(x="Time", y="int_ratio", data=LL_group,hue="Cluster_morph",ci=68,dodge=True,palette=palette,linestyles='--')
ax.axvspan(-1, -0.1, alpha=0.3, color='khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='khaki')
ax.get_legend().remove()
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
plt.xlim(-0.5,2.5) 
plt.ylim(0.1,0.7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
sns.despine()

ax = plt.subplot(rows,cols,3)
ax = sns.pointplot(x="Time", y="int_ratio", data=FR_group,hue="Cluster_morph",ci=68,dodge=True,palette=palette,linestyles='--')
ax.axvspan(-1, -0.1, alpha=0.3, color='khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='khaki')
ax.get_legend().remove()
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
plt.xlim(-0.5,2.5) 
plt.ylim(0.1,0.7)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
sns.despine()
#
#  ROC
 #baseline at zero
x1,y1 = [-0.5,2],[0,0]


ax = plt.subplot(rows,cols,4)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="ratio_roc", data=LD_group_no7,hue="Cluster_morph",ci=68,dodge=True, 
                   palette = palette_notype1,linestyles='--')
#g = sns.catplot(x="Time", y="Change", data=data2,hue="Fish_ID",ci=68,dodge=True, col = 'Segment K-means PCA',kind='point')
plt.xlim(-0.5,1.5 )
ax.axvspan(0.16, 0.92, alpha=0.3, color='grey')  
plt.ylim(ymin,ymax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.get_legend().remove()

# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('RoC (%)')    
sns.despine()

ax = plt.subplot(rows,cols,5)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="ratio_roc", data=LL_group_no7,hue="Cluster_morph",ci=68,dodge=True,palette=palette,linestyles='--')
ax.axvspan(0.16, 0.92, alpha=0.3, color='khaki') 
plt.xlim(-0.5,1.5 )  
plt.ylim(ymin,ymax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
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

ax = plt.subplot(rows,cols,6)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="ratio_roc", data=FR_group_no7,hue="Cluster_morph",ci=68,dodge=True,palette=palette,linestyles='--')
ax.axvspan(0.16, 0.92, alpha=0.3, color='khaki') 
plt.xlim(-0.5,1.5 )
plt.ylim(ymin,ymax)
ax.get_legend().remove()

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
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

sns.despine()

#%%
fig.tight_layout()#%%
#%%
# plt.savefig('20210528_intensity_per_morphology_laser_corrected_FINAL.png',transparent=True, dpi=1200)
