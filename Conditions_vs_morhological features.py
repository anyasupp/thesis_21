# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:06:30 2021
Compare morphogloical features of larvae rearing in different lighting conditions 
produce swarm plot with SEM (+stats) for each conditions
@author: AnyaS
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
#%%C:\Users\AnyaS\Documents\Python Scripts\20210219_PK2old_freeRunning\morphology_cluster_combined
df = pd.read_excel("20210428_segmentation_cluster_all_condition_PCA4keamsn4_change_cluster_name.xlsx",sheet_name='Sheet1')
#%%
#color matching timepoints data
palette= ['#5ec0eb','#dd90b5','#00ac87']

groups=df.groupby('Conditions')
LD_group = [groups.get_group('LD')]
LD_group = pd.concat(LD_group)

LL_group = [groups.get_group('LL')]
LL_group = pd.concat(LL_group)

FR_group = [groups.get_group('FR')]
FR_group = pd.concat(FR_group)

#%%for all in condition 
filsum_LD = sem(LD_group.Filament_Length_Sum, axis=None, ddof=0,nan_policy = 'omit')
filsum_LL = sem(LL_group.Filament_Length_Sum, axis=None, ddof=0,nan_policy = 'omit') # with 0
filsum_FR = sem(FR_group.Filament_Length_Sum, axis=None, ddof=0,nan_policy = 'omit')

AP_span_LD = sem(LD_group.AP_span, axis=None, ddof=0,nan_policy = 'omit')
AP_span_LL = sem(LL_group.AP_span, axis=None, ddof=0,nan_policy = 'omit') # with 0
AP_span_FR = sem(FR_group.AP_span, axis=None, ddof=0,nan_policy = 'omit')
AP_span_sem = [AP_span_LD,AP_span_LL,AP_span_FR]

Distance_Skin_LD = sem(LD_group.Distance_Skin, axis=None, ddof=0,nan_policy = 'omit')
Distance_Skin_LL = sem(LL_group.Distance_Skin, axis=None, ddof=0,nan_policy = 'omit') # with 0
Distance_Skin_FR = sem(FR_group.Distance_Skin, axis=None, ddof=0,nan_policy = 'omit')
Distance_Skin_sem =  [Distance_Skin_LD,Distance_Skin_LL,Distance_Skin_FR]

Darbour_Thickness_LD = sem(LD_group.Darbour_Thickness, axis=None, ddof=0,nan_policy = 'omit')
Darbour_Thickness_LL = sem(LL_group.Darbour_Thickness, axis=None, ddof=0,nan_policy = 'omit') # with 0
Darbour_Thickness_FR = sem(FR_group.Darbour_Thickness, axis=None, ddof=0,nan_policy = 'omit')
Darbour_Thickness_sem = [Darbour_Thickness_LD,Darbour_Thickness_LL,Darbour_Thickness_FR]

Darbour_loc_LD = sem(LD_group.Darbour_loc, axis=None, ddof=0,nan_policy = 'omit')
Darbour_loc_LL = sem(LL_group.Darbour_loc, axis=None, ddof=0,nan_policy = 'omit') # with 0
Darbour_loc_FR = sem(FR_group.Darbour_loc, axis=None, ddof=0,nan_policy = 'omit')
Darbour_loc_sem = [Darbour_loc_LD,Darbour_loc_LL,Darbour_loc_FR]

PA_loc_LD = sem(LD_group.PA_loc, axis=None, ddof=0,nan_policy = 'omit')
PA_loc_LL = sem(LL_group.PA_loc, axis=None, ddof=0,nan_policy = 'omit') # with 0
PA_loc_FR = sem(FR_group.PA_loc, axis=None, ddof=0,nan_policy = 'omit')
PA_loc_sem = [PA_loc_LD,PA_loc_LL,PA_loc_FR]

# %%
sns.set_style("ticks")
sns.set_context("talk")
fig, ax = plt.subplots(2, 3, sharex=True,figsize=(8,7))
fig.subplots_adjust(wspace=0.5)
fig.suptitle('Condition vs Features_swarm')
# fig.text(0.5,0.04,'Trend Cluster No.',va='center',rotation='horizontal')
x1,y1 = [-0.35,0.35],[LD_group.Filament_Length_Sum.mean(),LD_group.Filament_Length_Sum.mean()]
x2,y2 = [0.65,1.35],[LL_group.Filament_Length_Sum.mean(),LL_group.Filament_Length_Sum.mean()]
x3,y3 = [1.65,2.35],[FR_group.Filament_Length_Sum.mean(),FR_group.Filament_Length_Sum.mean()]

sem = [filsum_LD,filsum_LL,filsum_FR]
x_sem = [0,1,2]
y_sem =[LD_group.Filament_Length_Sum.mean(),LL_group.Filament_Length_Sum.mean(),FR_group.Filament_Length_Sum.mean()]

ax = plt.subplot(2,3,1)
# ax=plt.plot(x1,y1,'k-',alpha=1)
ax= sns.swarmplot(y='Filament_Length_Sum',data=df,x='Conditions',palette=palette)
plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.plot(x3,y3,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=sem, linestyle='None', capsize = 10, elinewidth=1,markeredgewidth=2,color='k')
# (_, caps, _) = plt.errorbar(x_sem, y_sem, yerr=sem, linestyle='None', fmt='')
# for cap in caps:
#     cap.set_color('k')
#     cap.set_markeredgewidth(10)

ax.set_xlabel('')
ax.set_ylabel('Filament Length Sum (μm)')
# plt.errorbar(x, y, linestyle='None', marker='o')

ax.set_xticks([])
sns.despine()
#
ax = plt.subplot(2,3,2)
x1,y1 = [-0.35,0.35],[LD_group.AP_span.mean(),LD_group.AP_span.mean()]
x2,y2 = [0.65,1.35],[LL_group.AP_span.mean(),LL_group.AP_span.mean()]
x3,y3 = [1.65,2.35],[FR_group.AP_span.mean(),FR_group.AP_span.mean()]
y_sem =[LD_group.AP_span.mean(),LL_group.AP_span.mean(),FR_group.AP_span.mean()]


ax= sns.swarmplot(y='AP_span',data=df,x='Conditions',palette=palette)
plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.plot(x3,y3,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=AP_span_sem, linestyle='None', capsize = 10, elinewidth=1,markeredgewidth=2,color='k')
ax.set_xlabel('')
ax.set_ylabel('A-P Span (μm)')
ax.set_xticks([])
sns.despine()

#
ax = plt.subplot(2,3,3)
x1,y1 = [-0.35,0.35],[LD_group.Distance_Skin.mean(),LD_group.Distance_Skin.mean()]
x2,y2 = [0.65,1.35],[LL_group.Distance_Skin.mean(),LL_group.Distance_Skin.mean()]
x3,y3 = [1.65,2.35],[FR_group.Distance_Skin.mean(),FR_group.Distance_Skin.mean()]
y_sem =[LD_group.Distance_Skin.mean(),LL_group.Distance_Skin.mean(),FR_group.Distance_Skin.mean()]

ax= sns.swarmplot(y='Distance_Skin',data=df,x='Conditions',palette=palette)
plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.plot(x3,y3,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=Distance_Skin_sem, linestyle='None', capsize = 10, elinewidth=1,markeredgewidth=2,color='k')
ax.set_xlabel('')
ax.set_ylabel('Distance from Skin (μm)')
ax.set_xticks([])
sns.despine()

#
ax = plt.subplot(2,3,4)
x1,y1 = [-0.35,0.35],[LD_group.Darbour_Thickness.mean(),LD_group.Darbour_Thickness.mean()]
x2,y2 = [0.65,1.35],[LL_group.Darbour_Thickness.mean(),LL_group.Darbour_Thickness.mean()]
x3,y3 = [1.65,2.35],[FR_group.Darbour_Thickness.mean(),FR_group.Darbour_Thickness.mean()]
y_sem =[LD_group.Darbour_Thickness.mean(),LL_group.Darbour_Thickness.mean(),FR_group.Darbour_Thickness.mean()]

ax= sns.swarmplot(y='Darbour_Thickness',data=df,x='Conditions',palette=palette)
plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.plot(x3,y3,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=Darbour_Thickness_sem, linestyle='None', capsize = 10, elinewidth=1,markeredgewidth=2,color='k')
ax.set_xlabel('')
ax.set_ylabel('Distal Arbour Thickness (μm)')

sns.despine()
# ax.legend(title = 'Trend Cluster',loc='center left', bbox_to_anchor=(0.85,1))


ax = plt.subplot(2,3,5)
x1,y1 = [-0.35,0.35],[LD_group.Darbour_loc.mean(),LD_group.Darbour_loc.mean()]
x2,y2 = [0.65,1.35],[LL_group.Darbour_loc.mean(),LL_group.Darbour_loc.mean()]
x3,y3 = [1.65,2.35],[FR_group.Darbour_loc.mean(),FR_group.Darbour_loc.mean()]
y_sem =[LD_group.Darbour_loc.mean(),LL_group.Darbour_loc.mean(),FR_group.Darbour_loc.mean()]

ax= sns.swarmplot(y='Darbour_loc',data=df,x='Conditions',palette=palette)
plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.plot(x3,y3,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=Darbour_loc_sem, linestyle='None', capsize = 10, elinewidth=1,markeredgewidth=2,color='k')
sns.despine()
ax.set_xlabel('')
ax.set_ylabel('Distal Arbour Location')

ax = plt.subplot(2,3,6)
x1,y1 = [-0.35,0.35],[LD_group.PA_loc.mean(),LD_group.PA_loc.mean()]
x2,y2 = [0.65,1.35],[LL_group.PA_loc.mean(),LL_group.PA_loc.mean()]
x3,y3 = [1.65,2.35],[FR_group.PA_loc.mean(),FR_group.PA_loc.mean()]
y_sem =[LD_group.PA_loc.mean(),LL_group.PA_loc.mean(),FR_group.PA_loc.mean()]

ax= sns.swarmplot(y='PA_loc',data=df,x='Conditions',palette=palette)
plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.plot(x3,y3,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=PA_loc_sem, linestyle='None', capsize = 10, elinewidth=1,markeredgewidth=2,color='k')
sns.despine()
ax.set_xlabel('')
ax.set_ylabel('Proximal Arbour Location')
#%%
fig.tight_layout()
# %%
# plt.savefig('features_vs_conditions_swarm_Sem.png', transparent=True, dpi=1200)
#%%Stats
from scipy import stats
import scikit_posthocs as sp

#perform Kruskal-Wallis Test == one way ANOVA but dones't assume normality
#maybe need toi check for normality - as KSW is not as sensitive to outliers than ANOVA will do both
#Filament_Length_Sum
filsum = stats.kruskal(LD_group.Filament_Length_Sum, LL_group.Filament_Length_Sum, FR_group.Filament_Length_Sum)
print(filsum)

# KruskalResult(statistic=7.443477704561417, pvalue=0.024191865171206306)
#if significant do Dunn's also use holm as 'best'
#it is more powerful than bonferroni but has no additional assumptions - both rely on general probability inequalities
print(sp.posthoc_dunn(df, val_col='Filament_Length_Sum', group_col='Conditions', p_adjust = 'holm'))

#           FR        LD        LL
# FR  1.000000  0.166666  0.771734
# LD  0.166666  1.000000  0.037484
# LL  0.771734  0.037484  1.000000

#
#%%
AP_span = stats.kruskal(LD_group.AP_span, LL_group.AP_span, FR_group.AP_span)
print(AP_span)
# KruskalResult(statistic=2.8303851321092566, pvalue=0.24287883872554902)
#%%
Distance_Skin = stats.kruskal(LD_group.Distance_Skin, LL_group.Distance_Skin, FR_group.Distance_Skin)
print(Distance_Skin)
# KruskalResult(statistic=2.105880675154026, pvalue=0.34891032522458243)
#%%
Darbour_Thickness = stats.kruskal(LD_group.Darbour_Thickness, LL_group.Darbour_Thickness, FR_group.Darbour_Thickness)
print(Darbour_Thickness)

# KruskalResult(statistic=24.449313237241963, pvalue=4.907938220912454e-06)
#dunnet's 
print(sp.posthoc_dunn(df, val_col='Darbour_Thickness', group_col='Conditions', p_adjust = 'holm'))
#           FR        LD        LL
# FR  1.000000  0.110819  0.110819
# LD  0.110819  1.000000  0.000003
# LL  0.110819  0.000003  1.000000
#%%
Darbour_loc = stats.kruskal(LD_group.Darbour_loc, LL_group.Darbour_loc, FR_group.Darbour_loc)
print(Darbour_loc)
# KruskalResult(statistic=22.439224671059662, pvalue=1.3408625882546315e-05)
#dunn's
print(sp.posthoc_dunn(df, val_col='Darbour_loc', group_col='Conditions', p_adjust = 'holm'))
#           FR        LD        LL
# FR  1.000000  0.114574  0.114574
# LD  0.114574  1.000000  0.000007
# LL  0.114574  0.000007  1.000000
#%%
PA_loc = stats.kruskal(LD_group.PA_loc, LL_group.PA_loc, FR_group.PA_loc)
print(PA_loc)
# KruskalResult(statistic=4.956284528906391, pvalue=0.08389894295523886)

#%% Try ANOVA one way
from scipy.stats import f_oneway
F, p = f_oneway(LD_group.Filament_Length_Sum,LL_group.Filament_Length_Sum,FR_group.Filament_Length_Sum)
# p= 0.01757265428913826
F, p = f_oneway(LD_group.Darbour_Thickness,LL_group.Darbour_Thickness,FR_group.Darbour_Thickness)
#p= 9.670611904931708e-07
F, p = f_oneway(LD_group.AP_span,LL_group.AP_span,FR_group.AP_span)
#p=0.18972512381301312