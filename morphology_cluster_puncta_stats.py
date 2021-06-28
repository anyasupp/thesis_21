# -*- coding: utf-8 -*-
"""
Created on Sat May  1 09:43:03 2021

@author: AnyaS
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%C:\Users\AnyaS\Documents\Python Scripts\20210219_PK2old_freeRunning\morphology_cluster_combined\morphology_puncta_dynamics
data = pd.read_excel("20210428_puncta_with_new_cluster_combined.xlsx",sheet_name='Sheet1')
#%%
data['Puncta_delta']=data['Puncta_delta'].fillna(0)


#%%
groups=data.groupby('Condition')

#make new dataframe with cyan,green,red with correct fish ID
LD_group = [groups.get_group('LD')]
LD_group = pd.concat(LD_group)
LL_group = [groups.get_group('LL')]
LL_group = pd.concat(LL_group)
FR_group = [groups.get_group('FR')]
FR_group = pd.concat(FR_group)
#%%
data_combined_3T_no7dpf =  data[data.Time !='dpf7_0']
LD_group_no7= LD_group[LD_group.Time !='dpf7_0']
LL_group_no7= LL_group[LL_group.Time !='dpf7_0']
FR_group_no7= FR_group[FR_group.Time !='dpf7_0']
#%%
palette = ['lightseagreen','orangered', 'dodgerblue','gold']
rows=2
cols=3
#%% Figure set up
sns.set_context("notebook")
sns.set_style('ticks')
fig, ax = plt.subplots(rows,cols, sharex=True)
fig.suptitle('Day/Night FingR.PSD95 dynamics by morphology cluster', fontsize=16)
fig.set_size_inches(8,9)
#fig.text(0.5, 0.04, 'Time', ha='center', va='center', fontsize=12)

#  puncta 
ax = plt.subplot(rows,cols,1)
ax = sns.pointplot(x="Time", y="Puncta", data=LD_group,hue="Cluster_morph",ci=68,dodge=True, palette = palette)
#g = sns.catplot(x="Time", y="Change", data=data2,hue="Fish_ID",ci=68,dodge=True, col = 'Segment K-means PCA',kind='point')
ax.axvspan(-1, -0.1, alpha=0.3, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.3, color='grey')
plt.ylim(70,220)
plt.xlim(-0.5,2.5) 
ax.get_legend().remove()
ax.tick_params(axis='both', which='major', labelsize=12)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('Puncta')    

sns.despine()

ax = plt.subplot(rows,cols,2)
ax = sns.pointplot(x="Time", y="Puncta", data=LL_group,hue="Cluster_morph",ci=68,dodge=True,palette=palette)
ax.axvspan(-1, -0.1, alpha=0.3, color='khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='khaki')
ax.get_legend().remove()
ax.tick_params(axis='both', which='major', labelsize=12)
plt.xlim(-0.5,2.5) 
plt.ylim(70,220)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
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
sns.despine()

ax = plt.subplot(rows,cols,3)
ax = sns.pointplot(x="Time", y="Puncta", data=FR_group,hue="Cluster_morph",ci=68,dodge=True,palette=palette)
ax.axvspan(-1, -0.1, alpha=0.3, color='khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='khaki')
ax.get_legend().remove()

plt.xlim(-0.5,2.5) 
plt.ylim(70,220) 
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
sns.despine()

#  ROC
 #baseline at zero
x1,y1 = [-0.5,2],[0,0]
ax = plt.subplot(rows,cols,4)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=LD_group_no7,hue="Cluster_morph",ci=68,dodge=True, palette = palette)
#g = sns.catplot(x="Time", y="Change", data=data2,hue="Fish_ID",ci=68,dodge=True, col = 'Segment K-means PCA',kind='point')
plt.xlim(-0.5,1.5 )
ax.axvspan(0.16, 0.92, alpha=0.3, color='grey')  
plt.ylim(-40,45)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.get_legend().remove()
ax.tick_params(axis='both', which='major', labelsize=12)
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('RoC (%)')    
sns.despine()

ax = plt.subplot(rows,cols,5)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=LL_group_no7,hue="Cluster_morph",ci=68,dodge=True,palette=palette)
ax.axvspan(0.16, 0.92, alpha=0.3, color='khaki') 
plt.xlim(-0.5,1.5 )  
plt.ylim(-40,45)
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
ax.tick_params(axis='both', which='major', labelsize=12)
sns.despine()

ax = plt.subplot(rows,cols,6)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=FR_group_no7,hue="Cluster_morph",ci=68,dodge=True,palette=palette)
ax.axvspan(0.16, 0.92, alpha=0.3, color='khaki') 
plt.xlim(-0.5,1.5 )
plt.ylim(-40,45)
ax.get_legend().remove()
ax.tick_params(axis='both', which='major', labelsize=12)
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
#%%  pdiff 
ax = plt.subplot(rows,cols,7)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_delta", data=LD_group_no7,hue="Cluster_morph",ci=68,dodge=True, palette = palette)
#g = sns.catplot(x="Time", y="Change", data=data2,hue="Fish_ID",ci=68,dodge=True, col = 'Segment K-means PCA',kind='point')

plt.xlim(-0.5,1.5 )
ax.axvspan(0.16, 0.92, alpha=0.3, color='grey')  
plt.ylim(-50,50)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.get_legend().remove()



# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('Δ Puncta')    
sns.despine()

ax = plt.subplot(rows,cols,8)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_delta", data=LL_group_no7,hue="Cluster_morph",ci=68,dodge=True,palette=palette)
ax.axvspan(0.16, 0.92, alpha=0.3, color='khaki') 
plt.xlim(-0.5,1.5 )
plt.ylim(-50,50)

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,9)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_delta", data=FR_group_no7,hue="Cluster_morph",ci=68,dodge=True,palette=palette)
ax.axvspan(0.16, 0.92, alpha=0.3, color='khaki') 
plt.xlim(-0.5,1.5 )
plt.ylim(-50,50)

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
#%%
fig.tight_layout()
#%%
# plt.savefig('20210501_morhology_subtypes_puncta_Notebok.png',transparent=True, dpi=1200)
#%%STATS
import pingouin as pg
#%% FOR LD
sns.set()
sns.pointplot(data=LD_group, x='Time', y='Puncta', hue='Cluster_morph', dodge=True,
	      capsize=.1, errwidth=1, palette='colorblind')

table = LD_group.groupby(['Time','Cluster_morph'])['Puncta'].agg(['mean','std']).round(2)
print(table)
#
aov= pg.mixed_anova(data=LD_group,dv='Puncta',within='Time',between='Cluster_morph',subject='Fish_ID')
pg.print_table(aov)
'''
     mean    std
Time    Cluster_morph               
dpf7_0  0.0            150.00    NaN
        1.0            152.09  30.30
        2.0            108.50   7.78
        3.0            128.14  40.88
dpf7_10 0.0            109.00    NaN
        1.0            182.45  51.84
        2.0            121.00   9.90
        3.0            134.50  30.54
dpf8_0  0.0            137.00    NaN
        1.0            170.82  39.72
        2.0            124.00   1.41
        3.0            129.36  27.41

=============
ANOVA SUMMARY
=============

Source                SS    DF1    DF2         MS      F    p-unc    np2      eps
-------------  ---------  -----  -----  ---------  -----  -------  -----  -------
Cluster_morph  31706.937      3     24  10568.979  3.530    0.030  0.306  nan
Time            2991.500      2     48   1495.750  3.023    0.058  0.112    0.911
Interaction     3638.647      6     48    606.441  1.225    0.310  0.133  nan
'''
#Has to take MT0 out as only int_combined_3T[int_combined_3T.Time!='7dpf_0']
posthocs = pg.pairwise_ttests(dv='Puncta', within='Time', between='Cluster_morph',
                              subject='Fish_ID', data=LD_group[LD_group.Cluster_morph!=0])
pg.print_table(posthocs)
'''
==============
POST HOC TESTS
==============

Contrast              Time     A        B        Paired    Parametric         T     dof  Tail         p-unc    BF10    hedges
--------------------  -------  -------  -------  --------  ------------  ------  ------  ---------  -------  ------  --------
Time                  -        dpf7_0   dpf7_10  True      True          -2.815  28.000  two-sided    0.009   5.065    -0.393
Time                  -        dpf7_0   dpf8_0   True      True          -1.976  28.000  two-sided    0.058   1.077    -0.242
Time                  -        dpf7_10  dpf8_0   True      True           1.470  28.000  two-sided    0.153   0.519     0.164
Cluster_morph         -        1.0      2.0      False     True           4.613  10.005  two-sided    0.001  28.033     1.357
Cluster_morph         -        1.0      3.0      False     True           2.822  18.722  two-sided    0.011   5.271     1.132
Cluster_morph         -        2.0      3.0      False     True          -1.673  13.012  two-sided    0.118   1.114    -0.439
Time * Cluster_morph  dpf7_0   1.0      2.0      False     True           4.088   8.023  two-sided    0.003  14.706     1.399
Time * Cluster_morph  dpf7_0   1.0      3.0      False     True           1.682  22.949  two-sided    0.106   1.010     0.632
Time * Cluster_morph  dpf7_0   2.0      3.0      False     True          -1.606  11.131  two-sided    0.136   1.055    -0.471
Time * Cluster_morph  dpf7_10  1.0      2.0      False     True           3.588  10.279  two-sided    0.005   8.024     1.154
Time * Cluster_morph  dpf7_10  1.0      3.0      False     True           2.719  15.321  two-sided    0.016   4.431     1.126
Time * Cluster_morph  dpf7_10  2.0      3.0      False     True          -1.255   4.875  two-sided    0.266   0.819    -0.432
Time * Cluster_morph  dpf8_0   1.0      2.0      False     True           3.896  10.135  two-sided    0.003  11.631     1.150
Time * Cluster_morph  dpf8_0   1.0      3.0      False     True           2.953  17.047  two-sided    0.009   6.609     1.203
Time * Cluster_morph  dpf8_0   2.0      3.0      False     True          -0.725  13.428  two-sided    0.481   0.623    -0.192'''

#%%LD ROC LD_group_no7

sns.set()
sns.pointplot(data=LD_group_no7, x='Time', y='RoC', hue='Cluster_morph', dodge=True,
	      capsize=.1, errwidth=1, palette='colorblind')

table = LD_group_no7.groupby(['Time','Cluster_morph'])['RoC'].agg(['mean','std']).round(2)
print(table)
#
aov= pg.mixed_anova(data=LD_group_no7,dv='RoC',within='Time',between='Cluster_morph',subject='Fish_ID')
pg.print_table(aov)
'''  mean    std
Time    Cluster_morph              
dpf7_10 0.0           -27.33    NaN
        1.0            20.27  26.15
        2.0            12.14  17.16
        3.0            10.89  29.00
dpf8_0  0.0            25.69    NaN
        1.0            -2.89  21.37
        2.0             2.87   9.59
        3.0            -1.95  18.49

=============
ANOVA SUMMARY
=============

Source               SS    DF1    DF2        MS      F    p-unc    np2      eps
-------------  --------  -----  -----  --------  -----  -------  -----  -------
Cluster_morph   325.427      3     24   108.476  0.485    0.696  0.057  nan
Time           2856.346      1     24  2856.346  3.144    0.089  0.116    1.000
Interaction    2738.224      3     24   912.741  1.005    0.408  0.112  nan'''
#Has to take MT0 out as only int_combined_3T[int_combined_3T.Time!='7dpf_0']
posthocs = pg.pairwise_ttests(dv='RoC', within='Time', between='Cluster_morph',
                              subject='Fish_ID', data=LD_group_no7[LD_group_no7.Cluster_morph!=0])
pg.print_table(posthocs)
'''==============
POST HOC TESTS
==============

Contrast              Time     A        B       Paired    Parametric         T     dof  Tail         p-unc    BF10    hedges
--------------------  -------  -------  ------  --------  ------------  ------  ------  ---------  -------  ------  --------
Time                  -        dpf7_10  dpf8_0  True      True           2.486  28.000  two-sided    0.019   2.643     0.815
Cluster_morph         -        1.0      2.0     False     True           0.289   4.640  two-sided    0.785   0.559     0.112
Cluster_morph         -        1.0      3.0     False     True           0.982  22.329  two-sided    0.337   0.525     0.379
Cluster_morph         -        2.0      3.0     False     True           0.758   4.463  two-sided    0.487   0.631     0.266
Time * Cluster_morph  dpf7_10  1.0      2.0     False     True           0.562   1.987  two-sided    0.631   0.594     0.297
Time * Cluster_morph  dpf7_10  1.0      3.0     False     True           0.849  22.502  two-sided    0.405   0.481     0.326
Time * Cluster_morph  dpf7_10  2.0      3.0     False     True           0.087   1.957  two-sided    0.939   0.543     0.042
Time * Cluster_morph  dpf8_0   1.0      2.0     False     True          -0.616   3.349  two-sided    0.577   0.605    -0.261
Time * Cluster_morph  dpf8_0   1.0      3.0     False     True          -0.116  19.927  two-sided    0.909   0.372    -0.046
Time * Cluster_morph  dpf8_0   2.0      3.0     False     True           0.574   2.296  two-sided    0.617   0.592     0.253
'''
#%%
#%% FOR LL
sns.set()
sns.pointplot(data=LL_group, x='Time', y='Puncta', hue='Cluster_morph', dodge=True,
	      capsize=.1, errwidth=1, palette='colorblind')

table = LL_group.groupby(['Time','Cluster_morph'])['Puncta'].agg(['mean','std']).round(2)
print(table)
#
aov= pg.mixed_anova(data=LL_group,dv='Puncta',within='Time',between='Cluster_morph',subject='Fish_ID')
pg.print_table(aov)
'''
                         mean    std
Time    Cluster_morph               
dpf7_0  0.0            110.67  14.29
        1.0             97.00    NaN
        2.0            121.00  38.16
        3.0            142.75  68.47
dpf7_10 0.0            108.00  10.86
        1.0            109.00    NaN
        2.0            113.69  30.41
        3.0            146.00  66.58
dpf8_0  0.0            102.67  12.04
        1.0            122.00    NaN
        2.0            112.69  31.97
        3.0            150.50  63.60

=============
ANOVA SUMMARY
=============

Source                SS    DF1    DF2        MS      F    p-unc    np2      eps
-------------  ---------  -----  -----  --------  -----  -------  -----  -------
Cluster_morph  12224.155      3     20  4074.718  1.085    0.378  0.140  nan
Time             244.333      2     40   122.167  0.647    0.529  0.031    0.936
Interaction      923.432      6     40   153.905  0.815    0.565  0.109  nan
'''
#%%
#Has to take MT1 out as only one
posthocs = pg.pairwise_ttests(dv='Puncta', within='Time', between='Cluster_morph',
                              subject='Fish_ID', data=LL_group[LL_group.Cluster_morph!=1])
pg.print_table(posthocs)
'''
==============
POST HOC TESTS
==============

Contrast              Time     A        B        Paired    Parametric         T     dof  Tail         p-unc    BF10    hedges
--------------------  -------  -------  -------  --------  ------------  ------  ------  ---------  -------  ------  --------
Time                  -        dpf7_0   dpf7_10  True      True           0.961  23.000  two-sided    0.346   0.325     0.108
Time                  -        dpf7_0   dpf8_0   True      True           1.664  23.000  two-sided    0.110   0.712     0.164
Time                  -        dpf7_10  dpf8_0   True      True           0.621  23.000  two-sided    0.541   0.256     0.060
Cluster_morph         -        0.0      2.0      False     True          -0.970  12.728  two-sided    0.350   0.581    -0.310
Cluster_morph         -        0.0      3.0      False     True          -1.201   3.014  two-sided    0.316   0.756    -0.884
Cluster_morph         -        2.0      3.0      False     True          -0.904   3.448  two-sided    0.425   0.600    -0.712
Time * Cluster_morph  dpf7_0   0.0      2.0      False     True          -0.855  16.701  two-sided    0.405   0.542    -0.299
Time * Cluster_morph  dpf7_0   0.0      3.0      False     True          -0.924   3.175  two-sided    0.420   0.642    -0.667
Time * Cluster_morph  dpf7_0   2.0      3.0      False     True          -0.607   3.593  two-sided    0.580   0.522    -0.450
Time * Cluster_morph  dpf7_10  0.0      2.0      False     True          -0.597  16.521  two-sided    0.558   0.478    -0.207
Time * Cluster_morph  dpf7_10  0.0      3.0      False     True          -1.132   3.107  two-sided    0.338   0.724    -0.824
Time * Cluster_morph  dpf7_10  2.0      3.0      False     True          -0.941   3.394  two-sided    0.409   0.612    -0.760
Time * Cluster_morph  dpf8_0   0.0      2.0      False     True          -0.989  16.721  two-sided    0.337   0.588    -0.346
Time * Cluster_morph  dpf8_0   0.0      3.0      False     True          -1.487   3.144  two-sided    0.230   0.924    -1.078
Time * Cluster_morph  dpf8_0   2.0      3.0      False     True          -1.145   3.479  two-sided    0.325   0.697    -0.890

'''

#%%LD ROC LL_group_no7

sns.set()
sns.pointplot(data=LL_group_no7, x='Time', y='RoC', hue='Cluster_morph', dodge=True,
	      capsize=.1, errwidth=1, palette='colorblind')
#%%
table = LL_group_no7.groupby(['Time','Cluster_morph'])['RoC'].agg(['mean','std']).round(2)
print(table)
#
aov= pg.mixed_anova(data=LL_group_no7,dv='RoC',within='Time',between='Cluster_morph',subject='Fish_ID')
pg.print_table(aov)

#Has to take MT0 out as only int_combined_3T[int_combined_3T.Time!='7dpf_0']
posthocs = pg.pairwise_ttests(dv='RoC', within='Time', between='Cluster_morph',
                              subject='Fish_ID', data=LL_group_no7[LL_group_no7.Cluster_morph!=1])
pg.print_table(posthocs)
'''  mean    std
Time    Cluster_morph              
dpf7_10 0.0            -0.04  22.76
        1.0            12.37    NaN
        2.0            -4.65  13.45
        3.0             2.74  12.34
dpf8_0  0.0            -4.65   9.26
        1.0            11.93    NaN
        2.0             0.45  17.03
        3.0             5.98  13.47

=============
ANOVA SUMMARY
=============

Source              SS    DF1    DF2       MS      F    p-unc    np2      eps
-------------  -------  -----  -----  -------  -----  -------  -----  -------
Cluster_morph  616.530      3     20  205.510  1.691    0.201  0.202  nan
Time            54.492      1     20   54.492  0.151    0.702  0.007    1.000
Interaction    198.940      3     20   66.313  0.183    0.907  0.027  nan


==============
POST HOC TESTS
==============

Contrast              Time     A        B       Paired    Parametric         T     dof  Tail         p-unc    BF10    hedges
--------------------  -------  -------  ------  --------  ------------  ------  ------  ---------  -------  ------  --------
Time                  -        dpf7_10  dpf8_0  True      True          -0.265  23.000  two-sided    0.793   0.222    -0.091
Cluster_morph         -        0.0      2.0     False     True          -0.048   6.364  two-sided    0.963   0.424    -0.028
Cluster_morph         -        0.0      3.0     False     True          -1.306   6.230  two-sided    0.238   0.812    -0.632
Cluster_morph         -        2.0      3.0     False     True          -2.603   9.590  two-sided    0.027   3.114    -1.044
Time * Cluster_morph  dpf7_10  0.0      2.0     False     True           0.460   6.668  two-sided    0.660   0.456     0.263
Time * Cluster_morph  dpf7_10  0.0      3.0     False     True          -0.249   7.841  two-sided    0.810   0.508    -0.129
Time * Cluster_morph  dpf7_10  2.0      3.0     False     True          -1.024   5.411  two-sided    0.349   0.643    -0.530
Time * Cluster_morph  dpf8_0   0.0      2.0     False     True          -0.842  16.271  two-sided    0.412   0.538    -0.321
Time * Cluster_morph  dpf8_0   0.0      3.0     False     True          -1.377   4.897  two-sided    0.228   0.853    -0.871
Time * Cluster_morph  dpf8_0   2.0      3.0     False     True          -0.673   6.296  two-sided    0.525   0.536    -0.321


'''
#%%
#%% FOR FR
sns.set()
sns.pointplot(data=FR_group, x='Time', y='Puncta', hue='Cluster_morph', dodge=True,
	      capsize=.1, errwidth=1, palette='colorblind')

table = FR_group.groupby(['Time','Cluster_morph'])['Puncta'].agg(['mean','std']).round(2)
print(table)
#
aov= pg.mixed_anova(data=FR_group,dv='Puncta',within='Time',between='Cluster_morph',subject='Fish_ID')
pg.print_table(aov)
'''  mean    std
Time    Cluster_morph              
dpf7_0  0.0             93.0   8.49
        1.0            168.5  24.75
        2.0            141.6  51.39
        3.0            104.4  17.94
dpf7_10 0.0            100.0  42.43
        1.0            192.0  33.94
        2.0            145.2  47.14
        3.0            110.8  17.48
dpf8_0  0.0             96.0   5.66
        1.0            176.0  18.38
        2.0            143.0  57.13
        3.0            104.2  28.97

=============
ANOVA SUMMARY
=============

Source                SS    DF1    DF2         MS      F    p-unc    np2      eps
-------------  ---------  -----  -----  ---------  -----  -------  -----  -------
Cluster_morph  31961.667      3     10  10653.889  2.837    0.092  0.460  nan
Time             478.714      2     20    239.357  0.993    0.388  0.090    0.838
Interaction      320.819      6     20     53.470  0.222    0.965  0.062  nan'''
#%%
sns.set()
sns.pointplot(data=FR_group_no7, x='Time', y='RoC', hue='Cluster_morph', dodge=True,
	      capsize=.1, errwidth=1, palette='colorblind')
#%%
table = FR_group_no7.groupby(['Time','Cluster_morph'])['RoC'].agg(['mean','std']).round(2)
print(table)
#
aov= pg.mixed_anova(data=FR_group_no7,dv='RoC',within='Time',between='Cluster_morph',subject='Fish_ID')
pg.print_table(aov)

#Has to take MT0 out as only int_combined_3T[int_combined_3T.Time!='7dpf_0']
posthocs = pg.pairwise_ttests(dv='RoC', within='Time', between='Cluster_morph',
                              subject='Fish_ID', data=FR_group_no7)
pg.print_table(posthocs)
#%%
# ean    std
# Time    Cluster_morph              
# dpf7_10 0.0             5.89  35.96
#         1.0            16.68  37.28
#         2.0             3.89  14.96
#         3.0             6.60  10.82
# dpf8_0  0.0             4.18  38.54
#         1.0            -7.74   6.73
#         2.0            -3.05   9.93
#         3.0            -6.90  14.77

# =============
# ANOVA SUMMARY
# =============

# Source              SS    DF1    DF2       MS      F    p-unc    np2      eps
# -------------  -------  -----  -----  -------  -----  -------  -----  -------
# Cluster_morph  124.085      3     10   41.362  0.319    0.812  0.087  nan
# Time           852.524      1     10  852.524  1.537    0.243  0.133    1.000
# Interaction    323.389      3     10  107.796  0.194    0.898  0.055  nan


# ==============
# POST HOC TESTS
# ==============

# Contrast              Time     A        B       Paired    Parametric         T     dof  Tail         p-unc    BF10    hedges
# --------------------  -------  -------  ------  --------  ------------  ------  ------  ---------  -------  ------  --------
# Time                  -        dpf7_10  dpf8_0  True      True           1.163  14.000  two-sided    0.264   0.465     0.520
# Cluster_morph         -        0.0      1.0     False     True           0.051   2.000  two-sided    0.964   0.616     0.029
# Cluster_morph         -        0.0      2.0     False     True           1.194   4.425  two-sided    0.293   0.811     0.516
# Cluster_morph         -        0.0      3.0     False     True           1.902   4.771  two-sided    0.118   1.234     0.845
# Cluster_morph         -        1.0      2.0     False     True           0.355   1.252  two-sided    0.773   0.589     0.336
# Cluster_morph         -        1.0      3.0     False     True           0.416   1.115  two-sided    0.743   0.597     0.456
# Cluster_morph         -        2.0      3.0     False     True           0.125   8.000  two-sided    0.904   0.495     0.071
# Time * Cluster_morph  dpf7_10  0.0      1.0     False     True          -0.295   2.000  two-sided    0.796   0.632    -0.168
# Time * Cluster_morph  dpf7_10  0.0      2.0     False     True           0.076   1.142  two-sided    0.950   0.570     0.080
# Time * Cluster_morph  dpf7_10  0.0      3.0     False     True          -0.028   1.073  two-sided    0.982   0.569    -0.032
# Time * Cluster_morph  dpf7_10  1.0      2.0     False     True           0.471   1.132  two-sided    0.713   0.605     0.504
# Time * Cluster_morph  dpf7_10  1.0      3.0     False     True           0.376   1.068  two-sided    0.768   0.592     0.440
# Time * Cluster_morph  dpf7_10  2.0      3.0     False     True          -0.329   8.000  two-sided    0.751   0.509    -0.188
# Time * Cluster_morph  dpf8_0   0.0      1.0     False     True           0.431   2.000  two-sided    0.709   0.650     0.246
# Time * Cluster_morph  dpf8_0   0.0      2.0     False     True           0.262   1.054  two-sided    0.835   0.580     0.314
# Time * Cluster_morph  dpf8_0   0.0      3.0     False     True           0.395   1.120  two-sided    0.755   0.594     0.430
# Time * Cluster_morph  dpf8_0   1.0      2.0     False     True          -0.720   2.941  two-sided    0.525   0.654    -0.421
# Time * Cluster_morph  dpf8_0   1.0      3.0     False     True          -0.103   4.440  two-sided    0.923   0.571    -0.052
# Time * Cluster_morph  dpf8_0   2.0      3.0     False     True           0.484   8.000  two-sided    0.641   0.530     0.276


#%%