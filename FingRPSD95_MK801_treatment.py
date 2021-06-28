# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:13:13 2019
- Calculate RoC, Puncta zscore, baseline percentage age, and absolute puncta number change from puncta number
- Stats using mixed design ANOVA
@author: AnyaS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
data = pd.read_excel("20190325_PK2old_MK801_combined.xlsx",sheet_name ='data')
#%% seperate all into conditions
groups=data.groupby('Drug')

#make new dataframe with cyan,green,red with correct fish ID
MK_group = [groups.get_group('MK801')]
MK_group = pd.concat(MK_group)

DMSO_group = [groups.get_group('DMSO')]
DMSO_group = pd.concat(DMSO_group)

#%%
"""
Do MK801 first 
"""
#prep data to have only change, fish id, and time
MK_dataprep = MK_group.drop(['Drug'],1)

#set index for fish)ID -- to wide
MK_dataprep=MK_dataprep.set_index('Fish_ID')

MK_dataprep = MK_dataprep.T
#%%#%%
#do puncta diff from previous row
MK_pdiff= MK_dataprep.diff()
# do RoC
MK_roc = MK_dataprep.pct_change()*100
# do percentage change from baseline
MK_initial = np.array([MK_dataprep.iloc[0,:]]*3) #create frame with inital datapoint (*3 because 3 timepoints)
MK_perch = np.array(((MK_dataprep-MK_initial)/MK_initial)*100)
MK_perch = pd.DataFrame(MK_perch) #perch = Percentage Change

#do z score
from scipy.stats import zscore
MK_puncta_zscore = MK_dataprep.apply(zscore, nan_policy='omit')

#%%combine all into LD_dataframe 
#for RoC
#then melt LL percent change baseline
MK_roc=MK_roc.T #transpose back to FishIDon index
MK_roc=MK_roc.reset_index()
#change nan firstTimepoint to zero 
MK_roc['Pre'] = MK_roc['Pre'].fillna(0)
MK_roc=pd.melt(MK_roc,id_vars='Fish_ID',value_vars=['Pre','Post','Post_20h'])
MK_roc=MK_roc.rename(columns={'value':'RoC','variable':'Time'})
#%%
# for pdiff
MK_pdiff=MK_pdiff.T #transpose back to FishIDon index
MK_pdiff=MK_pdiff.reset_index()
MK_pdiff=pd.melt(MK_pdiff,id_vars='Fish_ID',value_vars=['Pre','Post','Post_20h'])
MK_pdiff=MK_pdiff.rename(columns={'value':'Puncta_delta','variable':'Time'})
#%%
# for zscore
MK_puncta_zscore=MK_puncta_zscore.T #transpose back to FishIDon index
MK_puncta_zscore=MK_puncta_zscore.reset_index()
MK_puncta_zscore=pd.melt(MK_puncta_zscore,id_vars='Fish_ID',value_vars=['Pre','Post','Post_20h'])
MK_puncta_zscore=MK_puncta_zscore.rename(columns={'value':'Puncta_zscore','variable':'Time'})
#%%
#for baseline percentage change == perch
#get fish index as was using np.array before lost Index of DF
MK_Fish_ID = MK_dataprep.T
MK_Fish_ID=MK_Fish_ID.reset_index()
MK_perch =MK_perch.T
MK_perch.insert(1,'Fish_ID',MK_Fish_ID.Fish_ID,True)
#then melt LL percent change baseline
MK_perch=pd.melt(MK_perch,id_vars='Fish_ID',value_vars=[0,1,2])
MK_perch['Time'] = MK_perch['variable'].map({0:'Pre',1:'Post',2:'Post_20h'})
MK_perch=MK_perch.rename(columns={'value':'Baseline_perch'})
MK_perch=MK_perch.drop(['variable'],1)

# same format for puncta
MK_puncta = MK_group.drop(['Drug'],1)
MK_puncta=pd.melt(MK_puncta,id_vars='Fish_ID',value_vars=['Pre','Post','Post_20h'])
MK_puncta=MK_puncta.rename(columns={'value':'Puncta','variable':'Time'})

#same for condition
MK_cond = np.array(['MK801']*len(MK_puncta.index))
#%% combine LD
MK_puncta.insert(3,'RoC',MK_roc.RoC,True) 
MK_puncta.insert(4,'Puncta_zscore',MK_puncta_zscore.Puncta_zscore,True) 
MK_puncta.insert(5,'Baseline_perch',MK_perch.Baseline_perch,True) 
MK_puncta.insert(6,'Puncta_delta',MK_pdiff.Puncta_delta,True) 
MK_puncta.insert(7,'Condition',MK_cond,True) 

#%%
#%%
"""
Do DMSO first 
"""
#prep data to have only change, fish id, and time
DM_dataprep = DMSO_group.drop(['Drug'],1)

#set index for fish)ID -- to wide
DM_dataprep=DM_dataprep.set_index('Fish_ID')

DM_dataprep = DM_dataprep.T
#%%#%%
#do puncta diff from previous row
DM_pdiff= DM_dataprep.diff()
# do RoC
DM_roc = DM_dataprep.pct_change()*100
# do percentage change from baseline
DM_initial = np.array([DM_dataprep.iloc[0,:]]*3) #create frame with inital datapoint (*3 because 3 timepoints)
DM_perch = np.array(((DM_dataprep-DM_initial)/DM_initial)*100)
DM_perch = pd.DataFrame(DM_perch) #perch = Percentage Change

#do z score
from scipy.stats import zscore
DM_puncta_zscore = DM_dataprep.apply(zscore, nan_policy='omit')

#%%combine all into LD_dataframe 
#for RoC
#then melt LL percent change baseline
DM_roc=DM_roc.T #transpose back to FishIDon index
DM_roc=DM_roc.reset_index()
#change nan firstTimepoint to zero 
DM_roc['Pre'] = DM_roc['Pre'].fillna(0)
DM_roc=pd.melt(DM_roc,id_vars='Fish_ID',value_vars=['Pre','Post','Post_20h'])
DM_roc=DM_roc.rename(columns={'value':'RoC','variable':'Time'})
#%%
# for pdiff
DM_pdiff=DM_pdiff.T #transpose back to FishIDon index
DM_pdiff=DM_pdiff.reset_index()
DM_pdiff=pd.melt(DM_pdiff,id_vars='Fish_ID',value_vars=['Pre','Post','Post_20h'])
DM_pdiff=DM_pdiff.rename(columns={'value':'Puncta_delta','variable':'Time'})
#%%
# for zscore
DM_puncta_zscore=DM_puncta_zscore.T #transpose back to FishIDon index
DM_puncta_zscore=DM_puncta_zscore.reset_index()
DM_puncta_zscore=pd.melt(DM_puncta_zscore,id_vars='Fish_ID',value_vars=['Pre','Post','Post_20h'])
DM_puncta_zscore=DM_puncta_zscore.rename(columns={'value':'Puncta_zscore','variable':'Time'})
#%%
#for baseline percentage change == perch
#get fish index as was using np.array before lost Index of DF
DM_Fish_ID = DM_dataprep.T
DM_Fish_ID=DM_Fish_ID.reset_index()
DM_perch =DM_perch.T
DM_perch.insert(1,'Fish_ID',DM_Fish_ID.Fish_ID,True)
#then melt LL percent change baseline
DM_perch=pd.melt(DM_perch,id_vars='Fish_ID',value_vars=[0,1,2])
DM_perch['Time'] = DM_perch['variable'].map({0:'Pre',1:'Post',2:'Post_20h'})
DM_perch=DM_perch.rename(columns={'value':'Baseline_perch'})
DM_perch=DM_perch.drop(['variable'],1)

# same format for puncta
DM_puncta = DMSO_group.drop(['Drug'],1)
DM_puncta=pd.melt(DM_puncta,id_vars='Fish_ID',value_vars=['Pre','Post','Post_20h'])
DM_puncta=DM_puncta.rename(columns={'value':'Puncta','variable':'Time'})

#same for condition
DM_cond = np.array(['DMSO']*len(DM_puncta.index))
#%% combine LD
DM_puncta.insert(3,'RoC',DM_roc.RoC,True) 
DM_puncta.insert(4,'Puncta_zscore',DM_puncta_zscore.Puncta_zscore,True) 
DM_puncta.insert(5,'Baseline_perch',DM_perch.Baseline_perch,True) 
DM_puncta.insert(6,'Puncta_delta',DM_pdiff.Puncta_delta,True) 
DM_puncta.insert(7,'Condition',DM_cond,True) 
#%%
"""
make one big combined DF
"""
data_combined_MK801 = MK_puncta.append(DM_puncta)
#%%
# data_combined_MK801.to_excel("20210319_MK801_df.xlsx")
#%% stats
import pingouin as pg
data_nopre= data[data.Time !='Pre'] # for parameters that have nan/0 for pre time point
#summary table
table = data_nopre.groupby(['Time','Condition'])['Puncta'].agg(['mean','std']).round(2)
print(table)
#Anova mixed
aov= pg.mixed_anova(data=data,dv='Puncta',within='Time',between='Condition',subject='Fish_ID')
pg.print_table(aov)
#pairwise ttests
posthocs = pg.pairwise_ttests(dv='Puncta', within='Time', between='Condition',
                              subject='Fish_ID', data=data)
pg.print_table(posthocs)
