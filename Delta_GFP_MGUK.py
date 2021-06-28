# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:55:51 2019
Normalized each stack image to min/max values of GFP/MAGUK
make combine list of delta intensity per puncta for both channels
sort coloc from noncoloc 
@author: AnyaS
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
###FIND DIFF IN DISTANCE   
##makelist to append
Delta_GFP = []
Delta_MGUK = []

for n in range(0, 55): #start, stop(not including this no) 
    sheet = n
    x=n+1
    dt = pd.read_excel("20190115_PK_mnx_MGUK_2dpf_acetone.lif - F8_slice_1-22.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=255; minMGUK = 3; maxMGUK = 245
#    dt = pd.read_excel("20190115_PK_mnx_MGUK_2dpf_acetone.lif - F6_slice8-24.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=255; minMGUK = 8; maxMGUK = 121;y=17
#    dt = pd.read_excel("20190115_PK_mnx_F72.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=255; minMGUK = 6; maxMGUK = 100;y=17+33
#    dt = pd.read_excel("20190115_PK_mnx_F9_slice5-21.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=255; minMGUK = 2; maxMGUK = 106;y=17+33+69
#    dt = pd.read_excel("20190115_PK_mnx_F6_2_slice1-15.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=255; minMGUK = 4; maxMGUK = 226;y=17+33+69+71
#    dt = pd.read_excel("20190115_PK_mnx _4D_F5_2_slice6-22.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=255; minMGUK = 0; maxMGUK = 255;y=17+33+69+71+71

#    dt = pd.read_excel("MAX_20190125_PK2mnx_MGUK_2dpf_1.5h_acetone.lif - 1h_F1_slice6-19.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=255; minMGUK = 15; maxMGUK = 208;y=17+33+69+71+71+84
#    dt = pd.read_excel("MAX_20190125_PK2mnx_MGUK_2dpf_1.5h_acetone.lif - 1.5h_F3-3_slice1-14.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=255; minMGUK = 0; maxMGUK = 193;y=17+33+69+71+71+84+23
#    dt = pd.read_excel("MAX_20190125_PK2mnx_MGUK_2dpf_1.5h_acetone.lif - 1.5h_F3-2_slice5-14.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=255; minMGUK = 0; maxMGUK = 199;y=17+33+69+71+71+84+23+40
    # dt = pd.read_excel("MAX_20190125_PK2mnx_MGUK_2dpf_1.5h_acetone.lif - 1.5h_F3_slice2-14.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=255; minMGUK = 0; maxMGUK = 242;y=17+33+69+71+71+84+23+40+77

    #transform into array
    distance = np.array(dt['distance'])
    GFP = np.array(dt['GFP'])
    MGUK = np.array(dt['MGUK'])
    
    #max min value of the two channels from Histogram on ImageJ
    GFPn=(GFP-minGFP)/(maxGFP-minGFP)
    MGUKn=(MGUK-minMGUK)/(maxMGUK-minMGUK)
    Distance = np.around(distance,2)

    #trying to find max
    maxGFPn=max((GFPn))
    minGFPn=min((GFPn))
    maxMGUKn=max((MGUKn))
    minMGUKn=min((MGUKn))
    
    #find DeltaGFP and Delta GPHN
    dGFP = maxGFPn - minGFPn
    dMGUK = maxMGUKn - minMGUKn
    print('Delta_GFP is', dGFP)
    print('Delta_GPHN is', dMGUK)

    ##adding to the list 
    Delta_GFP.append(dGFP)
    Delta_MGUK.append(dMGUK)
#
#
Delta_GFP = np.array(Delta_GFP)
Delta_MGUK = np.array(Delta_MGUK)
#%%
####sorting these delta into "good" vs "bad" intensity according to Bool good/bad array
badarray = np.invert(array_combined)
##for MGUK
goodMGUK_int = Delta_MGUK[array_combined]
##invert True to False vice versa == making badarray
badMGUK_int = Delta_MGUK[badarray]

##for GFP
goodGFP_int = Delta_GFP[array_combined]
badGFP_int = Delta_GFP[badarray]
#%%
##for MGUK fwhm
new_goodMGUK_width = Width_MAGUK[array_combined]
##invert True to False vice versa == making badarray
new_badMGUK_width = Width_MAGUK[badarray]

##for GFP
new_goodGFP_width = Width_GFP[array_combined]
new_badGFP_width = Width_GFP[badarray]

#%%
##find R^2 value 
from scipy import stats
slope_g, intercept_g, r_value_g, p_value_g, std_err_g = stats.linregress(goodMGUK_int,goodGFP_int)
print("r-squared:", r_value_g**2)
good_r = r_value_g**2
slope, intercept, r_value, p_value, std_err = stats.linregress(badMGUK_int,badGFP_int)
print("r-squared:", r_value**2)
bad_r = r_value**2
#%%
fig, ax = plt.subplots(1,2)
fig.set_size_inches(10,5)
ax = plt.subplot(1,2,1)
ax = sns.regplot(x=goodGFP_int, y=goodMGUK_int, fit_reg = 'True',x_ci='sd')
ax = sns.regplot(x=badGFP_int, y=badMGUK_int, fit_reg='True',color='lightcoral',x_ci='sd')
ax=plt.tick_params(labelsize=13)
sns.despine()
plt.xlim(0.075, 0.9)
plt.ylabel("Delta MAGUK intensity", fontsize=13)
plt.xlabel("Delta GFP intensity", fontsize=13)

ax = plt.subplot(1,2,2)
ax = sns.regplot(x=new_goodGFP_width, y=new_goodMGUK_width, fit_reg = 'True',x_ci='sd')
ax = sns.regplot(x=new_badGFP_width, y=new_badMGUK_width, fit_reg='True',color='lightcoral',x_ci='sd')
ax=plt.tick_params(labelsize=13)
sns.despine()
plt.xlim(0.0, 1.5)
plt.ylabel("FWHM MAGUK", fontsize=13)
plt.xlabel("FWHM GFP", fontsize=13)
#%%
fig.tight_layout()
#%%
# fig.savefig('PSD95_regplot_intensity_FWHM.png',  transparent = True, dpi=1200)

