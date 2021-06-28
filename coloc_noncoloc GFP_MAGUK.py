# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:43:20 2019
sort out coloc from non-coloc GFP MAGUK puncta 
@author: AnyaS
"""


import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

##make list of 0 1 for coloc= good
goodlist = []
#%%
for n in range(0,11):
    sheet = n
    x=n+1
    dt = pd.read_excel("MAX_20190903_PK2GCamP7B_MAGUK_2h_fix.lif - Series022.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=218; minMGUK = 6; maxMGUK = 255;y=0
    # dt = pd.read_excel("MAX_20190903_PK2GCamP7B_MAGUK_2h_fix.lif - Series018_g_S5-26.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=203; minMGUK = 25; maxMGUK = 219;y=18
    # dt = pd.read_excel("MAX_20190903_PK2GCamP7B_MAGUK_2h_fix.lif - Series017_g_S11-38.xlsx",sheet_name=sheet); minGFP=8 ; maxGFP=217; minMGUK = 38; maxMGUK = 164;y=18+25
    # dt = pd.read_excel("MAX_20190903_PK2GCamP7B_MAGUK_2h_fix.lif - Series015_c_s27-73.xlsx",sheet_name=sheet); minGFP=8 ; maxGFP=178; minMGUK = 25; maxMGUK = 235;y=18+25+13
    # dt = pd.read_excel("MAX_20190903_PK2GCamP7B_MAGUK_2h_fix.lif - Series01_c9_S29-55.xlsx",sheet_name=sheet); minGFP=8 ; maxGFP=177; minMGUK = 29; maxMGUK = 219;y=18+25+13+4
#    
    # dt = pd.read_excel("MAX_20190903_PK2GCamP7B_MAGUK_2h_fix.lif - Series01_c9_S11-30.xlsx",sheet_name=sheet); minGFP=8 ; maxGFP=204; minMGUK = 31; maxMGUK = 248;y=18+25+13+4+6
    # dt = pd.read_excel("MAX_20190903_PK2GCamP7B_MAGUK_2h_fix.lif - F4_3_g_S4-21.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=255; minMGUK = 20; maxMGUK = 255;y=18+25+13+4+6+11
    # dt = pd.read_excel("MAX_20190903_PK2GCamP7B_MAGUK_2h_fix.lif - F4_2_g_S4-18.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=217; minMGUK = 15; maxMGUK = 255;y=18+25+13+4+6+11+37
    # dt = pd.read_excel("MAX_20190903_PK2GCamP7B_MAGUK_2h_fix.lif - F1_6g_S18-38.xlsx",sheet_name=sheet); minGFP=6 ; maxGFP=181; minMGUK = 31; maxMGUK = 255;y=18+25+13+4+6+11+37+22
    # dt = pd.read_excel("MAX_20190903_PK2GCamP7B_MAGUK_2h_fix.lif - F1_5g_S22-39.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=194; minMGUK = 18; maxMGUK = 255;y=18+25+13+4+6+11+37+22+8
#
    # dt = pd.read_excel("MAX_20190903_PK2GCamP7B_MAGUK_2h_fix.lif - F1_3_g_S16-33.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=255; minMGUK = 21; maxMGUK = 255;y=18+25+13+4+6+11+37+22+8+6
    # dt = pd.read_excel("MAX_20190903_PK2GCamP7B_MAGUK_2h_fix.lif - 4_1_g_S4-24.xlsx",sheet_name=sheet); minGFP=0 ; maxGFP=248; minMGUK = 14; maxMGUK = 240;y=18+25+13+4+6+11+37+22+8+6+6


    print("sheet=",x)
    print(dt)

    #transform into array
    distance = np.array(dt['distance'])
    GFP = np.array(dt['GFP'])
    MGUK = np.array(dt['MGUK'])
    #max min value of the two channels from Histogram on ImageJ
    GFPn=(GFP-minGFP)/(maxGFP-minGFP)
    MGUKn=(MGUK-minMGUK)/(maxMGUK-minMGUK)
    Distance = np.around(distance,2)
    print(GFPn, MGUKn)
    
    
    #  find max and min for MAGUK
    maxMGUKn=max(MGUKn)
    minMGUKn=min(MGUKn)
    
    #see if there is a peak in maguk channel 
    MGUKper = minMGUKn/maxMGUKn
    if MGUKper < 0.65:  #max has to be about 1.5(50%) times brighter than min to constitute as peak -- by eye
        print("sheet", x , "good")
        goodlist.append(True)
    else:
        print("sheet", x , "bad")
        goodlist.append(False)
#%%    
goodarray= np.array(goodlist)
print(goodarray)

#%%
#plt pie chart
total = len(goodarray)
positive=sum(goodarray)
percentgood = (positive/total)*100
percentbad = ((total-positive)/total)*100


 # Data to plot
labels = 'Coloc', 'Non-coloc'
size = [percentgood,percentbad]
colors = ['skyblue', 'lightcoral']
explode = (0.1, 0)  # explode 1st slice
##
# 
# Plot
fig,ax2 = plt.subplots(1,1)
ax2.pie(size, explode=explode, labels=labels, colors=colors,
        autopct='%.2f%%', shadow=False, startangle=140,textprops={'fontsize': 14})
ax2.axis('equal')
ax2.text(1.5, -1.2, 'n = 168', ha='right', va='bottom', fontsize=12)
plt.show()
#fig.savefig('20190911_PK2GCAMP7b_goodbad.jpg', format='jpg', dpi=1200)
##
##





#stacked bar
fig,ax = plt.subplots(1,1,figsize=(5,10))
# Data
r = [1]

# plot
barWidth = 0.85
# Create green Bars
ax.bar(r, percentgood, color='skyblue', edgecolor='white', width=barWidth)
# Create orange Bars
ax.bar(r, percentbad, bottom=percentgood, color='lightcoral', edgecolor='white', width=barWidth)
ax.text(2.5, 0.5, 'n = 450', ha='right', va='bottom', fontsize=8)
ax.text(1.5, 50, '90.44% Colocalized', fontsize=8)
ax.text(1.5, 95, '9.55% Non-Colocalized', fontsize=8)

plt.axis('off')
# Show graphic
plt.show()

# fig.savefig('20190512_Goodbad_barstack.jpg', format='jpg', dpi=1200)
