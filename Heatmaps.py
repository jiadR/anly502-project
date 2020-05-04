#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 00:32:31 2020

@author: Charlotte
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import imread
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize

path0 = 'kill_match_stats_final_0.csv'
path1 = 'kill_match_stats_final_1.csv'
path2 = 'kill_match_stats_final_2.csv'
path3 = 'kill_match_stats_final_3.csv'
path4 = 'kill_match_stats_final_4.csv'

kill0 = pd.read_csv(path0, header=0)
kill1 = pd.read_csv(path1, header=0)
kill2 = pd.read_csv(path2, header=0)
kill3 = pd.read_csv(path3, header=0)
kill4 = pd.read_csv(path4, header=0)

kill = pd.concat([kill0,kill1,kill2,kill3,kill4],ignore_index=True)

kill = kill.drop('killer_name',axis=1)
kill = kill.drop('match_id',axis=1)
kill = kill.drop('victim_name',axis=1)

kill.head(10)

edf = kill.loc[kill['map'] == 'ERANGEL']

victim_x_df = edf.filter(regex = 'victim_position_x')
victim_x_s = victim_x_df.values.ravel('F')
print(victim_x_s)

def killer_victim_df_maker(kill_0):
    #choose the position data 
    df = edf
    victim_x_df = df.filter(regex = 'victim_position_x')
    victim_y_df = df.filter(regex = 'victim_position_y')
    killer_x_df = df.filter(regex = 'killer_position_x')
    killer_y_df = df.filter(regex = 'killer_position_y')
    #ravel()the matrix
    victim_x_s = pd.Series(victim_x_df.values.ravel('F'))
    victim_y_s = pd.Series(victim_y_df.values.ravel('F'))
    killer_x_s = pd.Series(killer_x_df.values.ravel('F'))
    killer_y_s = pd.Series(killer_y_df.values.ravel('F'))
    
    vdata = {'x':victim_x_s, 'y':victim_y_s}
    kdata = {'x':killer_x_s, 'y':killer_y_s}
    
    victim_df = pd.DataFrame(data = vdata).dropna(how = 'any')
    victim_df = victim_df[victim_df['x'] > 0]
    killer_df = pd.DataFrame(data = kdata).dropna(how = 'any')
    killer_df = killer_df[killer_df['x'] > 0]
    return killer_df, victim_df

ekdf,evdf = killer_victim_df_maker(edf)

print(ekdf.head(10))
print(evdf.head(10))
print(len(ekdf), len(evdf))

plot_data_ev = evdf[['x','y']].values
plot_data_ek = ekdf[['x','y']].values

# transcribe location data to image size
plot_data_ev = plot_data_ev * 4040 /800000
plot_data_ek = plot_data_ek * 4040 /800000
# Check the position dataset
print(plot_data_ev)
print(plot_data_ek)

def heatmap(x, y, s, bins = 100):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins = bins)
    heatmap = gaussian_filter(heatmap, sigma = s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

bg = imread('/Users/Charlotte/Desktop/ANLY502/pubg-match-deaths/erangel.jpg')
hmap, extent = heatmap(plot_data_ev[:,0], plot_data_ev[:,1], 1.5, bins =800)
alphas = np.clip(Normalize(0, hmap.max()/100, clip=True)(hmap)*1.5,0.0,1.)
colors = Normalize(hmap.max()/100, hmap.max()/20, clip=True)(hmap)
colors = cm.bwr(colors)
colors[..., -1] = alphas

hmap2, extent2 = heatmap(plot_data_ek[:,0],plot_data_ek[:,1],1.5, bins = 800)
alphas2 = np.clip(Normalize(0, hmap2.max()/100, clip = True)(hmap2)*1.5, 0.0, 1.)
colors2 = Normalize(hmap2.max()/100, hmap2.max()/20, clip=True)(hmap2)
colors2 = cm.RdBu(colors2)
colors2[...,-1] = alphas2

#erangel Deathrate
fig, ax = plt.subplots(figsize = (12,12))
ax.set_xlim(0, 4096);ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent = extent, origin = 'lower', cmap = cm.bwr, alpha = 1)
plt.gca().invert_yaxis()
plt.title('Deathrate in Erangel')

#Erangel Killrate
fig, ax = plt.subplots(figsize = (12,12))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors2, extent = extent2, origin = 'lower', cmap = cm.RdBu, alpha = 1)
plt.gca().invert_yaxis()
#plt.colorbar()
plt.title('Killrate in Erangel')

def divbutnotbyzero(a, b):
    c = np.zeros(a.shape)
    for i, row in enumerate(b):
        for j, el in enumerate(row):
            if el == 0:
                c[i][j] = a[i][j]
            else: # got the kill/death ratio
                c[i][j] = a[i][j]/el
    return c

hmap, extent = heatmap(plot_data_ev[:,0], plot_data_ev[:,1], 0, bins = 800)
hmap2, extent2 = heatmap(plot_data_ek[:,0], plot_data_ek[:,1], 0, bins = 800)
hmap3 = divbutnotbyzero(hmap, hmap2)
alphas = np.clip(Normalize(0, hmap3.max()/100, clip=True)(hmap)*1.5, 0.0,1.)
colors = Normalize(hmap3.max()/100, hmap3.max()/20, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[...,-1] = alphas

fig, ax = plt.subplots(figsize = (12, 12))
ax.set_xlim(0,4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent = extent, origin = 'lower', cmap = cm.rainbow, alpha = 0.5)
plt.gca().invert_yaxis()
plt.title('Kill/Death ratio in Erangel')
