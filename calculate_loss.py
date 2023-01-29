import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm, colors, colorbar
from matplotlib import gridspec, pyplot as plt
import numpy as np
import random
import pandas as pd
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import importlib


df = pd.read_csv("/data3/wangfx/ai_ref/cloudsat-gan-master/csmodiscgan/data/cloudtype_data5.csv")
df_select = df.loc[df.cloud_type.isin(['no cloud','Cirrus','Altostratus','Altocumulus',
                         'St','Sc','Cumulus','Ns','Deep Convection']), :]

# Plot
sns.set_style("white")
gridobj = sns.lmplot(x="size", y="POD", hue="cloud_type", data=df_select, 
                      height=8, aspect=1.6, robust=False, 
                      palette=["#ff0000","#ff6347","#ffff00","#00ff00","#008000","#00ffff","#0000ff","#9a0eea"],
                      # order=2,
                      hue_order=['Cirrus','Altostratus','Altocumulus',
                          'St','Sc','Cumulus','Ns','Deep Convection'],
                      legend_out=False,
                      scatter_kws=dict(s=30, linewidths=.2, edgecolors='white'))

# ax = sns.jointplot(x="size", y="POD", hue="cloud_type", data=df_select, 
#                      palette=["#ff0000"],
#                      # order=2,
#                       hue_order=['Cirrus'],)
                      

#gridobj.legend(loc='center right', bbox_to_anchor=(0.8,0.1), ncol=1)
#Decorations

# gridobj.set(xlim=(0, 4096), ylim=(0, 1))
# plt.title("Scatterplot with line of best fit grouped by number of cylinders", fontsize=20)
plt.savefig("./ncloudtypes5.jpg")
# plt.close()
# grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
# # Plot
# sns.set_style("white")
# gridobj = sns.lmplot(x="size", y="POD", hue="cloud_type", data=df_select, 
#                      height=8, aspect=1.6, robust=False, 
#                      palette=["#ff6347","#008000","#00ffff","#9a0eea"],
#                      # order=2,
#                       hue_order=['Altostratus'
#                           ,'Sc','Cumulus','Deep Convection'],
#                      legend_out=False,
#                      scatter_kws=dict(s=30, linewidths=.2, edgecolors='white'))

# # gridobj.legend(loc='center right', bbox_to_anchor=(0.8,0.1), ncol=1)
# # Decorations
# ax_right = gridobj.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
# gridobj.set(xlim=(0, 4096), ylim=(0, 1))
# # plt.title("Scatterplot with line of best fit grouped by number of cylinders", fontsize=20)
# plt.savefig("./ncloudtypes51.jpg")
# plt.close()

# # Plot
# sns.set_style("white")
# gridobj = sns.lmplot(x="size", y="POD", hue="cloud_type", data=df_select, 
#                      height=8, aspect=1.6, robust=False, 
#                      palette=["#ff0000","#ffff00","#00ff00","#0000ff"],
#                      # order=2,
#                       hue_order=['Cirrus','Altocumulus','St','Ns'],
#                      legend_out=False,
#                      scatter_kws=dict(s=30, linewidths=.2, edgecolors='white'))

# # gridobj.legend(loc='center right', bbox_to_anchor=(0.8,0.1), ncol=1)
# # Decorations

# gridobj.set(xlim=(0, 4096), ylim=(0, 1))
# # plt.title("Scatterplot with line of best fit grouped by number of cylinders", fontsize=20)
# plt.savefig("./ncloudtypes52.jpg")
# plt.close()




# df = pd.read_csv("/data3/wangfx/ai_ref/cloudsat-gan-master/csmodiscgan/data/cloudtype_data-25.csv")
# df_select = df.loc[df.cloud_type.isin(['no cloud','Cirrus','Altostratus','Altocumulus',
#                          'St','Sc','Cumulus','Ns','Deep Convection']), :]

# # Plot
# sns.set_style("white")
# gridobj = sns.lmplot(x="size", y="POD", hue="cloud_type", data=df_select, 
#                      height=8, aspect=1.6, robust=False, 
#                      palette=["#ff0000","#ff6347","#ffff00","#00ff00","#008000","#00ffff","#0000ff","#9a0eea"],
#                      # order=2,
#                       hue_order=['Cirrus','Altostratus','Altocumulus',
#                           'St','Sc','Cumulus','Ns','Deep Convection'],
#                      legend_out=False,
#                      scatter_kws=dict(s=30, linewidths=.2, edgecolors='white'))

# # gridobj.legend(loc='center right', bbox_to_anchor=(0.8,0.1), ncol=1)
# # Decorations

# gridobj.set(xlim=(0, 4096), ylim=(0, 1))
# # plt.title("Scatterplot with line of best fit grouped by number of cylinders", fontsize=20)
# plt.savefig("./ncloudtypes-25.jpg")
# plt.close()

# # Plot
# sns.set_style("white")
# gridobj = sns.lmplot(x="size", y="POD", hue="cloud_type", data=df_select, 
#                      height=8, aspect=1.6, robust=False, 
#                      palette=["#ff6347","#008000","#00ffff","#9a0eea"],
#                      # order=2,
#                       hue_order=['Altostratus'
#                           ,'Sc','Cumulus','Deep Convection'],
#                      legend_out=False,
#                      scatter_kws=dict(s=30, linewidths=.2, edgecolors='white'))

# # gridobj.legend(loc='center right', bbox_to_anchor=(0.8,0.1), ncol=1)
# # Decorations

# gridobj.set(xlim=(0, 4096), ylim=(0, 1))
# # plt.title("Scatterplot with line of best fit grouped by number of cylinders", fontsize=20)
# plt.savefig("./ncloudtypes-251.jpg")
# plt.close()

# # Plot
# sns.set_style("white")
# gridobj = sns.lmplot(x="size", y="POD", hue="cloud_type", data=df_select, 
#                      height=8, aspect=1.6, robust=False, 
#                      palette=["#ff0000","#ffff00","#00ff00","#0000ff"],
#                      # order=2,
#                       hue_order=['Cirrus','Altocumulus','St','Ns'],
#                      legend_out=False,
#                      scatter_kws=dict(s=30, linewidths=.2, edgecolors='white'))

# # gridobj.legend(loc='center right', bbox_to_anchor=(0.8,0.1), ncol=1)
# # Decorations

# gridobj.set(xlim=(0, 4096), ylim=(0, 1))
# # plt.title("Scatterplot with line of best fit grouped by number of cylinders", fontsize=20)
# plt.savefig("./ncloudtypes-252.jpg")
# plt.close()