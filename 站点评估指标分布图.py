# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:25:00 2023

@author: YLM xs-

E-mail:202183820003@nuist.edu.cn

    杨柳满长堤，花明路不迷。
        画船人未起，侧枕听莺啼。
"""
# import xarray as xr
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from map_funs import add_Chinese_provinces, set_map_extent_and_ticks
# from matplotlib.lines import Line2D
# from matplotlib.cm import ScalarMappable
# import matplotlib.colors as mcolors
# # 读取全球地形数据
# ds = xr.open_dataset(r"F:\ETOPO2v2c_f4.nc")
# # 准备用于绘图的数据
# lon = np.linspace(min(ds['x'].data), max(ds['x'].data), len(ds['x'].data))  # 经度
# lat = np.linspace(min(ds['y'].data), max(ds['y'].data), len(ds['y'].data))  # 纬度
# lon, lat = np.meshgrid(lon, lat)  # 构建经纬网
# dem = ds['z'].data  # DEM数据
# # 绘制地图
# levels = [-8000, -6000, -4000, -2000, -1000, -200, -50, 0, 50, 200, 500, 1000, 1500, 2000, 3000, 4000, 
#           5000, 6000, 7000, 8000]  # 创建分级
# color = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#006837', 
#          '#31a354', '#78c679', '#addd8e', '#d9f0a3', '#f7fcb9', '#c9bc87', '#a69165', '#856b49', 
#          '#664830', '#ad9591', '#d7ccca']  # 设置色带

# # 读取数据文件
# data = pd.read_excel("F:\F TaiFong\影响中国的热带气旋降水预测研究：基于机器学习的探索\气象态\站点评估数据.xlsx")

# # 设置绘图区域的经纬度范围
# lonmin, lonmax = 90, 135
# latmin, latmax = 15, 50
# extents = [lonmin, lonmax, latmin, latmax]
# proj = ccrs.PlateCarree()

# # 创建地图
# fig = plt.figure(figsize=(10, 15))
# ax = fig.add_subplot(111, projection=proj)

# # 添加地理特征，如海岸线、省界等
# ax.coastlines(resolution='10m', lw=0.3)
# ax.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.3)
# ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)
# ax.add_feature(cfeature.OCEAN, edgecolor='blue', linewidth=0.3)
# add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
# ax.contourf(lon, lat, dem, levels=levels, extend='both', colors=color, vmin=0, vmax=7000)
# # 设置经纬度刻度
# set_map_extent_and_ticks(ax, extents, xticks=np.arange(lonmin, lonmax, 5), yticks=np.arange(latmin, latmax, 5), nx=0.1, ny=0.1)
# ax.tick_params(labelsize='small')
# lat = data['Lat']
# lon = data['Long']
# ax.plot(lon, lat, 'o', markerfacecolor='white', markeredgecolor='black', markersize=6, transform=proj, label='Stations')
# ax.legend(loc='lower right', fontsize=12)
# ax.set_title("Stations distribution map", fontsize=16)
# plt.savefig("F:\F TaiFong\PAPER\paper fig\观测站点分布图.png", dpi=1080, facecolor='white')  # 保存图片
# # 显示图形
# plt.show()
a,b=7,6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from map_funs import add_Chinese_provinces, set_map_extent_and_ticks
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

# 读取数据文件
data = pd.read_excel("F:\F TaiFong\影响中国的热带气旋降水预测研究：基于机器学习的探索\气象态\站点评估数据.xlsx")

# 设置绘图区域的经纬度范围
lonmin, lonmax = 90, 135
latmin, latmax = 15, 50
extents = [lonmin, lonmax, latmin, latmax]
proj = ccrs.PlateCarree()

# 创建地图
fig = plt.figure(figsize=(a, b))
ax = fig.add_subplot(111, projection=proj)

# 添加地理特征，如海岸线、省界等
ax.coastlines(resolution='10m', lw=0.3)
ax.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.3)
ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)
ax.add_feature(cfeature.OCEAN, edgecolor='blue', linewidth=0.3)
add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')

# 设置经纬度刻度
set_map_extent_and_ticks(ax, extents, xticks=np.arange(lonmin, lonmax, 5), yticks=np.arange(latmin, latmax, 5), nx=0.1, ny=0.1)
ax.tick_params(labelsize='small')

# 绘制站点
cmap = plt.get_cmap('Reds')  # 使用Reds颜色映射
R_min = 0
R_max = 1
norm = mcolors.Normalize(vmin=0, vmax=R_max)  # 根据R值范围进行归一化
for index, row in data.iterrows():
    lat = row['Lat']
    lon = row['Long']
    r = row['R']

    # 设置站点颜色和大小
    if pd.isna(r) or r < 0:
        ax.plot(lon, lat, 'v', color='black', markersize=5, transform=proj)
    else:
        ax.plot(lon, lat, 'o', color=cmap(norm(r)), markersize=5, transform=proj)

# 添加色条图例
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 设置虚拟数组
cbar = plt.colorbar(sm, ax=ax, location='bottom', pad=0.05, fraction=0.04)
cbar.set_label('R', fontsize=10)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(np.linspace(0, R_max, num=9))  # 设置刻度范围，例如从 0 到 R_max，分成 9 个刻度
cbar.set_ticklabels([f'{i:.2f}' for i in np.linspace(0, R_max, num=9)])  # 设置刻度标签
ax.plot(90, 10, 'v', color='black', markersize=10, transform=proj,label='NaN Station')
ax.legend(loc='lower right', fontsize=8)
# # # 图右下角添加图例说明
# handles = [Line2D([0], [0], marker='v', color='black', markerfacecolor='black', label='NaN Data Station', markersize=10)]
# ax.legend(handles=handles, loc='lower right', fontsize=8)
#ax.set_title("Correlation Coefficients for Stations", fontsize=12)
plt.title('(c)', fontsize=18, loc='left')
plt.savefig("F:\F TaiFong\PAPER\Submissions Needing Revision QAQ\Figure\Figure 4c.jpg", dpi=480, facecolor='white', bbox_inches='tight')  # 保存图片
# 显示图形
plt.show()






# #########绝对误差



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from map_funs import add_Chinese_provinces, set_map_extent_and_ticks
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

# 读取数据文件
data = pd.read_excel("F:\F TaiFong\影响中国的热带气旋降水预测研究：基于机器学习的探索\气象态\站点评估数据.xlsx")

# 设置绘图区域的经纬度范围
lonmin, lonmax = 90, 135
latmin, latmax = 15, 50
extents = [lonmin, lonmax, latmin, latmax]
proj = ccrs.PlateCarree()

# 创建地图
fig = plt.figure(figsize=(a, b))
ax = fig.add_subplot(111, projection=proj)

# 添加地理特征，如海岸线、省界等
ax.coastlines(resolution='10m', lw=0.3)
ax.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.3)
ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)
ax.add_feature(cfeature.OCEAN, edgecolor='blue', linewidth=0.3)
add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')

# 设置经纬度刻度
set_map_extent_and_ticks(ax, extents, xticks=np.arange(lonmin, lonmax, 5), yticks=np.arange(latmin, latmax, 5), nx=0.1, ny=0.1)
ax.tick_params(labelsize='small')

# 绘制站点
cmap = plt.get_cmap('jet')  # 使用jet颜色映射
norm = mcolors.Normalize(vmin=data['Bias_Abs'].min(), vmax=data['Bias_Abs'].max())  # 根据Bias_Abs值范围进行归一化
for index, row in data.iterrows():
    lat = row['Lat']
    lon = row['Long']
    bias_abs = row['Bias_Abs']

    # 设置站点颜色和大小
    if pd.isna(bias_abs):
        ax.plot(lon, lat, 'v', color='black', markersize=10, transform=proj)
    else:
        ax.plot(lon, lat, 'o', color=cmap(norm(bias_abs)), markersize=5, transform=proj)
# 添加色条图例
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 设置虚拟数组
cbar = plt.colorbar(sm, ax=ax, location='bottom', pad=0.05, fraction=0.04)
cbar.set_label('MAE(mm)', fontsize=10)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(np.linspace(data['Bias_Abs'].min(), data['Bias_Abs'].max(), num=9))  # 设置刻度范围
cbar.set_ticklabels(['{:.0f}'.format(val) for val in np.linspace(data['Bias_Abs'].min(), data['Bias_Abs'].max(), num=9)])  # 设置刻度标签
ax.plot(90, 10, 'v', color='black', markersize=5, transform=proj,label='NaN Station')
ax.legend(loc='lower right', fontsize=8)
plt.title('(a)', fontsize=18, loc='left')
# # 图下方说明范围
# cbar.ax.text(0.5, -1.2, 'Value Range: {:.3f}mm to {:.3f}mm'.format(data['Bias_Abs'].min(), data['Bias_Abs'].max()),
#               fontsize=8, ha='center')
#ax.set_title("Absolute Bias for Stations", fontsize=12)
plt.savefig("F:\F TaiFong\PAPER\Submissions Needing Revision QAQ\Figure\Figure 4a.jpg", dpi=480, facecolor='white', bbox_inches='tight')  # 保存图片
# 显示图形
plt.show()

# #####绝对百分比误差


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from map_funs import add_Chinese_provinces, set_map_extent_and_ticks
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

# 读取数据文件
data = pd.read_excel("F:\F TaiFong\影响中国的热带气旋降水预测研究：基于机器学习的探索\气象态\站点评估数据.xlsx")

# 设置绘图区域的经纬度范围
lonmin, lonmax = 90, 135
latmin, latmax = 15, 50
extents = [lonmin, lonmax, latmin, latmax]
proj = ccrs.PlateCarree()

# 创建地图
fig = plt.figure(figsize=(a, b))
ax = fig.add_subplot(111, projection=proj)

# 添加地理特征，如海岸线、省界等
ax.coastlines(resolution='10m', lw=0.3)
ax.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.3)
ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)
ax.add_feature(cfeature.OCEAN, edgecolor='blue', linewidth=0.3)
add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')

# 设置经纬度刻度
set_map_extent_and_ticks(ax, extents, xticks=np.arange(lonmin, lonmax, 5), yticks=np.arange(latmin, latmax, 5), nx=0.1, ny=0.1)
ax.tick_params(labelsize='small')

# 绘制站点
cmap = plt.get_cmap('jet')  # 使用jet颜色映射
norm = mcolors.Normalize(vmin=0, vmax=200)  # 将范围限制在 0 到 200

for index, row in data.iterrows():
    lat = row['Lat']
    lon = row['Long']
    pbias_abs = row['Pbias_Abs']

    # 设置站点颜色和大小
    if pd.isna(pbias_abs):
        ax.plot(lon, lat, 'v', color='black', markersize=5, transform=proj)
    # elif pbias_abs > 200:
    #     ax.plot(lon, lat, 'o', color='black', markersize=5, transform=proj)
    else:
        ax.plot(lon, lat, 'o', color=cmap(norm(pbias_abs)), markersize=5, transform=proj)

# 添加色条图例
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 设置虚拟数组
cbar = plt.colorbar(sm, ax=ax, location='bottom', pad=0.05, fraction=0.04)
cbar.set_label('MAPE', fontsize=10)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks(np.linspace(0, 200, num=9))  # 设置刻度范围，例如从 0 到 200，分成 9 个刻度
cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%', '125%', '150%', '175%', '200%'])  # 设置刻度标签
ax.plot(90, 10, 'v', color='black', markersize=10, transform=proj,label='NaN Station')
ax.legend(loc='lower right', fontsize=8)

# 添加图下方说明范围
#plt.text(0.5, 0.08, 'Error Range:0-200 % (Error>200% are black)', fontsize=10, transform=fig.transFigure,
#          horizontalalignment='center')
#ax.set_title("Absolute Bias for Stations", fontsize=12)
# 保存图片
plt.title('(b)', fontsize=18, loc='left')
plt.savefig("F:\F TaiFong\PAPER\Submissions Needing Revision QAQ\Figure\Figure 4b.jpg", dpi=480, facecolor='white', bbox_inches='tight')
# 显示图形
plt.show()



