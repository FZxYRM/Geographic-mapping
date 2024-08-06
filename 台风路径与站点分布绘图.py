# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:55:02 2023

@author: YLM xs- 202183820003@nuist.cn

暮色苍茫看劲松，乱石飞渡仍从容。
天生一个仙人洞，无限风光在险峰。

"""


# # # TALIM


# import pandas as pd
# import numpy as np
# from matplotlib.collections import LineCollection
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# import math
# import xarray as xr
# file = r"F:\ECMWRF\data\gribTalim\Talim 2304.txt"
# ds = xr.open_dataset(r"F:\ETOPO2v2c_f4.nc")
# # 准备用于绘图的数据
# lon1 = np.linspace(min(ds['x'].data), max(ds['x'].data), len(ds['x'].data))  # 经度
# lat1 = np.linspace(min(ds['y'].data), max(ds['y'].data), len(ds['y'].data))  # 纬度
# lon1, lat1 = np.meshgrid(lon1, lat1)  # 构建经纬网
# dem = ds['z'].data  # DEM数据

# # 绘制地图
# levels = [-8000, -6000, -4000, -2000, -1000, -200, -50, 0, 50, 200, 500, 1000, 1500, 2000, 3000, 4000, 
#           5000, 6000, 7000, 8000]  # 创建分级
# color = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#006837', 
#           '#31a354', '#78c679', '#addd8e', '#d9f0a3', '#f7fcb9', '#c9bc87', '#a69165', '#856b49', 
#           '#664830', '#ad9591', '#d7ccca']  # 设置色带

# # 使用pandas的read_table函数读取txt文件
# data = pd.read_table(file, delim_whitespace=True)
# time1 = data['时间月日'] 
# time2 = data['时间'] 
# presure = data['中心气压'] 
# long=data['中心经度']
# lat=data['中心纬度']
# nws=data['风速']
# level = data['level']
# time = [f"{m}-{h}" for m, h in zip(time1, time2)]
# print(time,len(time))

# # 创建Figure
# fig = plt.figure(figsize=(15, 12))
# # 绘制台风路径
# ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
# # 设置ax1的范围
# a, b, c, d = 105, 118, 15, 26
# ax1.set_extent([a, b, c, d])
# ax1.coastlines(resolution='10m', lw=0.3)
# ax1.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.3)
# ax1.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)
# ax1.add_feature(cfeature.OCEAN, edgecolor='blue', linewidth=0.3)
# ax1.contourf(lon1, lat1, dem, levels=levels, extend='both', colors=color, vmin=0, vmax=7000)
# # 为ax1添加地理经纬度标签及刻度
# ax1.set_xticks(np.arange(a, b+1, 2), crs=ccrs.PlateCarree())
# ax1.set_yticks(np.arange(c, d+1, 2), crs=ccrs.PlateCarree())
# ax1.xaxis.set_major_locator(MultipleLocator(2))
# ax1.yaxis.set_major_locator(MultipleLocator(2))
# ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
# ax1.yaxis.set_major_formatter(FormatStrFormatter("%d"))
# # 绘制台风路径
# lat = lat
# lon = long
# pressure = presure
# level = nws

# s1 = ax1.scatter(lon, lat, c=pressure, s=(20*(level-1)), cmap='Reds_r', vmax=1010, vmin=960, alpha=1)
# ax1.plot(lon, lat, linewidth=6)
# fig.colorbar(s1, ax=ax1, fraction=0.04)
# ax1.set_title("(a)", fontsize=16, loc='left')

# # 绘制台风路径
# ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

# # 设置ax2的范围
# ax2.set_extent([a, b, c, d])

# ax2.coastlines(resolution='10m', lw=0.3)
# ax2.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.3)
# ax2.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)
# ax2.add_feature(cfeature.OCEAN, edgecolor='blue', linewidth=0.3)
# ax2.contourf(lon1, lat1, dem, levels=levels, extend='both', colors=color, vmin=0, vmax=7000)
# # 为ax2添加地理经纬度标签及刻度
# ax2.set_xticks(np.arange(a, b+1, 2), crs=ccrs.PlateCarree())
# ax2.set_yticks(np.arange(c, d+1, 2), crs=ccrs.PlateCarree())
# ax2.xaxis.set_major_locator(MultipleLocator(2))
# ax2.yaxis.set_major_locator(MultipleLocator(2))
# ax2.xaxis.set_major_formatter(FormatStrFormatter("%d"))
# ax2.yaxis.set_major_formatter(FormatStrFormatter("%d"))

# # 将经纬度数据点存入同一数组
# points = np.array([lon, lat]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)

# # 设置色标的标准化范围(即将Z维度的数据对应为颜色数组)
# norm = plt.Normalize(10, 40)
# # 设置颜色线条
# linewidths = 6
# # 设置颜色线条
# lc = LineCollection(segments, cmap='jet', norm=norm, linewidths=linewidths, transform=ccrs.PlateCarree())
# lc.set_array(nws[:-1])
# # # 绘制线条
# line = ax2.add_collection(lc)
# fig.colorbar(lc, ax=ax2, fraction=0.04)
# ax2.set_title("(b)", fontsize=16, loc='left') #Station:'SHANTOU,GUANGZHOU,HAIKOU'" Typhoon Talim
# # 读取站点数据
# station_data = pd.read_excel(r"F:\F TaiFong\PAPER\paper fig\TL\TL站点.xlsx")

# # 绘制站点，外轮廓为粗黑轮廓，填充颜色为白
# ax1.scatter(station_data["Long_sta"], station_data["Lat_sta"], edgecolors='black', facecolors='white', marker='^', label='Stations', s=50, linewidths=2, zorder=3)
# ax2.scatter(station_data["Long_sta"], station_data["Lat_sta"], edgecolors='black', facecolors='white', marker='^', label='Stations', s=50, linewidths=2, zorder=3)
# # 在站点旁边写字
# # for i in range(len(station_data)):
# #     ax1.text(station_data["Long_sta"].iloc[i], station_data["Lat_sta"].iloc[i], station_data["ID1"].iloc[i], fontsize=11, ha='right', va='center', color='black')
# #     ax2.text(station_data["Long_sta"].iloc[i], station_data["Lat_sta"].iloc[i], station_data["ID1"].iloc[i], fontsize=11, ha='right', va='center', color='black')
# # 在站点位置左边的100个像素点写字，字母被一个圆圈包围，字用蓝色
# for i in range(len(station_data)):
#     x = station_data["Long_sta"].iloc[i] - 0.35
#     y = station_data["Lat_sta"].iloc[i]

#     # # 绘制圆圈
#     # circle1 = plt.Circle((x, y), radius=0.16, color='blue', fill=False, zorder=4)
#     # ax1.add_patch(circle1)
#     # circle2 = plt.Circle((x, y), radius=0.16, color='blue', fill=False, zorder=4)
#     # ax2.add_patch(circle2)
#     ax1.text(x, y, station_data["ID1"].iloc[i], fontsize=11, ha='center', va='center', color='white', zorder=5)
#     ax2.text(x, y, station_data["ID1"].iloc[i], fontsize=11, ha='center', va='center', color='white', zorder=5)
# # 指定要在哪些点写入文字的索引列表
# specified_indices = [0,10,22,35,46]  # 请根据实际需要修改这个列表
# # 在每个点上标记时间
# for i in range(len(lon)):
#     if i in specified_indices:
#         ax1.text(lon[i], lat[i], time[i], fontsize=11, rotation=45, ha='left', va='bottom', color='black', alpha=0.6, fontweight='bold')
#         ax2.text(lon[i], lat[i], time[i], fontsize=11, rotation=45, ha='left', va='bottom', color='black', alpha=0.6, fontweight='bold')
# #plt.suptitle("Typhoon Doksuri ID:2305", fontsize=16, y=0.73)
# plt.legend(loc=4, prop={'size': 10})
# plt.savefig("F:\F TaiFong\PAPER\Submissions Needing Revision QAQ\Figure 6ab.jpg",dpi=480, facecolor='white', bbox_inches='tight')

# plt.show()









from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

file = r"F:\ECMWRF\data\gribDSR\Doksuri 2305 路径.txt"
ds = xr.open_dataset(r"F:\ETOPO2v2c_f4.nc")

# 准备用于绘图的数据
lon1 = np.linspace(min(ds['x'].data), max(ds['x'].data), len(ds['x'].data))  # 经度
lat1 = np.linspace(min(ds['y'].data), max(ds['y'].data), len(ds['y'].data))  # 纬度
lon1, lat1 = np.meshgrid(lon1, lat1)  # 构建经纬网
dem = ds['z'].data  # DEM数据

# 绘制地图
levels = [-8000, -6000, -4000, -2000, -1000, -200, -50, 0, 50, 200, 500, 1000, 1500, 2000, 3000, 4000,
          5000, 6000, 7000, 8000]  # 创建分级
color = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#006837',
          '#31a354', '#78c679', '#addd8e', '#d9f0a3', '#f7fcb9', '#c9bc87', '#a69165', '#856b49',
          '#664830', '#ad9591', '#d7ccca']  # 设置色带

# 使用pandas的read_table函数读取txt文件
data = pd.read_table(file, delim_whitespace=True)
time1 = data['时间月日']
time2 = data['时间']
presure = data['中心气压']
long = data['中心经度']
lat = data['中心纬度']
nws = data['风速']
level = data['level']
time = [f"{m}-{h}" for m, h in zip(time1, time2)]

# 创建Figure
fig = plt.figure(figsize=(15, 12))

# 绘制台风路径
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
a, b, c, d = 110, 125, 20, 34
ax1.set_extent([a, b, c, d])
ax1.coastlines(resolution='10m', lw=0.3)
ax1.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.3)
ax1.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)
ax1.add_feature(cfeature.OCEAN, edgecolor='blue', linewidth=0.3)
ax1.contourf(lon1, lat1, dem, levels=levels, extend='both', colors=color, vmin=0, vmax=7000)
ax1.set_xticks(np.arange(a, b+1, 2), crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(c, d+1, 2), crs=ccrs.PlateCarree())
ax1.xaxis.set_major_locator(MultipleLocator(2))
ax1.yaxis.set_major_locator(MultipleLocator(2))
ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax1.yaxis.set_major_formatter(FormatStrFormatter("%d"))

# 绘制台风路径
lat = lat
lon = long
pressure = presure
level = nws

s1 = ax1.scatter(lon, lat, c=pressure, s=(20*(level-1)), cmap='Reds_r', vmax=1010, vmin=920, alpha=1)
ax1.plot(lon, lat, linewidth=6)
fig.colorbar(s1, ax=ax1, fraction=0.04)
ax1.set_title("(a)", fontsize=16, loc='left')

# 绘制台风路径
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
ax2.set_extent([a, b, c, d])
ax2.coastlines(resolution='10m', lw=0.3)
ax2.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.3)
ax2.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)
ax2.add_feature(cfeature.OCEAN, edgecolor='blue', linewidth=0.3)
ax2.contourf(lon1, lat1, dem, levels=levels, extend='both', colors=color, vmin=0, vmax=7000)
ax2.set_xticks(np.arange(a, b+1, 2), crs=ccrs.PlateCarree())
ax2.set_yticks(np.arange(c, d+1, 2), crs=ccrs.PlateCarree())
ax2.xaxis.set_major_locator(MultipleLocator(2))
ax2.yaxis.set_major_locator(MultipleLocator(2))
ax2.xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax2.yaxis.set_major_formatter(FormatStrFormatter("%d"))

# 将经纬度数据点存入同一数组
points = np.array([lon, lat]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 设置色标的标准化范围(即将Z维度的数据对应为颜色数组)
norm = plt.Normalize(10, 60)
# 设置颜色线条
linewidths = 6
# 设置颜色线条
lc = LineCollection(segments, cmap='jet', norm=norm, linewidths=linewidths, transform=ccrs.PlateCarree())
lc.set_array(nws[:-1])
line = ax2.add_collection(lc)
fig.colorbar(lc, ax=ax2, fraction=0.04)
ax2.set_title("(b)", fontsize=16, loc='left')

# 读取站点数据
station_data = pd.read_excel(r"F:\F TaiFong\PAPER\paper fig\DSR\DSR站点.xlsx")

# 绘制站点，外轮廓为粗黑轮廓，填充颜色为白
ax1.scatter(station_data["Long_sta"], station_data["Lat_sta"], edgecolors='black', facecolors='white', marker='^', label='Stations', s=50, linewidths=2, zorder=3)
ax2.scatter(station_data["Long_sta"], station_data["Lat_sta"], edgecolors='black', facecolors='white', marker='^', label='Stations', s=50, linewidths=2, zorder=3)
# 在站点旁边写字
# for i in range(len(station_data)):
#     ax1.text(station_data["Long_sta"].iloc[i], station_data["Lat_sta"].iloc[i], station_data["ID1"].iloc[i], fontsize=11, ha='right', va='center', color='black')
#     ax2.text(station_data["Long_sta"].iloc[i], station_data["Lat_sta"].iloc[i], station_data["ID1"].iloc[i], fontsize=11, ha='right', va='center', color='black')
# 在站点位置左边的100个像素点写字，字母被一个圆圈包围，字用蓝色
for i in range(len(station_data)):
    x = station_data["Long_sta"].iloc[i] - 0.4
    y = station_data["Lat_sta"].iloc[i]

    # # 绘制圆圈
    # circle1 = plt.Circle((x, y), radius=0.16, color='blue', fill=False, zorder=4)
    # ax1.add_patch(circle1)
    # circle2 = plt.Circle((x, y), radius=0.16, color='blue', fill=False, zorder=4)
    # ax2.add_patch(circle2)
    ax1.text(x, y, station_data["ID1"].iloc[i], fontsize=11, ha='center', va='center', color='white', zorder=5)
    ax2.text(x, y, station_data["ID1"].iloc[i], fontsize=11, ha='center', va='center', color='white', zorder=5)
# 指定要在哪些点写入文字的索引列表
specified_indices = [0, 5,10, 15, 22, 25, 35, 45]  # 请根据实际需要修改这个列表
# 在每个点上标记时间
for i in range(len(lon)):
    if i in specified_indices:
        ax1.text(lon[i], lat[i], time[i], fontsize=11, rotation=45, ha='left', va='bottom', color='black', alpha=0.6, fontweight='bold')
        ax2.text(lon[i], lat[i], time[i], fontsize=11, rotation=45, ha='left', va='bottom', color='black', alpha=0.6, fontweight='bold')
#plt.suptitle("Typhoon Doksuri ID:2305", fontsize=16, y=0.73)
plt.legend(loc=4, prop={'size': 10})
plt.savefig(r"F:\F TaiFong\PAPER\Submissions Needing Revision QAQ\Figure 5ab.jpg", dpi=480, facecolor='white', bbox_inches='tight')
plt.show()
