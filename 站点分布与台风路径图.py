# # -*- coding: utf-8 -*-
# """
# Created on Mon Oct  9 11:41:30 2023

# @author: YLM xs-
# """


# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import matplotlib.patches as mpatches
# import cartopy.crs as ccrs
# from map_funs import add_Chinese_provinces, set_map_extent_and_ticks
# import cartopy.feature as cfeature
# import os
# import pandas as pd
# from pathlib import Path
# from typing import List
# from typing import Union
# from typing import Tuple

# if __name__ == '__main__':
#     # 设置绘图区域.
#     lonmin, lonmax = 90, 135
#     latmin, latmax = 15, 50
#     extents = [lonmin, lonmax, latmin, latmax]
    
#     proj = ccrs.PlateCarree()
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection=proj)

#     # 添加海岸线和中国省界.
#     ax.coastlines(resolution='10m', lw=0.3)
#     ax.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.3)
#     # ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)
#     # ax.add_feature(cfeature.LAKES, edgecolor='blue', linewidth=0.3)
#     ax.add_feature(cfeature.OCEAN, edgecolor='blue', linewidth=0.3)
#     add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
#     # 设置经纬度刻度.
#     set_map_extent_and_ticks(
#         ax, extents,
#         xticks=np.arange(-180, 190, 15),
#         yticks=np.arange(-90, 100, 15),
#         nx=1, ny=1
#     )
#     ax.tick_params(labelsize='small')
#     plt.savefig("F:\F TaiFong\地图.png",dpi=1080,facecolor='white') # 保存图片
#     fig.show()







# import pandas as pd
# # 读取CSV文件
# data = pd.read_excel("F:\F TaiFong\China_Sta.xlsx")


# # 遍历每一行数据
# for index, row in data.iterrows():
#     # 处理纬度数据
#     lat_deg, lat_min_sec, _ = row['lat'].split()
#     lat_deg = int(lat_deg)
#     lat_min_sec = lat_min_sec[:-1]  # 去掉最后的字母N
#     lat_decimal = f"{lat_deg}.{lat_min_sec}"

#     # 处理经度数据
#     long_deg, long_min_sec, _ = row['long'].split()
#     long_deg = int(long_deg)
#     long_min_sec = long_min_sec[:-1]  # 去掉最后的字母E
#     long_decimal = f"{long_deg}.{long_min_sec}"

#     # 打印格式化后的经纬度数据
#     print(f"纬度: {lat_decimal}, 经度: {long_decimal}")


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from matplotlib.lines import Line2D
from map_funs import add_Chinese_provinces, set_map_extent_and_ticks

# 读取数据
ds = xr.open_dataset(r"F:\ETOPO2v2c_f4.nc")
lon1 = np.linspace(min(ds['x'].data), max(ds['x'].data), len(ds['x'].data))
lat1 = np.linspace(min(ds['y'].data), max(ds['y'].data), len(ds['y'].data))
lon1, lat1 = np.meshgrid(lon1, lat1)
dem = ds['z'].data

# 设置绘图区域
lonmin, lonmax = 75, 135
latmin, latmax = 15, 55
extents = [lonmin, lonmax, latmin, latmax]
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(10, 15))
ax = fig.add_subplot(111, projection=proj)

# 添加地图要素
ax.coastlines(resolution='10m', lw=0.3)
ax.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.3)
ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)
ax.add_feature(cfeature.OCEAN, edgecolor='blue', linewidth=0.3)
add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')

# 设置经纬度刻度
set_map_extent_and_ticks(
    ax, extents,
    xticks=np.arange(-180, 190, 5),
    yticks=np.arange(-90, 100, 5),
    nx=0.1, ny=0.1
)
ax.tick_params(labelsize='small')

# 绘制 DEM
levels = [-8000, -6000, -4000, -2000, -1000, -200, -50, 0, 50, 200, 500, 1000, 1500, 2000, 3000, 4000, 
          5000, 6000, 7000, 8000]
color = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#006837', 
         '#31a354', '#78c679', '#addd8e', '#d9f0a3', '#f7fcb9', '#c9bc87', '#a69165', '#856b49', 
         '#664830', '#ad9591', '#d7ccca']

ax.contourf(lon1, lat1, dem, levels=levels, extend='both', colors=color, vmin=0, vmax=7000)

# 读取站点数据
data = pd.read_excel("F:\F TaiFong\影响中国的热带气旋降水预测研究：基于机器学习的探索\气象态\站点评估数据.xlsx")

# 绘制站点
for index, row in data.iterrows():
    lat_decimal = row['Lat']
    long_decimal = row['Long']
    ax.plot(float(long_decimal), float(lat_decimal), 'o', color='white', markersize=4, transform=proj, zorder=4)

# 添加图例
station_legend = Line2D([0], [0], marker='o', color='white', label='Station', markersize=2)
ax.legend(handles=[station_legend], loc='lower right', fontsize='small', markerscale=8, handlelength=1)

# 添加标题
plt.title('(b)', fontsize=18, loc='left')

# 保存图片
plt.savefig("F:\F TaiFong\PAPER\Submissions Needing Revision QAQ\Figure\Figure 1b.jpg", dpi=480, facecolor='white', bbox_inches='tight')
plt.show()




#所有台风

# def read_typhoon_path(typhoon_txt, code):
#     typhoon_txt = Path(typhoon_txt)
#     if isinstance(code, int):
#         code = "{:04}".format(code)
#     with open(typhoon_txt, "r") as txt_handle:
#         while True:
#             header = txt_handle.readline().split()
#             if not header:
#                 raise ValueError(f"没有在文件里找到编号为{code}的台风")
#             if header[4].strip() == code:
#                 break
#             [txt_handle.readline() for _ in range(int(header[2]))]
#         data_path = pd.read_table(
#             txt_handle,
#             sep=r"\s+",
#             header=None,
#             names=["TIME", "I", "LAT", "LONG", "PRES", "WND", "OWD"],
#             nrows=int(header[2]),
#             dtype={
#                 "I": int,
#                 "LAT": np.float32,
#                 "LONG": np.float32,
#                 "PRES": np.float32,
#                 "WND": np.float32,
#                 "OWD": np.float32,
#                 },

#             parse_dates=True,
#             date_parser=lambda x: pd.to_datetime(x, format=f"%Y%m%d%H"),
#             index_col="TIME",
#         )
#         data_path["LAT"] = data_path["LAT"] / 10
#         data_path["LONG"] = data_path["LONG"] / 10
#         return header, data_path


# # 设置绘图区域.
# lonmin, lonmax = 90, 150
# latmin, latmax = 10, 50
# extents = [lonmin, lonmax, latmin, latmax]
# proj = ccrs.PlateCarree()
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection=proj)

# # 添加海岸线和中国省界.
# ax.coastlines(resolution='10m', lw=0.3)
# ax.add_feature(cfeature.LAND, edgecolor='black', linewidth=0.3)
# ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)
# ax.add_feature(cfeature.OCEAN, edgecolor='blue', linewidth=0.3)
# add_Chinese_provinces(ax, lw=0.6, ec='k', fc='none')
# # 设置经纬度刻度.
# set_map_extent_and_ticks(
#     ax, extents,
#     xticks=np.arange(-180, 190, 5),
#     yticks=np.arange(-90, 100, 5),
#     nx=0.1, ny=0.1
# )
# ax.tick_params(labelsize='small')



# # 文件夹路径
# typhoon_data_folder = "F:\F TaiFong\影响中国的热带气旋降水预测研究：基于机器学习的探索\数据\CMA - 副本 (2)"

# # 遍历文件夹内的每个文件
# for typhoon_file in os.listdir(typhoon_data_folder):
#     if typhoon_file.endswith(".txt"):  # 假设文件扩展名为.txt
#         # 初始化台风编号列表
#         typhoon_codes = []
        
#         # 读取当前文件的所有台风编号
#         with open(os.path.join(typhoon_data_folder, typhoon_file), "r") as txt_handle:
#             for line in txt_handle:
#                 if line.startswith("66666"):  # 根据文件格式找到编号行
#                     typhoon_codes.extend(line.split()[4].split(','))  # 获取台风编号并拆分成多个
                    
#         # 遍历当前文件的所有台风编号并绘制台风路径
#         for code in typhoon_codes:
#             try:
#                 header, data_path = read_typhoon_path(os.path.join(typhoon_data_folder, typhoon_file), code=int(code))
#                 # 绘制台风路径的代码
#                 plt.plot(data_path["LONG"], data_path["LAT"], linewidth=0.7,alpha=0.8)
#             except Exception as e:
#                 print(f"处理文件时出错：{typhoon_file}, code：{code}, 错误信息：{e}")
            
# # 显示图例
# ax.contourf(lon1, lat1, dem, levels=levels, extend='both', colors=color, vmin=0, vmax=7000)
# plt.legend()
# plt.title('(a)', fontsize=18, loc='left')
# plt.savefig("F:\F TaiFong\PAPER\Submissions Needing Revision QAQ\Figure\Figure 1.jpg", dpi=480, facecolor='white', bbox_inches='tight')  # 保存图片
# plt.show()




