# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 19:33:23 2023

@author: YLM xs- 
    杨柳堤 远方烟雨
        情人言语 画船人未起
            是谁在花港看鱼 而我在python :)
"""

dir = "F:\F TaiFong\ERA5-hours 500hpa"

'''
西太副高的面积（GM）:z≥588 gpm的格点所围成的面积总和(2.5*2.5)单位/个;每个格点S=6.25。没有就记录为-9999

强度（GQ）:z≥588 位势什米的格点所围成的面积与该格点高度值减去587gpm差值的乘积的总和(累加格点)。没有就记录为-9999

脊线位置（GX）：在10ºN以北、110ºE~150ºE范围内，
500hPa高度场上588gpm等值线所包围的西太副高体内纬向风切变线所在纬度位置的平均值
（若不存在588gpm等值线，则定义584gpm等值线内的纬向风切变线所在纬度位置的平均值；
若在某月不存在584gpm等值线，则以该日的最z最大值代替然后取平均）。

西伸脊点（GD）：500hPa高度场上588gpm最西格点所在的经度值
（若在90ºE以西则统一计为90ºE；若在某月不存在588gpm等值线，则以该月的历史最大值z所在的经度代替
'''
import netCDF4
from netCDF4 import Dataset
import pandas as pd
import os
import numpy as np




# 文件夹路径
dir_path = "F:\F TaiFong\ERA5-hours 500hpa"

# 创建一个空的DataFrame来存储数据
data = pd.DataFrame(columns=["GX"])

# 遍历文件夹中的nc文件
for filename in sorted(os.listdir(dir_path)):
    if filename.endswith(".nc"):
        file_path = os.path.join(dir_path, filename)
        print(f"Processing file: {file_path}")
        
        # 打开NetCDF文件
        ncfile = Dataset(file_path)
        long = ncfile["longitude"][:]
        lat = ncfile["latitude"][:]
        time_values = ncfile["time"][:]
        
        # 遍历时间步骤
        for i in range(len(time_values)):
            time_i = time_values[i]
            z = ncfile["z"][i]/98.0665
            z_values=z.data
            u = ncfile["u"][i]
            lat_ls=[]
            GX=0
            ls=[]
            
            h_588=0
            h_584=0
            for w in range(18,33):
                for v in range(8,25):
                    if z[w,v]>=588:
                        h_588+=1
                    elif z[w,v]>=584:
                        h_584+=1

            if h_588 >= 2:
                ls_lat=[]
                for j in range(16,33):
                    for k in range(8,25):
                        if z[j,k]>=588:
                            ls.append([j,k])
                            ls_lat.append(lat[j])
                

                for m in range(len(ls)):
                    if u[ls[m][0]-1,ls[m][1]]*u[ls[m][0],ls[m][1]]<0:
                        lat_ls.append(lat[ls[m][0]-1])
                        lat_ls.append(lat[ls[m][0]])
                if len(lat_ls)>0: 
                    GX=sum(lat_ls)/len(lat_ls)
                elif len(lat_ls)==0:
                    GX=sum(ls_lat)/len(ls_lat)

                    
            elif h_588 <= 1 and h_584 >= 2:
                for j in range(16,33):
                    for k in range(8,25):
                        if z[j,k]>=584:
                            ls.append([j,k])
                for m in range(len(ls)):
                    if u[ls[m][0]-1,ls[m][1]]*u[ls[m][0],ls[m][1]]<0:
                        lat_ls.append(lat[ls[m][0]-1])
                        lat_ls.append(lat[ls[m][0]])
                if len(lat_ls)>0: 
                    GX=sum(lat_ls)/len(lat_ls)
                elif len(lat_ls)==0:
                    GX=sum(ls_lat)/len(ls_lat)
                    

            else:
                # 寻找z的最大值（可能有多个相等的最大值）
                max_z_value = np.max(z_values)
                max_indices = np.argwhere(z_values == max_z_value)
                # 计算这些最大值对应的纬度的平均值
                lat_max_values = [lat[j] for j, k in max_indices]
                GX = np.mean(lat_max_values)

                
                

            # 转换时间格式（以小时为单位，累积从1900年1月1日0时开始）
            hours_since_1900 = int(time_i)
            start_time = pd.to_datetime("1900-01-01 00:00:00")
            current_time = start_time + pd.to_timedelta(hours_since_1900, unit='h')
            time_str = current_time.strftime("%Y%m%d")

            # 将数据添加到DataFrame中
            data = data.append({"Time": time_str, "GX": GX}, ignore_index=True)

# 将数据写入指定的Excel文件
output_file = "F:\F TaiFong\影响中国的热带气旋降水预测研究：基于机器学习的探索\气象态\副高指数2023.xlsx"
data.to_excel(output_file, index=False)


import netCDF4
from netCDF4 import Dataset
import pandas as pd
import os

# 文件夹路径
dir_path = 'F:\F TaiFong\ERA5-hours 500hpa'

# 创建一个空的DataFrame来存储数据
data = pd.DataFrame(columns=["Time", "GM", "GQ", "GX", "GD"])

# 遍历文件夹中的nc文件
for filename in sorted(os.listdir(dir_path)):
    if filename.endswith(".nc"):
        file_path = os.path.join(dir_path, filename)
        print(f"Processing file: {file_path}")
        
        # 打开NetCDF文件
        ncfile = Dataset(file_path)
        long = ncfile["longitude"][:]
        lat = ncfile["latitude"][:]
        time_values = ncfile["time"][:]
        z = ncfile["z"][:]/98.0665
        u = ncfile["u"][:]



        # 遍历时间步骤
        for i in range(len(time_values)):
            time_i = time_values[i]
            GM = 0
            GQ = 0
            GD = 0
            GX=0
            max_z_value = 0
            longitude = []

            for j in range(18,33):
                for k in range(37):
                    if z[i, j, k] > 588:
                        GQ += (6.25 * (z[i, j, k] - 587))
                        GM += 6.25
                        longitude.append(long[k])

                    # 找到最大值对应的经度
                    if z[i, j, k] > max_z_value:
                        max_z_value = z[i, j, k]
                        max_long = long[k]

            if len(longitude) > 0:
                GD = min(longitude)
            else:
                GD = max_long

            GM = round(GM, 1)
            GQ = round(GQ, 1)

            # 转换时间格式（以小时为单位，累积从1900年1月1日0时开始）
            hours_since_1900 = int(time_i)
            start_time = pd.to_datetime("1900-01-01 00:00:00")
            current_time = start_time + pd.to_timedelta(hours_since_1900, unit='h')
            time_str = current_time.strftime("%Y%m%d")

            # 将数据添加到DataFrame中
            data = data.append({"Time": time_str, "GM": GM, "GQ": GQ, "GX": GX, "GD": GD}, ignore_index=True)

# 将数据写入指定的Excel文件
output_file = "F:\F TaiFong\影响中国的热带气旋降水预测研究：基于机器学习的探索\气象态\副高指数2023A.xlsx"
data.to_excel(output_file, index=False)
print(f"Data written to {output_file}")

