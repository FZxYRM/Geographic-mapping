# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 13:12:02 2023

@author: YLM xs- 

"""

'''
东亚大槽强度：
长江中游地区霾日的年际和年代际变化及其城乡差异成因研究:
    东亚大槽用( 25 45° N、110 145° E) 区 域 内 500 hPa 高度 场 的 平 均 值 作 标 准 化 后 的 值 表 示;

标准化值（Z） = （X - 平均值） / 标准差；  平均值，标准差为当前时间 高度场数据集合计算得出【a,b,c,……
                                        X为当前时次每个格点值相加取平均                                 …………
                                                                                ……,d,e,f】

'''
import netCDF4
from netCDF4 import Dataset
import pandas as pd
import os
import numpy as np

#part 1
# 文件夹路径
dir_path = "F:\F TaiFong\ERA5\E4"
# # 创建一个空的DataFrame来存储数据
data = pd.DataFrame(columns=["Time","CQ"])
# 遍历文件夹中的nc文件
for filename in sorted(os.listdir(dir_path)):
    if filename.endswith(".nc"):
        file_path = os.path.join(dir_path, filename)
        print(f"Processing file: {file_path}")
        # 打开NetCDF文件
        ncfile = Dataset(file_path)
        time_values = ncfile["time"][:]
        # 遍历时间步骤
        for i in range(len(time_values)):
            time_i = time_values[i]
            z = ncfile["z"][:]/98.0665
            z_values=z.data
            long = ncfile["longitude"][:]
            lat = ncfile["latitude"][:]
            ls=[]
        # 每小时 20-25纬度 全经度
            for w in range(18,27):
                for v in range(8,23):
                    ls.append(z[i,w,v])
            mean=np.mean(ls)
        
            # 转换时间格式（以小时为单位，累积从1900年1月1日0时开始）
            hours_since_1900 = int(time_i)
            start_time = pd.to_datetime("1900-01-01 00:00:00")
            current_time = start_time + pd.to_timedelta(hours_since_1900, unit='h')
            time_str = current_time.strftime("%Y%m%d%H")
            data = data.append({"Time":time_str,"CQ": mean}, ignore_index=True)


# # 将数据写入指定的Excel文件
output_file = "F:\F TaiFong\ERA5\副高2021\CQ.xlsx"
data.to_excel(output_file, index=False)


# part2 指定计算时次（也可参与全部计算）这里只选取1天内4时次
input_file = "F:\\F TaiFong\\ERA5\\副高2021\\CQ.xlsx"
data = pd.read_excel(input_file)
# 明确将Time列转换为字符串类型
data['Time'] = data['Time'].astype(str)
# 提取时间中的日期部分
data['Date'] = data['Time'].str[:8]
# 选择特定时间点的数据（00时、06时、12时和18时）
selected_times = ['00', '06', '12', '18']
filtered_data = data[data['Time'].str[-2:].isin(selected_times)]
print(filtered_data)
# 按日期分组并计算每日参数数据的平均值
daily_averages = filtered_data.groupby('Date')[['CQ']].mean()
# 重置索引以保持Time列
daily_averages.reset_index(inplace=True)
# 保存结果到新的xlsx文件
output_file = "F:\\F TaiFong\\ERA5\\副高2021\\日均副高指数.xlsx"
daily_averages.to_excel(output_file, index=False)


# part3
#时序标准化 得到大槽指数  *可以添加研究时间以外(historic前20年或更多)参与标准化计算
file_path = r"F:\F TaiFong\ERA5\副高2021\日均24小时CQ指数.xlsx"
data = pd.read_excel(file_path)
# 计算平均值和标准差
mean_value = data['CQ'].mean()
std_deviation = data['CQ'].std()
# 计算标准化值
data['Standardized_CQ'] = (data['CQ'] - mean_value) / std_deviation
# 创建新的DataFrame只包含时间和标准化值
result_data = data[['Time', 'Standardized_CQ']]
print(result_data,mean_value,std_deviation)
output_file = r"F:\F TaiFong\ERA5\副高2021\标准化CQ指数.xlsx"
result_data.to_excel(output_file, index=False)
