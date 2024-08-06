# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:09:03 2023

@author: YLM xs- 202183820003@nuist.cn

暮色苍茫看劲松，乱石飞渡仍从容。
天生一个仙人洞，无限风光在险峰。

"""

import cdsapi
import calendar
from subprocess import call

def idmDownloader(task_url, folder_path, file_name):
    """
    IDM下载器
    :param task_url: 下载任务地址
    :param folder_path: 存放文件夹
    :param file_name: 文件名
    :return:
    """
    # IDM安装目录
    idm_engine = "C:\\Program Files (x86)\\Internet Download Manager\\IDMan.exe"
    # 将任务添加至队列
    call([idm_engine, '/d', task_url, '/p', folder_path, '/f', file_name, '/a'])
    # 开始任务队列
    call([idm_engine, '/s'])

if __name__ == '__main__':
    c = cdsapi.Client()  # 创建用户
    # 数据信息字典
    dic = {
        'product_type': 'reanalysis',  # 产品类型
        'format': 'netcdf',  # 数据格式
        'variable': ['geopotential', 'u_component_of_wind', 'v_component_of_wind',
        ],  # 变量名称
        'pressure_level':'500',
        'year': '',  # 年，设为空
        'month': '',  # 月，设为空
        'day': '',  # 日，设为空
        'time': ['00:00','06:00','12:00','18:00','19:00','20:00','21:00'],  # 小时
        'time': ['00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
                ],
        'area': [90,90,10,180], # 设置地理范围
        'grid': [2.5, 2.5]
    }
    for y in range(2023, 2024):  # 遍历年
        for m in range(5,13):  # 遍历月
            day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
            # 将年、月、日更新至字典中
            dic['year'] = str(y)
            dic['month'] = str(m).zfill(2)
            dic['day'] = [str(d).zfill(2) for d in range(1, day_num + 1)]

            r = c.retrieve('reanalysis-era5-pressure-levels', dic, )  # 文件下载器
            url = r.location  # 获取文件下载地址
            path = "F:\F TaiFong\ERA5\E4"  # 存放文件夹
            filename = str(y) + str(m).zfill(2) + '.nc'  # 文件名
            idmDownloader(url, path, filename)  # 添加进IDM中下载