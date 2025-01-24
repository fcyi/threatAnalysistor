import numpy as np
import math
import matplotlib.pyplot as plt


def calculate_distance(jing1, wei1, jing2, wei2):
    # 基于两点的经纬度计算两点之间的距离（米）
    lng1, lat1, lng2, lat2 = map(math.radians, [jing1, wei1, jing2, wei2])  # 经纬度转换成弧度
    # haversine公式
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    distance = 2 * math.asin(math.sqrt(a)) * 6371.393 * 1000  # 地球平均半径，6371.393km

    return distance


def get_angle(lonA, latA, lonB, latB):
    """
    参数:
        点A (lonA, latA)
        点B (lonB, latB)
    返回:
        点B基于点A的方向，以北为基准,顺时针方向的角度，0~360度
        计算一点基于另外一点方向的航向角，以北为基准
    """
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)
    dLon = radLonB - radLonA
    y = math.sin(dLon) * math.cos(radLatB)
    x = math.cos(radLatA) * math.sin(radLatB) - math.sin(radLatA) * math.cos(radLatB) * math.cos(dLon)
    angle = math.degrees(math.atan2(y, x))
    angle = (angle + 360) % 360
    return angle



