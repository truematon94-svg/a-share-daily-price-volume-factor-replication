# -*- coding: utf-8 -*-
"""
03_factor_combination / step1_factor_correlation.py
====================================================
因子截面相关性（华泰口径：各因子两两截面相关系数的全区间均值）。

读 monthly_panel.csv + improved_factors.csv，内存合并，对 12 个因子逐截面
计算两两 Pearson 相关，再取全区间均值，输出相关矩阵。

运行：
  ./.venv/Scripts/python.exe notebooks/03_factor_combination/step1_factor_correlation.py
"""

import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
PANEL = os.path.join(_ROOT, "data", "processed", "monthly_panel.csv")
IMP = os.path.join(_ROOT, "data", "processed", "improved_factors.csv")

FACTORS = [
    "return_1m", "return_3m", "return_6m", "return_12m",
    "wgt_return_1m", "wgt_return_3m", "wgt_return_6m", "wgt_return_12m",
    "exp_wgt_return_1m", "exp_wgt_return_3m", "exp_wgt_return_6m", "exp_wgt_return_12m",
]


def main():
    panel = pd.read_csv(PANEL).merge(pd.read_csv(IMP), on=["cross_date", "code"], how="left")

    # 逐截面两两相关，NaN-aware 平均（预热月里部分因子全 NaN，跳过不计）
    corr_list = []
    for _, g in panel.groupby("cross_date"):
        g = g[FACTORS]
        if g.notna().sum().sum() < len(FACTORS) * 30:   # 截面有效样本太少则跳过
            continue
        corr_list.append(g.corr())

    corr = pd.concat(corr_list).groupby(level=0).mean().round(2)

    print("因子截面相关矩阵（全区间均值，%d 个截面）" % len(corr_list))
    print()
    print(corr.to_string())


if __name__ == "__main__":
    main()
