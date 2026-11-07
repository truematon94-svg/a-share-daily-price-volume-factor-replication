# -*- coding: utf-8 -*-
"""
02_single_factor_tests / step2_grouping.py
==========================================
十组分层回测（全市场排序版，华泰原版是行业内排序，后续补）。

每个因子逐截面：
  1. 按因子值从小到大排序，分 10 组（等数量）
  2. 组 1 = 因子最小（过去跌最多），组 10 = 因子最大
  3. 每组下月超额收益 = 组内个股 fwd_excess_return 等权平均
  4. 多空 = 组 1 - 组 10

A 股反转：组 1 收益应最高、组 10 最低，多空为正。

运行：
  ./.venv/Scripts/python.exe notebooks/02_single_factor_tests/step2_grouping.py
"""

import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
PANEL_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "monthly_panel.csv")

FACTORS = ["return_1m", "return_3m", "return_6m", "return_12m"]
N_GROUPS = 10


def main():
    panel = pd.read_csv(PANEL_PATH)

    for f in FACTORS:
        group_rets = {i: [] for i in range(N_GROUPS)}
        for date, g in panel.groupby("cross_date"):
            s, fwd = g[f], g["fwd_excess_return"]
            m = s.notna() & fwd.notna()
            if m.sum() < 100:
                continue
            try:
                grp = pd.qcut(s[m], N_GROUPS, labels=False, duplicates="drop")
            except ValueError:
                continue
            for i in range(N_GROUPS):
                sel = grp == i
                if sel.sum() > 0:
                    group_rets[i].append(fwd[m][sel].mean())

        annual = []
        print("\n因子 %s（组1=因子最小=多头，年化=月均×12）" % f)
        for i in range(N_GROUPS):
            r = pd.Series(group_rets[i]).dropna()
            ann = r.mean() * 12
            annual.append(ann)
            print("  组%2d 年化超额 %8.2f%%" % (i + 1, ann * 100))
        ls = annual[0] - annual[-1]
        print("  多空(组1-组10)年化 %8.2f%%" % (ls * 100))


if __name__ == "__main__":
    main()
