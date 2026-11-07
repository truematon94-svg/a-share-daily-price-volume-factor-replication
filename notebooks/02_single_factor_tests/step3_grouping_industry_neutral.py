# -*- coding: utf-8 -*-
"""
02_single_factor_tests / step3_grouping_industry_neutral.py
============================================================
十组分层回测（行业中性版，华泰原版口径）。

每个因子逐截面：
  1. 在每个申万一级行业内部，按因子值升序分 10 组
  2. 组 1 = 行业内因子最小（过去跌最多），组 10 = 因子最大
  3. 每组下月超额收益 = 跨行业等权平均（行业成分各组相同 → 行业中性）
  4. 多空 = 组 1 - 组 10

A 股反转：组 1 收益最高、组 10 最低，多空为正。

运行：
  ./.venv/Scripts/python.exe notebooks/02_single_factor_tests/step3_grouping_industry_neutral.py
"""

import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
PANEL_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "monthly_panel.csv")

FACTORS = ["return_1m", "return_3m", "return_6m", "return_12m"]
N_GROUPS = 5   # 华泰主表用 5 层（10 层用于分年度排名，另见 step2）


def decile(s, n=N_GROUPS):
    """按值升序分 n 组（0=最小=组1，n-1=最大），等数量。"""
    r = s.rank(method="average")
    return ((r - 1) / r.max() * n).astype(int).clip(0, n - 1)


def main():
    panel = pd.read_csv(PANEL_PATH)
    panel["industry_sw_l1"] = panel["industry_sw_l1"].fillna("未知")

    for f in FACTORS:
        group_rets = {i: [] for i in range(N_GROUPS)}
        for date, g in panel.groupby("cross_date"):
            labels = pd.Series(np.nan, index=g.index)
            # 行业内分组
            for _, ig in g.groupby("industry_sw_l1"):
                s = ig[f]
                m = s.notna() & ig["fwd_excess_return"].notna()
                if m.sum() < 10:
                    continue
                labels.loc[ig.index[m]] = decile(s[m]).values
            # 每组收益（跨行业合并，组内等权）
            for i in range(N_GROUPS):
                sel = g[labels == i]
                if sel["fwd_excess_return"].notna().sum() > 0:
                    group_rets[i].append(sel["fwd_excess_return"].mean())

        annual = []
        print("\n因子 %s（行业中性，组1=因子最小=多头）" % f)
        for i in range(N_GROUPS):
            r = pd.Series(group_rets[i]).dropna()
            ann = r.mean() * 12
            annual.append(ann)
            print("  组%2d 年化超额 %8.2f%%" % (i + 1, ann * 100))
        print("  多空(组1-组%d)年化 %8.2f%%" % (N_GROUPS, (annual[0] - annual[-1]) * 100))


if __name__ == "__main__":
    main()
