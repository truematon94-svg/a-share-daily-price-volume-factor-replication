# -*- coding: utf-8 -*-
"""
02_single_factor_tests / step1_rankic.py
========================================
单因子 RankIC 检验（华泰口径）。

读 monthly_panel.csv，对每个因子逐截面：
  1. 去极值：中位数 ± 5×MAD 缩尾
  2. 标准化：Z-score
  3. 中性化：对 ln(总市值) + 申万一级行业哑变量 做 OLS 回归，取残差
  4. IC = 残差与下月超额收益的 Pearson 相关系数（华泰原文 corr 即 Pearson）

输出：每个因子的 IC均值、IC标准差、IR比率、IC>0占比，对照华泰原文
（return_1m：IC 均值约 -5.7%、IR 约 0.63，方向为负 = 反转）。

运行（本地，先下载 monthly_panel.csv 到 data/processed/）：
  ./.venv/Scripts/python.exe notebooks/02_single_factor_tests/step1_rankic.py
"""

import os

import numpy as np
import pandas as pd

# 相对脚本自身定位项目根目录，避免受运行目录影响
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
PANEL_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "monthly_panel.csv")
FACTORS = ["return_1m", "return_3m", "return_6m", "return_12m"]
MAD_K = 5.0


def winsorize_mad(s, k=MAD_K):
    """中位数 ± k*MAD 缩尾。"""
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return s
    return s.clip(lower=med - k * mad, upper=med + k * mad)


def residualize(s, mcap, industry):
    """对 ln(市值) + 行业哑变量(+截距) 做 OLS，返回残差（中性化）。"""
    idx = s.index
    X = pd.DataFrame({"ln_mcap": np.log(mcap).reindex(idx).values}, index=idx)
    dummies = pd.get_dummies(industry.reindex(idx), prefix="ind").astype(float)
    X = pd.concat([X, dummies], axis=1)
    X = X.assign(const=1.0)
    y = s.values.astype(float)

    # 只用 mcap>0 且非 NaN 的样本回归
    mask = (~np.isnan(y)) & (np.log(mcap.reindex(idx).values) > -np.inf)
    Xm, ym = X[mask].values, y[mask]
    if Xm.shape[0] < Xm.shape[1] + 10:
        return pd.Series(np.nan, index=idx)

    beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    resid = np.full(len(y), np.nan)
    resid[mask] = ym - Xm @ beta
    return pd.Series(resid, index=idx)


def rank_ic(factor, fwd):
    """Pearson 相关系数（华泰 IC 口径），丢弃 NaN 对。"""
    m = factor.notna() & fwd.notna()
    if m.sum() < 10:
        return np.nan
    return factor[m].corr(fwd[m], method="pearson")


def main():
    panel = pd.read_csv(PANEL_PATH)
    panel["industry_sw_l1"] = panel["industry_sw_l1"].fillna("未知")

    print("面板：%d 行，%d 截面，%d 股票" %
          (len(panel), panel["cross_date"].nunique(), panel["code"].nunique()))
    print()
    print("%-12s %10s %10s %9s %9s" % ("因子", "IC均值", "IC标准差", "IR比率", "IC>0占比"))

    for f in FACTORS:
        ics = []
        for date, g in panel.groupby("cross_date"):
            g = g.set_index("code")
            s, mcap, ind, fwd = g[f], g["market_cap"], g["industry_sw_l1"], g["fwd_excess_return"]
            if s.notna().sum() < 30:
                continue
            w = winsorize_mad(s)                       # 1. 去极值
            z = (w - w.mean()) / w.std()               # 2. 标准化
            r = residualize(z, mcap, ind)              # 3. 中性化
            ics.append(rank_ic(r, fwd))                # 4. RankIC
        ics = pd.Series(ics).dropna()
        ir = abs(ics.mean() / ics.std()) if ics.std() > 0 else np.nan
        print("%-12s %10.4f %10.4f %9.3f %8.1f%%" %
              (f, ics.mean(), ics.std(), ir, (ics > 0).mean() * 100))


if __name__ == "__main__":
    main()
