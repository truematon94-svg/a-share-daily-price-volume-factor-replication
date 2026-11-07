# -*- coding: utf-8 -*-
"""
02_single_factor_tests / step5_rankic_all_factors.py
====================================================
全部 12 个因子的 IC 检验（return / wgt_return / exp_wgt_return）。

独立脚本：读 monthly_panel.csv + improved_factors.csv，内存合并（不覆盖文件），
对全部因子逐截面：去极值(MAD±5) → 标准化 → 行业+市值中性化 → IC(Pearson)。

运行：
  ./.venv/Scripts/python.exe notebooks/02_single_factor_tests/step5_rankic_all_factors.py
"""

import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
PANEL_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "monthly_panel.csv")
IMP_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "improved_factors.csv")

FACTORS = [
    "return_1m", "return_3m", "return_6m", "return_12m",
    "wgt_return_1m", "wgt_return_3m", "wgt_return_6m", "wgt_return_12m",
    "exp_wgt_return_1m", "exp_wgt_return_3m", "exp_wgt_return_6m", "exp_wgt_return_12m",
]
MAD_K = 5.0


def winsorize_mad(s, k=MAD_K):
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return s
    return s.clip(lower=med - k * mad, upper=med + k * mad)


def residualize(s, mcap, industry):
    idx = s.index
    X = pd.DataFrame({"ln_mcap": np.log(mcap).reindex(idx).values}, index=idx)
    X = pd.concat([X, pd.get_dummies(industry.reindex(idx), prefix="ind").astype(float)], axis=1)
    X = X.assign(const=1.0)
    y = s.values.astype(float)
    mask = (~np.isnan(y)) & (np.log(mcap.reindex(idx).values) > -np.inf)
    Xm, ym = X[mask].values, y[mask]
    if Xm.shape[0] < Xm.shape[1] + 10:
        return pd.Series(np.nan, index=idx)
    beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    resid = np.full(len(y), np.nan)
    resid[mask] = ym - Xm @ beta
    return pd.Series(resid, index=idx)


def ic(factor, fwd):
    m = factor.notna() & fwd.notna()
    if m.sum() < 10:
        return np.nan
    return factor[m].corr(fwd[m], method="pearson")


def main():
    panel = pd.read_csv(PANEL_PATH)
    imp = pd.read_csv(IMP_PATH)
    panel = panel.merge(imp, on=["cross_date", "code"], how="left")  # 内存合并，不落盘
    panel["industry_sw_l1"] = panel["industry_sw_l1"].fillna("未知")

    print("合并后：%d 行，%d 截面" % (len(panel), panel["cross_date"].nunique()))
    print()
    print("%-20s %10s %10s %9s %9s" % ("因子", "IC均值", "IC标准差", "IR比率", "IC>0占比"))

    for f in FACTORS:
        ics = []
        for _, g in panel.groupby("cross_date"):
            g = g.set_index("code")
            s, mcap, ind, fwd = g[f], g["market_cap"], g["industry_sw_l1"], g["fwd_excess_return"]
            if s.notna().sum() < 30:
                continue
            w = winsorize_mad(s)
            z = (w - w.mean()) / w.std()
            r = residualize(z, mcap, ind)
            ics.append(ic(r, fwd))
        ics = pd.Series(ics).dropna()
        ir = abs(ics.mean() / ics.std()) if ics.std() > 0 else np.nan
        print("%-20s %10.4f %10.4f %9.3f %8.1f%%" %
              (f, ics.mean(), ics.std(), ir, (ics > 0).mean() * 100))


if __name__ == "__main__":
    main()
