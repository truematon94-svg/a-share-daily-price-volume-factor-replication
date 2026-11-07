# -*- coding: utf-8 -*-
"""
02_single_factor_tests / step6_compare_halflife.py
====================================================
对比不同 exp 半衰期变体的 IC，定位最接近华泰的衰减口径。

华泰原文 IR（中性化后）：
  return_1m 0.63, wgt_return_1m 0.92, exp_wgt_return_1m 0.75
  exp_wgt_return_3m 0.96, exp_wgt_return_6m 0.92, exp_wgt_return_12m 0.78
"""

import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
PANEL = os.path.join(_ROOT, "data", "processed", "monthly_panel.csv")
IMP = os.path.join(_ROOT, "data", "processed", "improved_factors.csv")


def winsorize_mad(s, k=5.0):
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return s
    return s.clip(lower=med - k * mad, upper=med + k * mad)


def residualize(s, mcap, industry):
    idx = s.index
    X = pd.DataFrame({"ln": np.log(mcap).reindex(idx).values}, index=idx)
    X = pd.concat([X, pd.get_dummies(industry.reindex(idx), prefix="i").astype(float)], axis=1)
    X = X.assign(c=1.0)
    y = s.values.astype(float)
    m = (~np.isnan(y)) & (np.log(mcap.reindex(idx).values) > -np.inf)
    Xm, ym = X[m].values, y[m]
    if Xm.shape[0] < Xm.shape[1] + 10:
        return pd.Series(np.nan, index=idx)
    b, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    r = np.full(len(y), np.nan)
    r[m] = ym - Xm @ b
    return pd.Series(r, index=idx)


def ir(f, panel):
    ics = []
    for _, g in panel.groupby("cross_date"):
        g = g.set_index("code")
        s, mc, ind, fwd = g[f], g["market_cap"], g["industry_sw_l1"], g["fwd_excess_return"]
        if s.notna().sum() < 30:
            continue
        z = (winsorize_mad(s) - winsorize_mad(s).mean()) / winsorize_mad(s).std()
        r = residualize(z, mc, ind)
        m = r.notna() & fwd.notna()
        if m.sum() >= 10:
            ics.append(r[m].corr(fwd[m], method="pearson"))
    ics = pd.Series(ics).dropna()
    return abs(ics.mean() / ics.std()) if ics.std() > 0 else np.nan


def main():
    panel = pd.read_csv(PANEL).merge(pd.read_csv(IMP), on=["cross_date", "code"], how="left")
    panel["industry_sw_l1"] = panel["industry_sw_l1"].fillna("未知")

    # 华泰对照
    ht = {
        "return_1m": 0.63, "return_3m": 0.43, "return_6m": 0.32, "return_12m": 0.25,
        "wgt_return_1m": 0.92, "wgt_return_3m": 0.74, "wgt_return_6m": 0.60, "wgt_return_12m": 0.51,
        "exp_wgt_return_1m": 0.75, "exp_wgt_return_3m": 0.96, "exp_wgt_return_6m": 0.92, "exp_wgt_return_12m": 0.78,
    }

    print("因子                      我的IR   华泰IR   差距")
    print("-" * 55)
    for n in [1, 3, 6, 12]:
        for name, ht_key in [
            ("return_%dm" % n, "return_%dm" % n),
            ("wgt_return_%dm" % n, "wgt_return_%dm" % n),
            ("exp_wgt_hl1_%dm" % n, "exp_wgt_return_%dm" % n),
            ("exp_wgt_hl2_%dm" % n, "exp_wgt_return_%dm" % n),
            ("exp_wgt_hl4_%dm" % n, "exp_wgt_return_%dm" % n),
        ]:
            if name not in panel.columns:
                continue
            my_ir = ir(name, panel)
            ht_ir = ht.get(ht_key, np.nan)
            gap = my_ir - ht_ir if not np.isnan(my_ir) else np.nan
            print("%-24s %6.3f   %6.3f   %+6.3f" % (name, my_ir, ht_ir, gap))


if __name__ == "__main__":
    main()
