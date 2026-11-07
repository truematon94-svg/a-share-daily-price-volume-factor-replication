# -*- coding: utf-8 -*-
"""
03_factor_combination / step2_combine_and_backtest.py
====================================================
因子合成 + 纯多头回测（华泰之十口径：等权法 + 最大化 IC_IR 加权法）。

合成流程：
  1. 逐因子算 RankIC 序列（Spearman，去极值→标准化→行业+市值中性化后）
  2. 等权法：w = 1/N
  3. 最大化 IC_IR：等价凸 QP：min w'Σw s.t. w≥0, w'·|IC̄| = 1（Σ 为 RankIC 样本协方差+收缩）
  4. 合成打分 = Σ w_k × 标准化因子_k
  5. 纯多头：每月按合成打分升序取前 10% 等权，看下月超额收益年化

运行：
  ./.venv/Scripts/python.exe notebooks/03_factor_combination/step2_combine_and_backtest.py
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
PANEL = os.path.join(_ROOT, "data", "processed", "monthly_panel.csv")
IMP = os.path.join(_ROOT, "data", "processed", "improved_factors.csv")

FACTORS = [
    "return_1m", "return_3m", "return_6m", "return_12m",
    "wgt_return_1m", "wgt_return_3m", "wgt_return_6m", "wgt_return_12m",
    "exp_wgt_return_1m", "exp_wgt_return_3m", "exp_wgt_return_6m", "exp_wgt_return_12m",
]
TOP_Q = 0.10


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


def rank_ic_series(panel, factor):
    ics = []
    for _, g in panel.groupby("cross_date"):
        g = g.set_index("code")
        s, mc, ind, fwd = g[factor], g["market_cap"], g["industry_sw_l1"], g["fwd_excess_return"]
        if s.notna().sum() < 30:
            continue
        w = winsorize_mad(s)
        z = (w - w.mean()) / w.std()
        r = residualize(z, mc, ind)
        m = r.notna() & fwd.notna()
        if m.sum() >= 10:
            ics.append(r[m].corr(fwd[m], method="spearman"))
    return pd.Series(ics).dropna()


def max_icir_weights(ic_df):
    """最大化 IC_IR 权重（等价凸 QP，w≥0）。"""
    ic_mean = ic_df.mean().abs().values          # 用 |IC̄| 当强度
    cov = ic_df.cov().values                     # RankIC 样本协方差
    n = len(ic_mean)
    cov = 0.8 * cov + 0.2 * np.trace(cov) / n * np.eye(n)   # 向单位阵收缩，避免病态

    # 等价凸 QP：min w'Σw，s.t. w≥0, w'·ic_mean = 1
    def obj(w):
        return w @ cov @ w
    res = minimize(obj, np.ones(n) / n, method="SLSQP",
                   bounds=[(0, None)] * n,
                   constraints={"type": "eq", "fun": lambda w: w @ ic_mean - 1})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def add_composite(panel, w, col):
    """给 panel 加合成打分列。"""
    parts = []
    for date, g in panel.groupby("cross_date"):
        g = g.set_index("code")
        mat = g[FACTORS]
        z = (mat - mat.mean()) / mat.std()
        s = z.mul(w, axis=1).sum(axis=1)
        parts.append(pd.DataFrame({"cross_date": date, "code": s.index, col: s.values}))
    sc = pd.concat(parts, ignore_index=True)
    return panel.merge(sc, on=["cross_date", "code"], how="left")


def backtest(panel, score_col):
    """纯多头：每月按 score 升序取前 TOP_Q 等权，下月超额收益年化。"""
    rets = []
    for _, g in panel.groupby("cross_date"):
        s = g[score_col]
        fwd = g["fwd_excess_return"]
        m = s.notna() & fwd.notna()
        if m.sum() < 30:
            continue
        k = max(1, int(m.sum() * TOP_Q))
        top = s[m].nsmallest(k).index          # 反转：打分最低 = 多头
        rets.append(fwd[m][top].mean())
    rets = pd.Series(rets).dropna()
    return rets.mean() * 12, (1 + rets).prod() ** (12 / len(rets)) - 1


def main():
    panel = pd.read_csv(PANEL).merge(pd.read_csv(IMP), on=["cross_date", "code"], how="left")
    panel["industry_sw_l1"] = panel["industry_sw_l1"].fillna("未知")

    # 1. 各因子 RankIC 序列
    ic_df = pd.DataFrame({f: rank_ic_series(panel, f) for f in FACTORS})
    print("RankIC 均值（绝对值）:")
    print(ic_df.mean().abs().round(3).to_string())
    print()

    # 2. 权重
    w_ir = max_icir_weights(ic_df)
    w_eq = np.ones(len(FACTORS)) / len(FACTORS)
    print("权重对比（等权 vs 最大化IC_IR）:")
    for f, we, wi in zip(FACTORS, w_eq, w_ir):
        print("  %-22s 等权 %.3f   IC_IR %.3f" % (f, we, wi))
    print()

    # 3. 合成打分
    panel = add_composite(panel, w_eq, "score_eq")
    panel = add_composite(panel, w_ir, "score_ir")

    # 4. 回测
    for name, col in [("等权", "score_eq"), ("最大化IC_IR", "score_ir")]:
        ann, ann_c = backtest(panel, col)
        print("%s 纯多头(前10%%)：简单年化 %.2f%%，复合年化 %.2f%%" % (name, ann * 100, ann_c * 100))

    print()
    print("单因子对照（前10%多头简单年化）:")
    for f in ["return_1m", "return_3m", "exp_wgt_return_3m"]:
        ann, _ = backtest(panel, f)
        print("  %-22s %.2f%%" % (f, ann * 100))


if __name__ == "__main__":
    main()
