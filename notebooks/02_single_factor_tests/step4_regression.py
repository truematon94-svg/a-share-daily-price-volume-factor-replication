# -*- coding: utf-8 -*-
"""
02_single_factor_tests / step4_regression.py
============================================
回归法单因子检验（WLS，华泰口径）。

每个因子逐截面：
  1. 因子处理：去极值(MAD±5) → 标准化(Z-score) → 缺失置 0
  2. 回归：下月超额收益 ~ 截距 + 行业哑变量 + 因子
  3. 加权最小二乘 WLS，权重 = 流通市值平方根（Barra）
  4. 取因子的回归系数（因子收益率）和 t 值

输出（对照华泰图表111 return_1m）：
  |t|均值 ≈ 4.89，|t|>2 占比 ≈ 73%，t 均值 ≈ -2.7（负 = 反转），
  因子收益率均值 ≈ -0.64%，因子收益率序列 t 检验 ≈ -5.0。

运行：
  ./.venv/Scripts/python.exe notebooks/02_single_factor_tests/step4_regression.py
"""

import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
PANEL_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "monthly_panel.csv")

FACTORS = ["return_1m", "return_3m", "return_6m", "return_12m"]
MAD_K = 5.0
MIN_STOCKS = 30


def winsorize_mad(s, k=MAD_K):
    """中位数 ± k*MAD 缩尾。"""
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return s
    return s.clip(lower=med - k * mad, upper=med + k * mad)


def regression_test(panel, factor):
    betas, tvals = [], []
    for _, g in panel.groupby("cross_date"):
        s = g[factor]
        fwd = g["fwd_excess_return"]
        mcap = g["circulating_market_cap"]
        ind = g["industry_sw_l1"]

        # 因子有效样本太少（早期预热期全 NaN），跳过该截面
        if s.notna().sum() < MIN_STOCKS:
            continue

        # 1. 因子处理：去极值 → 标准化 → 缺失置 0
        s = winsorize_mad(s)
        s = (s - s.mean()) / s.std()
        s = s.fillna(0.0)

        m = fwd.notna() & mcap.notna() & (mcap > 0)
        if m.sum() < MIN_STOCKS:
            continue

        # 2. 设计矩阵：截距 + 行业哑变量(drop first) + 因子
        X = pd.DataFrame({"const": 1.0}, index=g.index)
        dummies = pd.get_dummies(ind, prefix="ind", drop_first=True).astype(float)
        X = pd.concat([X, dummies], axis=1)
        X["factor"] = s

        # 3. WLS：权重 = sqrt(流通市值)
        Xm = X[m].values
        ym = fwd[m].values
        wm = np.sqrt(mcap[m].values)      # 权重 = sqrt(流通市值)
        sw = np.sqrt(wm)                  # 变换因子 = sqrt(权重) = mcap^{1/4}

        Xw = Xm * sw[:, None]
        yw = ym * sw
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = yw - Xw @ beta
        n, k = Xw.shape
        dof = n - k
        if dof <= 0:
            continue
        sigma2 = (resid @ resid) / dof
        se = np.sqrt(np.diag(np.linalg.pinv(Xw.T @ Xw)) * sigma2)
        t = beta / se

        betas.append(beta[-1])   # 因子的回归系数（因子收益率）
        tvals.append(t[-1])      # 因子的 t 值

    betas = pd.Series(betas)
    tvals = pd.Series(tvals)
    n = len(betas)
    return {
        "|t|均值": tvals.abs().mean(),
        "|t|>2占比": (tvals.abs() > 2).mean(),
        "t均值": tvals.mean(),
        "t均值/t标准差": tvals.mean() / tvals.std(),
        "因子收益率均值": betas.mean(),
        "因子收益率序列t检验": betas.mean() / betas.std() * np.sqrt(n) if betas.std() > 0 else np.nan,
    }


def main():
    panel = pd.read_csv(PANEL_PATH)
    panel["industry_sw_l1"] = panel["industry_sw_l1"].fillna("未知")

    print("%-12s %8s %9s %8s %10s %12s %14s" %
          ("因子", "|t|均值", "|t|>2占比", "t均值", "t均值/t标准差", "因子收益率均值", "因子收益率t检验"))
    for f in FACTORS:
        r = regression_test(panel, f)
        print("%-12s %8.2f %8.1f%% %8.2f %10.2f %12.4f %14.2f" %
              (f, r["|t|均值"], r["|t|>2占比"] * 100, r["t均值"],
               r["t均值/t标准差"], r["因子收益率均值"], r["因子收益率序列t检验"]))


if __name__ == "__main__":
    main()
