# -*- coding: utf-8 -*-
"""
01_data_pipeline / step2_build_improved_factors.py
====================================================
计算改进因子 wgt_return（换手率加权）与 exp_wgt_return（指数衰减换手率加权）。

权重用「换手率等价」= 成交量 / 复权因子 factor：
  换手率 = 成交量/流通股本；送股/转增会让股本翻倍、成交量也翻倍，但换手率不变。
  复权因子 factor 与股本同比例变化，故 成交量/factor ∝ 换手率，可消除送股造成的跳增。

产出 improved_factors.csv，之后与 monthly_panel.csv 按 (cross_date, code) 合并。
运行：聚宽研究环境。断点续跑。
"""

import os
import time

import numpy as np
import pandas as pd

SAMPLE_START = "2005-01-01"
SAMPLE_END = "2016-12-31"
BENCH = "000300.XSHG"
WINDOWS = [1, 3, 6, 12]
LOOKBACK_MONTHS = 14
CACHE_FILE = "improved_factors_cache.csv"
FIELDS = ["close", "volume", "factor"]


def field_matrix(data, field):
    if isinstance(data.columns, pd.MultiIndex):
        if field in data.columns.get_level_values(0):
            return data[field]
        return data.xs(field, axis=1, level=-1)
    if 'code' in data.columns and field in data.columns:
        if isinstance(data.index, pd.DatetimeIndex):
            d = data.copy(); d['_d'] = data.index
            return d.pivot(index='_d', columns='code', values=field)
        date_col = next((c for c in ('date', 'time', 'datetime') if c in data.columns), None)
        if date_col is None:
            raise ValueError("长表找不到日期列，实际列名=%s" % list(data.columns))
        return data.pivot(index=date_col, columns='code', values=field)
    if field in data.columns:
        return data[[field]]
    raise ValueError("未识别的返回结构，实际列名=%s" % list(data.columns))


# ① 交易日历 + 截面日
print("① 取交易日历...")
bench = get_price(BENCH, SAMPLE_START, SAMPLE_END, frequency="daily",
                  fields=["close"], fq="pre", panel=False)
bench_close = bench["close"].sort_index()
bench_close.index = pd.to_datetime(bench_close.index)
cross_dates = bench_close.groupby(bench_close.index.strftime("%Y-%m")).tail(1).index.tolist()
print("   截面日 %d 个" % len(cross_dates))

# ② 断点续跑
done = set()
if os.path.exists(CACHE_FILE):
    done = set(pd.read_csv(CACHE_FILE, usecols=["cross_date"])["cross_date"].astype(str))
    print("② 已完成 %d 个截面" % len(done))
pairs = [(i, cross_dates[i]) for i in range(len(cross_dates) - 1)
         if cross_dates[i].strftime("%Y-%m-%d") not in done]
print("   待处理 %d 个" % len(pairs))

# ③ 逐月计算
rows = []
t0 = time.time()
for k, (i, cross_date) in enumerate(pairs):
    date_str = cross_date.strftime("%Y-%m-%d")
    sec = get_all_securities(types=["stock"], date=date_str)
    codes = list(sec.index)
    if not codes:
        continue
    start_str = (cross_date - pd.DateOffset(months=LOOKBACK_MONTHS)).strftime("%Y-%m-%d")
    raw = get_price(codes, start_str, date_str, frequency="daily",
                    fields=FIELDS, fq="pre", panel=False)
    close = field_matrix(raw, "close").sort_index()
    volume = field_matrix(raw, "volume").sort_index()
    factor = field_matrix(raw, "factor").sort_index()
    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)
    factor.index = pd.to_datetime(factor.index)

    weight = volume / factor            # 换手率等价权重（除复权因子，消除送股跳增）
    r_daily = close.pct_change()
    row = {"cross_date": date_str, "code": list(close.columns)}
    for n in WINDOWS:
        if i >= n:
            w = weight.loc[cross_dates[i - n]:cross_date]
            rr = r_daily.loc[cross_dates[i - n]:cross_date]
            row["wgt_return_%dm" % n] = ((w * rr).sum() / w.sum()).values
            days_back = np.arange(len(w))[::-1]
            decay = np.exp(-days_back / (n * 21 / 4.0))
            num = (w * rr).mul(decay, axis=0).sum()
            den = w.mul(decay, axis=0).sum()
            row["exp_wgt_return_%dm" % n] = (num / den).values
        else:
            n_stock = len(close.columns)
            row["wgt_return_%dm" % n] = np.full(n_stock, np.nan)
            row["exp_wgt_return_%dm" % n] = np.full(n_stock, np.nan)
    rows.append(pd.DataFrame(row))

    if (k + 1) % 12 == 0 or k == len(pairs) - 1:
        if rows:
            chunk = pd.concat(rows, ignore_index=True)
            if os.path.exists(CACHE_FILE):
                chunk = pd.concat([pd.read_csv(CACHE_FILE), chunk], ignore_index=True)
            chunk.to_csv(CACHE_FILE, index=False)
            rows = []
        print("   进度 %d/%d，累计 %.1fs" % (k + 1, len(pairs), time.time() - t0))

# ④ 输出
out = pd.read_csv(CACHE_FILE).sort_values(["cross_date", "code"]).reset_index(drop=True)
out.to_csv("improved_factors.csv", index=False)
print("④ 完成：%d 行，列 %s" % (len(out), list(out.columns)))
print(out.head().to_string())
