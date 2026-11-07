# -*- coding: utf-8 -*-
"""
01_data_pipeline / step1_build_monthly_panel.py
================================================
全量月频面板构建（华泰口径）。

产出 monthly_panel.csv：每行 = 一个「股票 × 截面日」。
列：cross_date, code, return_1m, return_3m, return_6m, return_12m,
    fwd_excess_return, market_cap, circulating_market_cap, industry_sw_l1

运行：聚宽研究环境。按月循环、逐月缓存，中断重跑自动续跑（断点续跑）。
口径见 docs/factor_spec.md。
"""

import pandas as pd
import numpy as np
import os
import time

# ==================== 参数 ====================
SAMPLE_START = "2005-01-01"   # 取数起点（提前到1月，让 2005-04 起有 1/3 个月历史算 return_1m/3m）
SAMPLE_END = "2016-12-31"     # 取数终点（多取一个月，给 2016-11 的下月收益当分母）
BENCH = "000300.XSHG"
WINDOWS = [1, 3, 6, 12]        # return_Nm 的 N（月）
LOOKBACK_MONTHS = 14           # 取数回看 14 个自然月，覆盖 return_12m（12个月）+ 缓冲
CACHE_FILE = "monthly_panel_cache.csv"
FIELDS = ["close", "paused"]   # 面板当前只用到 close 和 paused；volume 待 wgt_return 因子再加


def field_matrix(data, field):
    """把 get_price 多标的返回转成 行=日期、列=代码 的宽表。"""
    if isinstance(data.columns, pd.MultiIndex):
        if field in data.columns.get_level_values(0):
            return data[field]                        # (field, code)
        return data.xs(field, axis=1, level=-1)        # (code, field)
    if 'code' in data.columns and field in data.columns:
        if isinstance(data.index, pd.DatetimeIndex):
            d = data.copy()
            d['_d'] = data.index
            return d.pivot(index='_d', columns='code', values=field)
        date_col = next((c for c in ('date', 'time', 'datetime') if c in data.columns), None)
        if date_col is None:
            raise ValueError("长表找不到日期列，实际列名=%s" % list(data.columns))
        return data.pivot(index=date_col, columns='code', values=field)
    if field in data.columns:
        return data[[field]]
    raise ValueError("未识别的返回结构，实际列名=%s" % list(data.columns))


def get_industry_map(codes, date_str):
    """批量取申万一级行业，返回 {code: sw_l1_name}。先试列表，失败则逐股。"""
    m = {}
    try:
        res = get_industry(list(codes), date_str)
        for c, v in res.items():
            m[c] = v.get("sw_l1", {}).get("industry_name")
        return m
    except Exception:
        for c in codes:
            try:
                m[c] = get_industry(c, date_str).get("sw_l1", {}).get("industry_name")
            except Exception:
                m[c] = np.nan
        return m


# ==================== ① 交易日历 + 截面日 ====================
print("① 取交易日历（沪深300日线）...")
bench = get_price(BENCH, SAMPLE_START, SAMPLE_END, frequency="daily",
                  fields=["close"], fq="pre", panel=False)
bench_close = bench["close"].sort_index()
bench_close.index = pd.to_datetime(bench_close.index)

# 截面日 = 每月最后交易日（Timestamp 列表）
cross_dates = bench_close.groupby(bench_close.index.strftime("%Y-%m")).tail(1).index.tolist()
print("   截面日共 %d 个，首=%s 末=%s" % (len(cross_dates), cross_dates[0], cross_dates[-1]))

# 基准月收益（相邻截面日）→ 超额收益标签
bench_ret = bench_close.reindex(cross_dates).pct_change()

# ==================== ② 断点续跑 ====================
done = set()
if os.path.exists(CACHE_FILE):
    done = set(pd.read_csv(CACHE_FILE, usecols=["cross_date"])["cross_date"].astype(str))
    print("② 已有缓存，已完成 %d 个截面" % len(done))

# 相邻 (截面日, 下月截面日) 对；剔除最后一个月（无下月收益）和已完成月
pairs = [(i, cross_dates[i], cross_dates[i + 1]) for i in range(len(cross_dates) - 1)
         if cross_dates[i].strftime("%Y-%m-%d") not in done]
print("   待处理截面 %d 个" % len(pairs))

# ==================== ③ 逐月截面 ====================
rows = []
t_start = time.time()

for k, (i, cross_date, next_cross) in enumerate(pairs):
    date_str = cross_date.strftime("%Y-%m-%d")
    next_str = next_cross.strftime("%Y-%m-%d")

    # 3.1 股票池：全A股，剔 ST/PT
    sec = get_all_securities(types=["stock"], date=date_str)
    codes = list(sec.loc[~sec["display_name"].str.contains("ST|PT", na=False)].index)
    if not codes:
        continue

    # 3.2 取日线（回看 LOOKBACK_MONTHS 个月 + 下月）
    start_str = (cross_date - pd.DateOffset(months=LOOKBACK_MONTHS)).strftime("%Y-%m-%d")
    raw = get_price(codes, start_str, next_str, frequency="daily",
                    fields=FIELDS, fq="pre", panel=False)
    close = field_matrix(raw, "close").sort_index()
    close.index = pd.to_datetime(close.index)
    paused = field_matrix(raw, "paused").sort_index()
    paused.index = pd.to_datetime(paused.index)

    # 3.3 剔下期停牌（截面日到下月截面日之间停牌过的股票）
    susp = paused.loc[cross_date:next_cross].any()
    valid = [c for c in codes if c in close.columns and not bool(susp.get(c, False))]
    close = close[valid]

    # 3.4 算 return_Nm 因子（华泰口径：最近 N 个自然月收益 = 本月末 / N 个月前月末 - 1）
    factor = {}
    for n in WINDOWS:
        if i >= n:
            factor["return_%dm" % n] = close.loc[cross_date] / close.loc[cross_dates[i - n]] - 1.0
        else:
            factor["return_%dm" % n] = pd.Series(np.nan, index=close.columns)

    # 3.5 下月超额收益（个股下月收益 - 沪深300下月收益）
    fwd_excess = close.loc[next_cross] / close.loc[cross_date] - 1.0 - bench_ret.loc[next_cross]

    # 3.6 市值 + 行业（查询必须显式带 valuation.code，否则结果里没有股票代码，无法对应）
    mc = pd.Series(np.nan, index=valid)
    cmc = pd.Series(np.nan, index=valid)
    for c0 in range(0, len(valid), 300):
        batch = valid[c0:c0 + 300]
        try:
            sub = get_fundamentals(
                query(valuation.code, valuation.market_cap, valuation.circulating_market_cap)
                .filter(valuation.code.in_(batch)),
                date=date_str)
            if len(sub):
                sub = sub.set_index('code')   # 显式查了 code，现在是列，转成 index 后按 code 对齐
                mc.loc[batch] = sub['market_cap'].reindex(batch).values
                cmc.loc[batch] = sub['circulating_market_cap'].reindex(batch).values
        except Exception as e:
            print("   [市值失败] %s: %s" % (date_str, str(e)[:60]))

    ind_map = get_industry_map(valid, date_str)

    # 3.7 组装本截面
    row = {"cross_date": date_str, "code": valid}
    for name, s in factor.items():
        row[name] = s.reindex(valid).values
    row["fwd_excess_return"] = fwd_excess.reindex(valid).values
    row["market_cap"] = mc.reindex(valid).values
    row["circulating_market_cap"] = cmc.reindex(valid).values
    row["industry_sw_l1"] = [ind_map.get(c, np.nan) for c in valid]
    rows.append(pd.DataFrame(row))

    # 3.8 每 12 个月写一次缓存（断点续跑 + 防丢）
    if (k + 1) % 12 == 0 or k == len(pairs) - 1:
        if rows:
            chunk = pd.concat(rows, ignore_index=True)
            if os.path.exists(CACHE_FILE):
                chunk = pd.concat([pd.read_csv(CACHE_FILE), chunk], ignore_index=True)
            chunk.to_csv(CACHE_FILE, index=False)
            rows = []
        print("   进度 %d/%d，累计 %.1fs" % (k + 1, len(pairs), time.time() - t_start))

# ==================== ④ 汇总输出 ====================
panel = pd.read_csv(CACHE_FILE).sort_values(["cross_date", "code"]).reset_index(drop=True)
print("\n④ 面板完成：%d 行，%d 个截面，%d 只股票" %
      (len(panel), panel["cross_date"].nunique(), panel["code"].nunique()))
print("   列：", list(panel.columns))
print(panel.head().to_string())
panel.to_csv("monthly_panel.csv", index=False)
print("   已写出 monthly_panel.csv")
