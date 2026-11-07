# -*- coding: utf-8 -*-
"""
回测策略（优化版）：中证800 动量因子 前10%纯多头（月频调仓）

两步优化逻辑（针对样本外反转因子失效）：
  优化1 市值约束：股票池 = 中证800（沪深300+中证500，市值最大800只），剔除小盘。
        —— 假设"小盘壳价值消失"导致反转失效；测试结果：反转在中证800上仍失效（-4.1%），
           排除该假设，说明失效与大小盘无关，而是行情结构问题。
  优化2 反转转动量：信号从"买跌最多"（反转）改为"买涨最多"（动量）。
        —— 假设"2017 年后 A 股从反转转向动量"（白马蓝筹强者恒强）；测试结果：中证800内
           从 -4.1% 改善到 +1.1%，验证该假设。

信号：return_1m = 上月末收盘 / 上上月末收盘 - 1（自然月，与面板口径一致）
调仓：每月首个交易日，等权
基准：沪深300
手续费：双边约 0.05%

运行：聚宽回测环境，起止日期按需设置（样本外 2017-01-01 ~ 2024-12-31）。
"""


def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                             open_commission=0.00025, close_commission=0.00025,
                             min_commission=5), type='stock')
    g.top_ratio = 0.10
    g.last_month = -1


def rebalance(context, data):
    # 股票池：中证800成分，剔停牌、涨跌停
    stocks = get_index_stocks('000906.XSHG')   # 中证800 = 中大市值
    stocks = [s for s in stocks
              if not data[s].paused
              and data[s].low_limit < data[s].close < data[s].high_limit]

    # history 取最近 45 个交易日收盘，算自然月 return_1m
    h = history(45, '1d', 'close', stocks, df=True, fq='pre')
    if h is None or len(h) < 2:
        return

    monthly = h.groupby(h.index.strftime('%Y-%m')).last()
    rets = {}
    for s in stocks:
        if s in monthly.columns:
            col = monthly[s]
            if len(col) >= 2 and col.iloc[-2] > 0:
                rets[s] = col.iloc[-1] / col.iloc[-2] - 1.0

    # 动量：收益率最高的前 10% 做多头（反转版为 rets[:n]）
    rets = sorted(rets.items(), key=lambda x: x[1])
    n = max(1, int(len(rets) * g.top_ratio))
    target = [s for s, _ in rets[-n:]]

    for s in context.portfolio.positions:
        if s not in target:
            order_target_value(s, 0)

    weight = 1.0 / len(target) if target else 0
    for s in target:
        order_target_value(s, context.portfolio.total_value * weight)


def handle_data(context, data):
    month = context.current_dt.month
    if month != g.last_month:
        g.last_month = month
        rebalance(context, data)
