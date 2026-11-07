# -*- coding: utf-8 -*-
"""
回测策略：日频量价反转因子 前10%纯多头（月频调仓）

信号：return_1m = 上月末收盘 / 上上月末收盘 - 1（自然月月末对月末，与面板口径一致）
股票池：全A股，剔 ST/PT、停牌
调仓：每月首个交易日（= 华泰"月末核算、下月首个交易日换仓"）
基准：沪深300
手续费：双边约 0.05%（印花税 0.1% 卖出 + 佣金双边 0.025%）

运行：聚宽回测环境，起止日期 2005-04-29 ~ 2016-11-30。
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
    # 股票池：全A股，剔 ST/PT、停牌
    sec = get_all_securities(types=['stock'])
    stocks = [s for s in sec.index
              if 'ST' not in sec.loc[s, 'display_name']
              and 'PT' not in sec.loc[s, 'display_name']
              and not data[s].paused]

    # history 取最近 45 个交易日收盘（覆盖两个月末），df=True 返回 DataFrame(日期×股票)
    h = history(45, '1d', 'close', stocks, df=True, fq='pre')
    if h is None or len(h) < 2:
        return

    # 每月最后一个交易日收盘
    monthly = h.groupby(h.index.strftime('%Y-%m')).last()

    # return_1m = 上月末 / 上上月末 - 1
    rets = {}
    for s in stocks:
        if s in monthly.columns:
            col = monthly[s]
            if len(col) >= 2 and col.iloc[-2] > 0:
                rets[s] = col.iloc[-1] / col.iloc[-2] - 1.0

    # 反转：收益率最低的前 10% 做多头
    rets = sorted(rets.items(), key=lambda x: x[1])
    n = max(1, int(len(rets) * g.top_ratio))
    target = [s for s, _ in rets[:n]]

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
