"""Demo 2 - From optimization to MPC: the same battery, three days, wrong forecasts.

Day 1: the forecast is right.
Day 2: forecast sunny, it turns out cloudy.
Day 3: normal forecast, but guests arrive in the evening (load +60%).

Three ways to run the battery:
  * perfect foresight   - knows the future (impossible; the benchmark)
  * open loop           - plans once at midnight of day 1, then follows the plan
  * MPC                 - every hour: measure the battery, the sun and the load,
                          correct the forecast, re-plan the next 24 h,
                          apply only the first hour, repeat

Forecast correction (MPC only): if the sun now gives 35% of the clear-sky value,
expect 35% for the rest of today; the same for the house load.  Tomorrow is
forecast as a normal day.

All three policies may end day 3 with any charge; the difference from the
starting charge is then settled at the night price (same rule for all).

Run:  python 2_battery_mpc.py
"""
import importlib
import numpy as np
import matplotlib.pyplot as plt

from data import LOAD, PV_CLEAR, BUY, SELL, BATT, repeat_days

plan_battery = importlib.import_module("1_battery_day").plan_battery

DAYS, H = 3, 24                       # simulated days, MPC look-ahead (hours)
T = DAYS * 24
Q0 = BATT["Q"] / 2                    # battery half full at the start

# --- what the forecast says ---------------------------------------------------
pv_fc = repeat_days(PV_CLEAR, DAYS)
load_fc = repeat_days(LOAD, DAYS)
# --- what actually happens ----------------------------------------------------
pv_true = pv_fc * np.repeat([1.0, 0.35, 1.0], 24)
load_true = load_fc.copy()
load_true[48 + 17:48 + 23] *= 1.6
buy, sell = repeat_days(BUY, DAYS + 1), repeat_days(SELL, DAYS + 1)   # +1 day for look-ahead


def hour_cost(t, load, pv, ch, dis):
    net = load - pv + ch - dis
    return buy[t] * max(net, 0) - sell[t] * max(-net, 0) + BATT["wear"] * (ch + dis)


def apply(t, q, ch, dis):
    """The real battery: it cannot go below empty or above full."""
    Q, eta = BATT["Q"], BATT["eta"]
    ch = min(ch, (Q - q) / eta)
    dis = min(dis, q * eta)
    return q + eta * ch - dis / eta, ch, dis


LOAD_FC, PV_FC = repeat_days(LOAD, DAYS + 1), repeat_days(PV_CLEAR, DAYS + 1)


def forecast(t):
    """Next H hours as seen at hour t: the current hour is measured, and the rest
    of today is the normal forecast scaled by what is measured now."""
    ld, pv = LOAD_FC[t:t + H].copy(), PV_FC[t:t + H].copy()
    today = min((t // 24 + 1) * 24, t + H) - t          # hours left today, this one included
    ld[:today] *= load_true[t] / LOAD_FC[t]              # e.g. guests: +60% for the rest of today
    if PV_FC[t] > 0.2:                                   # daylight: e.g. cloudy, 35% of clear sky
        pv[:today] *= pv_true[t] / PV_FC[t]
    ld[0], pv[0] = load_true[t], pv_true[t]              # the current hour is measured
    return ld, pv


def run(policy):
    q, qs, cost = Q0, [Q0], 0.0
    plan = None
    if policy == "open loop":                        # one plan, made at t = 0
        plan = plan_battery(load_fc, pv_fc, buy[:T], sell[:T], **BATT, q0=Q0)
    for t in range(T):
        if policy == "perfect foresight" and t == 0:
            plan = plan_battery(load_true, pv_true, buy[:T], sell[:T], **BATT, q0=Q0)
        if policy == "MPC":
            ld, pv = forecast(t)
            s = plan_battery(ld, pv, buy[t:t + H], sell[t:t + H], **BATT, q0=q)
            ch, dis = s["ch"][0], s["dis"][0]        # apply only the first move
        else:
            ch, dis = plan["ch"][t], plan["dis"][t]  # follow the plan
        q, ch, dis = apply(t, q, ch, dis)
        cost += hour_cost(t, load_true[t], pv_true[t], ch, dis)
        qs.append(q)
    # end of day 3, same rule for all: energy taken from the battery is bought back
    # at the night price, energy left over is credited at the same price
    return np.array(qs), cost + buy[T] * (Q0 - q) / BATT["eta"]


def main():
    res = {p: run(p) for p in ("perfect foresight", "open loop", "MPC")}
    for p, (_, c) in res.items():
        print(f"{p:18s}: {c:6.2f} $ over {DAYS} days")

    hrs = np.arange(T)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax[0].step(hrs, pv_fc, where="mid", color="#E8A33D", ls="--", label="PV forecast")
    ax[0].fill_between(hrs, pv_true, step="mid", color="#E8A33D", alpha=.35, label="PV actual")
    ax[0].step(hrs, load_fc, where="mid", color="k", ls="--", lw=1, label="load forecast")
    ax[0].step(hrs, load_true, where="mid", color="k", lw=2, label="load actual")
    ax[0].text(36, 3.6, "cloudy!", ha="center", fontsize=12)
    ax[0].text(61, 3.6, "guests!", ha="center", fontsize=12)
    ax[0].set_ylabel("kW"); ax[0].set_ylim(0, 4.6)
    ax[0].legend(loc="upper left", ncol=2, fontsize=9)
    style = {"perfect foresight": ("#999999", ":"), "open loop": ("#C0392B", "-"),
             "MPC": ("#2E8B57", "-")}
    for p, (qs, c) in res.items():
        col, ls = style[p]
        ax[1].plot(np.arange(T + 1) - .5, qs, color=col, ls=ls, lw=2.5,
                   label=f"{p}  ({c:.2f} $)")
    ax[1].set_ylabel("stored kWh"); ax[1].set_xlabel("hour")
    ax[1].set_ylim(-.3, BATT["Q"] * 1.1); ax[1].legend(loc="upper left", fontsize=9)
    ax[1].set_xticks(range(0, T + 1, 12))
    for a in ax:
        for d in (24, 48):
            a.axvline(d - .5, color="grey", lw=.6)
    fig.tight_layout()
    fig.savefig("fig_battery_mpc.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
