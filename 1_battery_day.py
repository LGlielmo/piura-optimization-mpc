"""Demo 1 - Optimization: schedule a home battery for one day.

Say what you want (lowest bill), state the rules (physics and limits),
and let the solver find the schedule.

Run:  python 1_battery_day.py            (default battery, 10 kWh)
      python 1_battery_day.py 0          (no battery: the baseline)
      python 1_battery_day.py 20         (a bigger battery)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from data import HOURS, LOAD, PV_CLEAR, BUY, SELL, BATT


def plan_battery(load, pv, buy, sell, Q, B, eta, wear, hold, q0=None, q_end=None):
    """Minimum-cost battery schedule. Returns a dict of hourly arrays."""
    T = len(load)
    imp, exp = cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True)  # grid in / out
    ch, dis = cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True)   # battery in / out
    q = cp.Variable(T + 1)                                               # stored energy

    cost = buy @ imp - sell @ exp + wear * cp.sum(ch + dis) + hold * cp.sum(q[1:])
    rules = [pv + imp + dis == load + ch + exp,          # power balance, every hour
             q[1:] == q[:-1] + eta * ch - dis / eta,     # battery dynamics
             0 <= q, q <= Q, ch <= B, dis <= B]          # limits
    rules += [q[0] == q[-1]] if q0 is None else [q[0] == q0]
    if q_end is not None:
        rules += [q[-1] >= q_end]

    cp.Problem(cp.Minimize(cost), rules).solve()
    return dict(imp=imp.value, exp=exp.value, ch=ch.value, dis=dis.value,
                q=q.value, cost=cost.value)


def bill_without_battery(load, pv, buy, sell):
    net = load - pv
    return buy @ np.maximum(net, 0) - sell @ np.maximum(-net, 0)


def main():
    Q = float(sys.argv[1]) if len(sys.argv) > 1 else BATT["Q"]
    p = BATT | {"Q": Q}
    s = plan_battery(LOAD, PV_CLEAR, BUY, SELL, **p)
    base = bill_without_battery(LOAD, PV_CLEAR, BUY, SELL)
    print(f"daily bill without battery: {base:6.2f} $")
    print(f"daily bill with {Q:4.0f} kWh battery: {s['cost']:6.2f} $"
          f"   (saving {base - s['cost']:.2f} $/day)")

    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    ax[0].fill_between(HOURS, PV_CLEAR, step="mid", alpha=.35, color="#E8A33D", label="PV")
    ax[0].step(HOURS, LOAD, where="mid", color="k", lw=2, label="load")
    ax[0].set_ylabel("kW"); ax[0].legend(loc="upper left"); ax[0].set_title("The house")

    ax[1].step(HOURS, s["imp"] - s["exp"], where="mid", color="#3B6FB6", lw=2,
               label="grid (+ buy, − sell)")
    ax[1].step(HOURS, s["dis"] - s["ch"], where="mid", color="#C0392B", lw=2,
               label="battery (+ discharge, − charge)")
    ax[1].axhline(0, color="grey", lw=.5); ax[1].set_ylabel("kW")
    ax[1].legend(loc="lower left", fontsize=9); ax[1].set_title("What the optimizer decides")

    ax2 = ax[2].twinx()
    ax[2].plot(np.arange(25) - .5, s["q"], color="#2E8B57", lw=2.5)
    ax[2].set_ylim(-.3, max(Q, 1) * 1.08); ax[2].set_ylabel("stored kWh", color="#2E8B57")
    ax2.step(HOURS, BUY, where="mid", color="grey", ls="--", label="buy price")
    ax2.set_ylabel("$/kWh", color="grey"); ax2.set_ylim(0, .4)
    ax[2].set_title("Battery charge, against the price"); ax[2].set_xlabel("hour of day")
    ax[2].set_xticks(range(0, 25, 3))
    fig.tight_layout()
    fig.savefig(f"fig_battery_day_Q{Q:g}.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
