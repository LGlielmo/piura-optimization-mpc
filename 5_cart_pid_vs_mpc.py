"""Demo 5 - PID or MPC? The cart and the wall of demo 3, with a controller you know.

Cart (double integrator): position p, speed v, push u, |u| <= 1 m/s^2 (motor and brakes).
Goal: reach the wall at p = 10 m as fast as possible, and stop there.
"Stopped at the wall" here means: within 5 cm of it, and slower than 5 cm/s.

A PD controller (a PID with Ki = 0: no friction, no constant force to cancel):
    u = Kp * (10 - p) - Kd * v,     then clipped to [-1, 1] by the motor.
The gains know nothing about the wall or the motor limit.

Three runs, each a question for the room:

    python 5_cart_pid_vs_mpc.py           gentle PD, snappy PD, MPC (N = 40, 4 s ahead)
    python 5_cart_pid_vs_mpc.py tune      search 10 000 pairs of gains (Kp, Kd) in [0, 20]:
                                          which is the fastest PD that does not crash?
    python 5_cart_pid_vs_mpc.py moving    the same controllers, but the cart starts
                                          already moving towards the wall at 2 m/s

The MPC is the one of demo 3 (same model, same cost, same soft wall).
"""
import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt

cart_plan = importlib.import_module("3_cart_mpc").cart_plan

DT, UMAX, WALL = 0.1, 1.0, 10.0
Q_POS, R_U = 1.0, 0.1
GENTLE = (0.5, 1.4)        # Kp, Kd: tuned by hand, cautious
SNAPPY = (1.0, 1.0)        # tuned by hand, "a quick response"
TUNED = (4.8, 8.0)         # what "python 5_cart_pid_vs_mpc.py tune" finds


def pd(kp, kd):
    return lambda p, v: kp * (WALL - p) - kd * v


def mpc(N=40):
    def control(p, v):
        u, _ = cart_plan(p, v, N, DT, UMAX, WALL, WALL, Q_POS, R_U, False)
        return u
    return control


def simulate(control, p0=0.0, v0=0.0, t_end=15.0):
    """Returns the trajectory and what happened: ('crash', t, speed) or ('arrived', t, None)."""
    p, v, out = p0, v0, [(0.0, p0, v0, 0.0)]
    arrived = None
    for k in range(int(round(t_end / DT))):
        u = float(np.clip(control(p, v), -UMAX, UMAX))       # the motor saturates
        p_new, v_new = p + DT * v + 0.5 * DT**2 * u, v + DT * u
        if p_new > WALL + 0.01:                               # 1 cm past the wall: crash
            frac = (WALL - p) / (p_new - p)
            t_hit, v_hit = (k + frac) * DT, v + frac * (v_new - v)
            out.append((t_hit, WALL, v_hit, u))
            return np.array(out), ("crash", t_hit, v_hit)
        p, v = p_new, v_new
        out.append(((k + 1) * DT, p, v, u))
        if arrived is None and abs(p - WALL) < 0.05 and abs(v) < 0.05:
            arrived = (k + 1) * DT
    if arrived is not None:
        return np.array(out), ("arrived", arrived, None)
    return np.array(out), ("short", None, p)


def describe(name, result):
    what, t, x = result
    if what == "crash":
        return f"{name:28s}: CRASH at t = {t:4.1f} s, speed {x:.1f} m/s"
    if what == "arrived":
        return f"{name:28s}: stopped at the wall at t = {t:4.1f} s"
    return f"{name:28s}: safe, but only at {x:.1f} m"


def plot(cases, p0, v0, title, fname):
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for name, control, col in cases:
        tr, res = simulate(control, p0, v0)
        print(describe(name, res))
        t, p, v, u = tr.T
        ax[0].plot(t, p, color=col, lw=2.5, label=name)
        ax[1].plot(t, v, color=col, lw=2.5)
        ax[2].step(t[1:], u[1:], where="pre", color=col, lw=2)
        if res[0] == "crash":
            ax[0].plot(res[1], WALL, "X", color=col, ms=16)
    ax[0].axhline(WALL, color="k", lw=4)
    ax[0].set_ylabel("position [m]"); ax[0].set_ylim(0, 11.5)
    ax[0].legend(loc="lower right", fontsize=9); ax[0].set_title(title)
    ax[1].set_ylabel("speed [m/s]")
    ax[2].set_ylabel("push [m/s²]"); ax[2].set_xlabel("time [s]")
    for s in (1, -1):
        ax[2].axhline(s * UMAX, color="grey", ls=":")
    fig.tight_layout()
    fig.savefig(fname, dpi=160)
    plt.show()


def tune():
    """Try 100 x 100 pairs of gains; keep the fastest one that does not crash.
    Many pairs tie at the fastest time: the first one found is kept."""
    kps = np.linspace(0.2, 20, 100)
    kds = np.linspace(0.2, 20, 100)
    T = np.full((len(kds), len(kps)), np.nan)          # arrival time
    slow = np.zeros_like(T)                             # 1 = safe, but not there in 20 s
    best = None
    for i, kd in enumerate(kds):
        for j, kp in enumerate(kps):
            _, (what, t, _) = simulate(pd(kp, kd), t_end=20.0)
            if what == "arrived":
                T[i, j] = t
                if best is None or t < best[0]:
                    best = (t, kp, kd)
            elif what == "short":
                slow[i, j] = 1
    t, kp, kd = best
    print(f"10 000 simulations. Fastest PD that does not crash: "
          f"Kp = {kp:.1f}, Kd = {kd:.1f}, stopped at the wall in {t:.1f} s")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_facecolor("#C0392B")                         # red background = crash
    ax.pcolormesh(kps, kds, np.where(slow == 1, 1.0, np.nan), cmap="Greys",
                  vmin=0, vmax=2, shading="auto")       # grey = too slow
    im = ax.pcolormesh(kps, kds, T, cmap="viridis_r", shading="auto")
    ax.plot(kp, kd, "*", color="w", ms=18, mec="k")
    fig.colorbar(im, label="time to stop at the wall [s]")
    ax.set_xlabel("Kp"); ax.set_ylabel("Kd")
    ax.set_title("red: crash into the wall    grey: not there after 20 s")
    fig.tight_layout()
    fig.savefig("fig_pd_tuning.png", dpi=160)
    plt.show()


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "tune":
        tune()
    elif arg == "moving":
        plot([("gentle PD", pd(*GENTLE), "#E8A33D"),
              ("tuned PD (from 'tune')", pd(*TUNED), "#C0392B"),
              ("MPC, 4 s ahead", mpc(40), "#2E8B57")],
             0.0, 2.0, "The cart starts at 2 m/s towards the wall", "fig_pd_moving.png")
    else:
        plot([("gentle PD", pd(*GENTLE), "#E8A33D"),
              ("snappy PD", pd(*SNAPPY), "#C0392B"),
              ("MPC, 4 s ahead", mpc(40), "#2E8B57")],
             0.0, 0.0, "From rest", "fig_pd_vs_mpc.png")


if __name__ == "__main__":
    main()
