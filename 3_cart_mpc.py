"""Demo 3 - MPC on a mechanical system: bring a cart to a wall without hitting it.

Cart (double integrator):  position p, velocity v, push u (force/mass), |u| <= 1 m/s^2.
Goal: arrive at the wall, at p = 10 m, and stop there.

The wall is a "soft" rule for the planner: going past it is allowed in a plan,
but at a huge price.  So the planner always returns its least-bad plan, and if
that plan goes through the wall, the cart really crashes.

Three controllers, all MPC, all solving a small QP every 0.1 s:
  * short look-ahead  (N = 5,  0.5 s): sees the wall too late to brake
  * short look-ahead + "every plan ends stopped"  (v_N = 0)
  * long look-ahead   (N = 40, 4 s)

Run:  python 3_cart_mpc.py           (the three controllers above)
      python 3_cart_mpc.py knobs     (the two knobs: N = 5, 10, 40, without and
                                      with the terminal rule, side by side)
      add --es for Spanish labels and messages (figures saved as *_es.png)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

DT, UMAX, WALL, TARGET = 0.1, 1.0, 10.0, 10.0
Q_POS, R_U, WALL_PRICE = 1.0, 0.1, 1000.0
ES = "--es" in sys.argv[1:]
SUFFIX = "_es" if ES else ""


def tr(en, es):
    return es if ES else en


def cart_plan(p0, v0, N, dt, umax, wall, target, q_pos, r_u, stop_at_end):
    """One MPC step: returns the first move and the whole plan."""
    p, v, u = cp.Variable(N + 1), cp.Variable(N + 1), cp.Variable(N)
    s = cp.Variable(N, nonneg=True)                            # how far past the wall
    cost = (q_pos * cp.sum_squares(p[1:] - target) + r_u * cp.sum_squares(u)
            + WALL_PRICE * cp.sum(s))
    rules = [p[0] == p0, v[0] == v0,
             p[1:] == p[:-1] + dt * v[:-1] + 0.5 * dt**2 * u,   # the model
             v[1:] == v[:-1] + dt * u,
             cp.abs(u) <= umax,                                 # the motor
             p[1:] <= wall + s]                                 # the wall (soft)
    if stop_at_end:
        rules += [v[N] == 0]                                    # terminal rule
    cp.Problem(cp.Minimize(cost), rules).solve(solver=cp.CLARABEL)
    return u.value[0], dict(u=u.value, p=p.value[1:], v=v.value[1:])


def simulate(N, stop_at_end, t_end=12.0):
    p, v, out = 0.0, 0.0, [(0.0, 0.0, 0.0, 0.0)]
    for k in range(int(t_end / DT)):
        u, _ = cart_plan(p, v, N, DT, UMAX, WALL, TARGET, Q_POS, R_U, stop_at_end)
        p_new, v_new = p + DT * v + 0.5 * DT**2 * u, v + DT * u
        if p_new > WALL + 0.01:                    # 1 cm past the wall: crash
            frac = (WALL - p) / (p_new - p)        # where in the step it hit
            t_hit, v_hit = (k + frac) * DT, v + frac * (v_new - v)
            out.append((t_hit, WALL, v_hit, u))
            return np.array(out), (t_hit, v_hit)
        p, v = p_new, v_new
        out.append(((k + 1) * DT, p, v, u))
    return np.array(out), None


def report(name, tr_, crash):
    """One line per controller: crash, arrival, or where it got to."""
    t, p = tr_[:, 0], tr_[:, 1]
    if crash:
        th, vh = crash
        print(tr(f"{name:30s}: CRASHED into the wall at t = {th:.1f} s, speed {vh:.2f} m/s",
                 f"{name:30s}: CHOCÓ contra la pared en t = {th:.1f} s, a {vh:.2f} m/s"))
        return
    near = np.abs(p - TARGET) < 0.05
    if near.any():
        print(tr(f"{name:30s}: at the wall, stopped, at t = {t[near.argmax()]:.1f} s",
                 f"{name:30s}: en la pared, detenido, en t = {t[near.argmax()]:.1f} s"))
    else:
        print(tr(f"{name:30s}: safe, but only at {p[-1]:.1f} m after {t[-1]:.0f} s",
                 f"{name:30s}: seguro, pero solo a {p[-1]:.1f} m después de {t[-1]:.0f} s"))


def knobs(t_end=15.0):
    """The two knobs side by side: horizon N (colours) and terminal rule (panels)."""
    horizons = [(5, "#C0392B"), (10, "#E8A33D"), (40, "#2E8B57")]
    titles = [tr("no terminal rule", "sin regla terminal"),
              tr("every plan ends stopped ($v_N = 0$)", "cada plan termina detenido ($v_N = 0$)")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, stop, title in zip(axs, [False, True], titles):
        for N, col in horizons:
            name = f"N = {N}" + (tr(" + stop at the end", " + parar al final") if stop else "")
            tr_, crash = simulate(N, stop, t_end)
            report(name, tr_, crash)
            ax.plot(tr_[:, 0], tr_[:, 1], color=col, lw=2.6, label=f"N = {N}  ({N * DT:g} s)")
            if crash:
                th, vh = crash
                ax.plot(th, WALL, "X", color=col, ms=15, mew=1)
                left = N == 5                              # labels on both sides: no crossing arrows
                ax.annotate(tr(f"crash, {vh:.1f} m/s", f"choque, {vh:.1f} m/s"), (th, WALL),
                            xytext=(0.5 if left else th + 1.6, WALL - 1.8), color=col, fontsize=11,
                            arrowprops=dict(arrowstyle="->", color=col, shrinkB=6))
        ax.axhline(WALL, color="k", lw=4)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(tr("time [s]", "tiempo [s]"))
        ax.set_xlim(0, t_end); ax.set_ylim(0, 11.5)
    axs[0].set_ylabel(tr("position [m]", "posición [m]"))
    axs[0].text(0.3, WALL + 0.35, tr("wall", "pared"), fontsize=11)
    axs[1].legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"fig_cart_knobs{SUFFIX}.png", dpi=160)
    plt.show()


def main():
    if "knobs" in sys.argv[1:]:
        knobs()
        return
    cases = [(tr("short look-ahead (N = 5)", "horizonte corto (N = 5)"), 5, False, "#C0392B"),
             (tr("N = 5 + stop at the end", "N = 5 + parar al final"), 5, True, "#E8A33D"),
             (tr("long look-ahead (N = 40)", "horizonte largo (N = 40)"), 40, False, "#2E8B57")]
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for name, N, stop, col in cases:
        tr_, crash = simulate(N, stop)
        t, p, v, u = tr_.T
        ax[0].plot(t, p, color=col, lw=2.5, label=name)
        ax[1].plot(t, v, color=col, lw=2.5)
        ax[2].step(t[1:], u[1:], where="pre", color=col, lw=2)
        if crash:
            th, vh = crash
            ax[0].plot(th, WALL, "X", color=col, ms=16, mew=1)
            ax[0].annotate(tr(f"CRASH at {vh:.1f} m/s\n(saw the wall too late)",
                              f"CHOQUE a {vh:.1f} m/s\n(vio la pared demasiado tarde)"),
                           (th, WALL), xytext=(th + 0.7, WALL - 4.5), color=col,
                           arrowprops=dict(arrowstyle="->", color=col))
        report(name, tr_, crash)
    ax[0].axhline(WALL, color="k", lw=4)
    ax[0].text(11.9, WALL + .3, tr("wall", "pared"), fontsize=11, ha="right")
    ax[0].set_ylabel(tr("position [m]", "posición [m]")); ax[0].set_ylim(0, 11.5)
    ax[0].legend(loc="upper left", bbox_to_anchor=(0, 0.80), fontsize=9)
    ax[1].set_ylabel(tr("speed [m/s]", "velocidad [m/s]"))
    ax[2].set_ylabel(tr("push [m/s²]", "empuje [m/s²]")); ax[2].set_xlabel(tr("time [s]", "tiempo [s]"))
    ax[2].axhline(UMAX, color="grey", ls=":"); ax[2].axhline(-UMAX, color="grey", ls=":")
    fig.tight_layout()
    fig.savefig(f"fig_cart_mpc{SUFFIX}.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
