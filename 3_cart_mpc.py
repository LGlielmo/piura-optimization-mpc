"""Demo 3 - MPC on a mechanical system: bring a cart to a wall without hitting it.

Cart (double integrator):  position p, velocity v, push u (force/mass), |u| <= 1 m/s^2.
Goal: arrive at the wall, at p = 10 m, and stop there.

The wall is a "soft" rule for the planner: going past it is allowed in a plan,
but at a huge price.  So the planner always returns its least-bad plan, and if
that plan goes through the wall, the cart really crashes.

Three controllers, all MPC, all solving a small QP every 0.1 s:
  * short look-ahead  (N = 5,  0.5 s): sees the wall too late to brake
  * short look-ahead + "be able to stop at the end of the plan"  (v_N = 0)
  * long look-ahead   (N = 40, 4 s)

Run:  python 3_cart_mpc.py
"""
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

DT, UMAX, WALL, TARGET = 0.1, 1.0, 10.0, 10.0
Q_POS, R_U, WALL_PRICE = 1.0, 0.1, 1000.0


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


def main():
    cases = [("short look-ahead (N = 5)", 5, False, "#C0392B"),
             ("N = 5 + must be able to stop", 5, True, "#E8A33D"),
             ("long look-ahead (N = 40)", 40, False, "#2E8B57")]
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for name, N, stop, col in cases:
        tr, crash = simulate(N, stop)
        t, p, v, u = tr.T
        ax[0].plot(t, p, color=col, lw=2.5, label=name)
        ax[1].plot(t, v, color=col, lw=2.5)
        ax[2].step(t[1:], u[1:], where="pre", color=col, lw=2)
        if crash:
            th, vh = crash
            ax[0].plot(th, WALL, "X", color=col, ms=16, mew=1)
            ax[0].annotate(f"CRASH at {vh:.1f} m/s\n(saw the wall too late)",
                           (th, WALL), xytext=(th + 0.7, WALL - 4.5), color=col,
                           arrowprops=dict(arrowstyle="->", color=col))
            print(f"{name:30s}: CRASHED into the wall at t = {th:.1f} s, speed {vh:.2f} m/s")
        else:
            near = np.abs(p - TARGET) < 0.05
            if near.any():
                print(f"{name:30s}: at the wall, stopped, at t = {t[near.argmax()]:.1f} s")
            else:
                print(f"{name:30s}: safe, but only at {p[-1]:.1f} m after {t[-1]:.0f} s")
    ax[0].axhline(WALL, color="k", lw=4)
    ax[0].text(11.9, WALL + .3, "wall", fontsize=11, ha="right")
    ax[0].set_ylabel("position [m]"); ax[0].set_ylim(0, 11.5)
    ax[0].legend(loc="upper left", bbox_to_anchor=(0, 0.80), fontsize=9)
    ax[1].set_ylabel("speed [m/s]")
    ax[2].set_ylabel("push [m/s²]"); ax[2].set_xlabel("time [s]")
    ax[2].axhline(UMAX, color="grey", ls=":"); ax[2].axhline(-UMAX, color="grey", ls=":")
    fig.tight_layout()
    fig.savefig("fig_cart_mpc.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
