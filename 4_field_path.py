"""Demo 4 - Planning a drone flight over a field.

The field is a grid of 10 m x 10 m cells.  Some cells are blocked: a row of
tall trees, a barn, a water tank.  The drone starts from its base (bottom left).

Two questions, both optimization problems:
  1. A farmer spots a sick plant.  What is the SHORTEST flight from the base to
     that plant, going around the obstacles?  -> A* search on the grid.
  2. We want a picture of the WHOLE field.  The drone flies back and forth like
     a lawn mower; when an obstacle is in the way, it uses A* to get around it.
     Is the battery enough?

Moves: to the 8 neighbouring cells (straight = 10 m, diagonal = 14.1 m).

Run:  python 4_field_path.py            (default field, 30 x 20 cells)
      python 4_field_path.py 40 25      (a bigger field)
      python 4_field_path.py --explore  (watch how the shortest path is found:
                                         Dijkstra vs A*, side by side)
"""
import sys
import heapq
import numpy as np
import matplotlib.pyplot as plt

CELL = 10.0            # m, side of one cell
SPEED = 5.0            # m/s, cruise speed while taking pictures
BATTERY_MIN = 20.0     # minutes of flight on one battery
ANIM_MS = 150          # --explore: milliseconds per frame (bigger = slower animation)

MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def make_field(nx, ny):
    """True = free cell, False = obstacle."""
    free = np.ones((nx, ny), dtype=bool)
    free[int(0.35 * nx), int(0.15 * ny):int(0.80 * ny)] = False          # row of trees
    free[int(0.60 * nx):int(0.75 * nx), int(0.55 * ny):int(0.80 * ny)] = False  # barn
    cx, cy, r = int(0.75 * nx), int(0.25 * ny), max(2, nx // 12)      # water tank
    for i in range(nx):
        for j in range(ny):
            if (i - cx) ** 2 + (j - cy) ** 2 <= r ** 2:
                free[i, j] = False
    return free


def astar(free, start, goal, guess=True, explored=None):
    """Shortest path on the grid.  Returns the list of cells, start to goal.

    Every square gets a score = metres already flown + (with guess=True) metres
    still to go in a straight line.  Always continue from the lowest score.
      guess=True  -> A*:       the search leans towards the goal
      guess=False -> Dijkstra: the search spreads like a ripple from the start
    If a list is passed as `explored`, the squares are appended to it in the
    order they are explored, with their distance from the start.
    """
    nx, ny = free.shape

    def h(c):                       # straight-line distance: never overestimates
        return np.hypot(c[0] - goal[0], c[1] - goal[1]) if guess else 0.0

    best = {start: 0.0}
    came = {}
    done = set()
    queue = [(h(start), start)]
    while queue:
        _, c = heapq.heappop(queue)
        if c in done:
            continue                # already explored by a shorter way
        done.add(c)
        if explored is not None:
            explored.append((c, best[c]))
        if c == goal:
            path = [c]
            while c in came:
                c = came[c]
                path.append(c)
            return path[::-1]
        for dx, dy in MOVES:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or not free[n]:
                continue
            if dx and dy and not (free[c[0] + dx, c[1]] and free[c[0], c[1] + dy]):
                continue            # do not cut the corner of an obstacle
            g = best[c] + np.hypot(dx, dy)
            if g < best.get(n, np.inf):
                best[n], came[n] = g, c
                heapq.heappush(queue, (g + h(n), n))
    return None


def length_m(path):
    p = np.array(path, dtype=float)
    return CELL * np.sum(np.hypot(*np.diff(p, axis=0).T))


def coverage(free, start):
    """Lawn-mower: sweep column by column, up and down; A* to get around obstacles."""
    nx, ny = free.shape
    order = []
    for i in range(nx):
        col = [(i, j) for j in range(ny) if free[i, j]]
        order += col if i % 2 == 0 else col[::-1]
    path = [start]
    for c in order:
        if c == path[-1]:
            continue
        last = path[-1]
        if max(abs(c[0] - last[0]), abs(c[1] - last[1])) == 1:
            path.append(c)          # next cell is a neighbour: just go
        else:
            path += astar(free, last, c)[1:]   # jump: shortest way around
    path += astar(free, path[-1], start)[1:]   # fly home
    return path


def draw_field(ax, free, base, goal=None):
    nx, ny = free.shape
    ext = (-0.5, nx - 0.5, -0.5, ny - 0.5)
    ax.imshow(~free.T, origin="lower", cmap="Greens_r", alpha=0.35, extent=ext)
    ax.imshow(np.ma.masked_where(free.T, free.T), origin="lower", cmap="Greys",
              vmin=-1, vmax=0, extent=ext)
    ax.plot(*base, "s", color="#0d1b2a", ms=11)
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    ax.annotate("base", base, xytext=(8, 8), textcoords="offset points", fontsize=11,
                bbox=box, zorder=6)
    if goal is not None:
        ax.plot(*goal, "*", color="#c0392b", ms=20, zorder=5)
        ax.annotate("sick plant", goal, xytext=(-95, -6), textcoords="offset points",
                    fontsize=11, color="#c0392b", bbox=box, zorder=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")


def explore(free, base, goal, gif=False):
    """Dijkstra and A* side by side: the explored squares appear one by one,
    coloured by their distance from the base; then the path is drawn."""
    from matplotlib.animation import FuncAnimation
    nx, ny = free.shape
    runs = []
    for guess, name in [(False, "Dijkstra"), (True, "A*")]:
        order = []
        path = astar(free, base, goal, guess=guess, explored=order)
        runs.append((name, order, np.array(path)))
        print(f"{name:8s}: explored {len(order):4d} squares, path {length_m(path):.0f} m")

    dmax = max(d for _, order, _ in runs for _, d in order)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    grids, lines, titles = [], [], []
    for ax, (name, order, path) in zip(axs, runs):
        draw_field(ax, free, base, goal)
        g = np.full((ny, nx), np.nan)
        im = ax.imshow(g, origin="lower", cmap="YlOrBr", vmin=0, vmax=dmax, alpha=0.85,
                       extent=(-0.5, nx - 0.5, -0.5, ny - 0.5))
        line, = ax.plot([], [], color="#0d1b2a", lw=2.6)
        grids.append((g, im)); lines.append(line)
        titles.append(ax.set_title(name, fontsize=13))

    step = max(1, len(runs[0][1]) // 120)          # about 120 frames
    nframes = len(runs[0][1]) // step + 25         # + a pause at the end

    def update(k):
        for (name, order, path), (g, im), line, t in zip(runs, grids, lines, titles):
            m = min(len(order), (k + 1) * step)
            g[:] = np.nan                                  # start from an empty field
            for c, d in order[:m]:
                g[c[1], c[0]] = d
            im.set_data(g)
            if m == len(order):
                line.set_data(path[:, 0], path[:, 1])
            else:
                line.set_data([], [])
            t.set_text(f"{name}: {m} squares explored")
        return []

    anim = FuncAnimation(fig, update, frames=nframes, interval=ANIM_MS, repeat=False)
    fig.tight_layout()
    if gif:
        anim.save("fig_explore.gif", writer="pillow", fps=25)
    plt.show()                                     # the animation runs here
    update(nframes - 1)                            # final picture, saved as PNG
    fig.savefig("fig_explore.png", dpi=160)


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    nums = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
    nx = nums[0] if len(nums) > 0 else 30
    ny = nums[1] if len(nums) > 1 else 20
    free = make_field(nx, ny)
    base = (0, 0)
    sick = (nx - 2, ny - 2)
    if "--explore" in flags:
        explore(free, base, sick, gif="--gif" in flags)
        return

    short = astar(free, base, sick)
    cover = coverage(free, base)

    L1, L2 = length_m(short), length_m(cover)
    t2 = L2 / SPEED / 60
    area = free.sum() * CELL**2 / 1e4
    print(f"Field: {nx*CELL:.0f} m x {ny*CELL:.0f} m, {area:.1f} ha to photograph")
    print(f"1. Shortest flight base -> sick plant: {L1:.0f} m "
          f"(straight line would be {CELL*np.hypot(*np.subtract(sick, base)):.0f} m)")
    print(f"2. Lawn-mower over the whole field: {L2/1000:.2f} km, "
          f"{t2:.0f} min at {SPEED:.0f} m/s")
    print(f"   Battery: {BATTERY_MIN:.0f} min -> "
          + ("enough." if t2 <= BATTERY_MIN else
             f"NOT enough: {int(np.ceil(t2 / BATTERY_MIN))} batteries needed."))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    titles = [f"Shortest flight: {L1:.0f} m",
              f"Whole field: {L2/1000:.1f} km, {t2:.0f} min "
              f"(battery {BATTERY_MIN:.0f} min)"]
    for ax, path, title, col in zip(axs, [short, cover], titles, ["#b45a0a", "#1a7a8a"]):
        draw_field(ax, free, base, sick if path is short else None)
        p = np.array(path)
        ax.plot(p[:, 0], p[:, 1], color=col, lw=2.2 if path is short else 1.4)
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig("fig_field_path.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
