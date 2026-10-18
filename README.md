# Optimization and Model Predictive Control — five small demos

Companion code for the seminar *Cyber-Physical Systems: optimization and model
predictive control*, given by Luigi Glielmo (Università di Napoli Federico II)
at the opening of the Master in Automation Engineering, Piura, Perú, October 2026.

Demos 1–3 and 5 are short Python scripts built on [CVXPY](https://www.cvxpy.org):
you write down *what you want* (an objective and some constraints) and a
solver finds *how* to get it. Demo 4 (flight planning for a drone) is a
plain-Python shortest-path search.

## Install and run

Get the code:

    git clone https://github.com/LGlielmo/piura-optimization-mpc
    cd piura-optimization-mpc

**With Anaconda or Miniconda (recommended):** create a separate environment
with everything the demos need, so nothing else on your computer is changed:

    conda env create -f environment.yml
    conda activate piura

**With plain Python:** preferably inside a virtual environment,

    pip install cvxpy matplotlib numpy

Then run, from inside the repository folder (the scripts import `data.py`):

    python 1_battery_day.py

Each script opens a figure and saves it as a PNG. The first run in a new
environment can take a minute while matplotlib builds its font cache.

## The demos

**1. `1_battery_day.py` — optimization.**
A house with rooftop solar and a battery, under a time-of-use tariff. One
linear program decides, hour by hour, when to buy, sell, charge and discharge
so that the daily bill is as small as possible.
Try different battery sizes: `python 1_battery_day.py 5`, `... 10`, `... 20`.
Why does doubling the battery from 10 to 20 kWh barely help?

**2. `2_battery_mpc.py` — from optimization to MPC.**
The same house over three days, but the forecasts are wrong: day 2 is cloudy
instead of sunny, and on day 3 guests arrive in the evening. Compare:
- *perfect foresight*: knows the future (impossible; a benchmark),
- *open loop*: makes one plan at midnight and follows it,
- *MPC*: every hour measures the battery, the sun and the load, corrects the
  forecast (if the sun now gives 35% of a clear day, expect 35% for the rest of
  today; the same for the load), re-plans the next 24 hours, applies only the
  first hour, and repeats.

All three may end day 3 with any charge; the difference is settled at the night
price, the same rule for all. Expected: 3.82 $ / 4.65 $ / 4.42 $. MPC recovers
part of the gap; the rest needed knowing the clouds the night before.

**3. `3_cart_mpc.py` — MPC on a mechanical system.**
Bring a cart to a wall and stop, with limited braking. With a short
look-ahead the controller goes too fast, sees the wall too late, and crashes;
a rule "be able to stop at the end of the plan" makes it safe but slow; a
longer look-ahead is both safe and fast.

**4. `4_field_path.py` — planning a drone flight** (seminar *Drones for
precision agriculture*). A field with trees, a barn and a water tank, as a grid
of 10 m cells. First the shortest flight from the base to a sick plant (A*
search); then a lawn-mower path that photographs the whole field, going around
the obstacles. Is one battery enough? Only numpy and matplotlib are needed.
Try a bigger field: `python 4_field_path.py 40 25`.
To *watch* how the shortest path is found, run `python 4_field_path.py --explore`:
Dijkstra (left) spreads from the base like a ripple; A* (right) adds to each
square the straight-line distance still to go, and leans towards the target.
Same path length, far fewer squares explored.
Add `--es` for Spanish titles, labels and messages (`python 4_field_path.py --es`,
`python 4_field_path.py --explore --es`); figures are then saved as `*_es.png`.

**5. `5_cart_pid_vs_mpc.py` — PID or MPC?** (seminar *From PID to Model
Predictive Control*). The cart of demo 3, now with a PD controller whose push the
motor clips to ±1 m/s². A gentle tuning is safe but slow; a snappy one crashes.
`python 5_cart_pid_vs_mpc.py tune` tries 10 000 pairs of gains and finds a PD as
fast as MPC; `python 5_cart_pid_vs_mpc.py moving` starts the cart at 2 m/s
towards the wall: the tuned PD crashes, the MPC (unchanged) does not.
Needs `3_cart_mpc.py` in the same folder.

## Things to try

- In `data.py`, change the tariff, the solar size or the battery efficiency.
- In demo 2, make the forecast errors larger, or shorten the MPC look-ahead `H`.
- In demo 3, try `N = 10, 20, 30`: what is the shortest look-ahead that still
  stops safely?
- In demo 5, try other starting speeds or a wall at 5 m: which controllers
  still arrive safely, and which need to be re-tuned?
- In demo 4, change `SPEED` or `BATTERY_MIN`: how fast must the drone fly to
  cover the field on one battery? (Faster also means blurrier pictures.)

## Credits

The residential-energy example follows the spirit of the one in S. Boyd and
B. Meyers, *Convex Optimization with Smart Grid Examples* (IEEE SmartGridComm
2025). All data here are synthetic and illustrative.

## License

MIT — see [LICENSE](LICENSE). You are free to use and adapt this code, with attribution.
