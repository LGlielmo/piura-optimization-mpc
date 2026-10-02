# Optimization and Model Predictive Control — three small demos

Companion code for the seminar *Cyber-Physical Systems: optimization and model
predictive control*, given by Luigi Glielmo (Università di Napoli Federico II)
at the opening of the Master in Automation Engineering, Piura, Perú, October 2026.

Each demo is a short Python script built on [CVXPY](https://www.cvxpy.org):
you write down *what you want* (an objective and some constraints) and a
solver finds *how* to get it.

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
- *MPC*: every hour measures the battery, re-plans the next 24 hours, applies
  only the first hour, and repeats.

**3. `3_cart_mpc.py` — MPC on a mechanical system.**
Bring a cart to a wall and stop, with limited braking. With a short
look-ahead the controller goes too fast, sees the wall too late, and crashes;
a rule "be able to stop at the end of the plan" makes it safe but slow; a
longer look-ahead is both safe and fast.

## Things to try

- In `data.py`, change the tariff, the solar size or the battery efficiency.
- In demo 2, make the forecast errors larger, or shorten the MPC look-ahead `H`.
- In demo 3, try `N = 10, 20, 30`: what is the shortest look-ahead that still
  stops safely?

## Credits

The residential-energy example follows the spirit of the one in S. Boyd and
B. Meyers, *Convex Optimization with Smart Grid Examples* (IEEE SmartGridComm
2025). All data here are synthetic and illustrative.

## License

MIT — see [LICENSE](LICENSE). You are free to use and adapt this code, with attribution.
