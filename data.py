"""Shared, synthetic data for the Piura demos (hourly, illustrative numbers).

A household with rooftop PV and a battery, under a time-of-use tariff.
Nothing here is fitted to real data: the shapes are chosen to be readable.
"""
import numpy as np

HOURS = np.arange(24)

# --- household load [kW]: small base, morning bump, evening peak -------------
LOAD = (0.4
        + 0.8 * np.exp(-0.5 * ((HOURS - 7.5) / 1.0) ** 2)
        + 2.0 * np.exp(-0.5 * ((HOURS - 20.0) / 1.5) ** 2))

# --- clear-sky PV production [kW] for a 3 kWp roof ----------------------------
PV_PEAK = 3.0
PV_CLEAR = PV_PEAK * np.clip(np.sin(np.pi * (HOURS - 6) / 12), 0, None)

# --- time-of-use prices [$/kWh] ------------------------------------------------
BUY = np.full(24, 0.18)          # daytime
BUY[0:6] = 0.10                  # night, off-peak
BUY[18:23] = 0.32                # evening peak
SELL = np.full(24, 0.05)         # what the grid pays for exported energy

# --- battery -------------------------------------------------------------------
BATT = dict(
    Q=10.0,      # capacity [kWh]
    B=3.0,       # max charge / discharge power [kW]
    eta=0.95,    # one-way efficiency
    wear=0.01,   # small cost per kWh moved, discourages useless cycling [$/kWh]
    hold=0.001,  # tiny cost per kWh kept stored for an hour (self-discharge);
                 # it also breaks ties, so every solver finds the same schedule
)


def repeat_days(profile, days):
    """Tile a 24-hour profile over several days."""
    return np.tile(profile, days)
