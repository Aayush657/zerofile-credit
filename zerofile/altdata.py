"""Simulated utility, mobile and income signals, modelled on what the official HI-04 brief asks for.

No public dataset links these signals to repayment, so their strength is an explicit assumption.
They are drawn using the real repayment outcome, so nothing built on them is a measured result:
it answers "how much would this data be worth IF it were this strong?", and must be labelled that way.
"""

import numpy as np

from zerofile.features import ALT_SIGNALS

# Gap between defaulters and repayers in the shared habit, in standard deviations.
SCENARIOS = {"Weak": 0.25, "Moderate": 0.5, "Strong": 0.77}
CENTRES = {"bills_on_time": 1.4, "topup_regularity": 0.6, "income_regularity": 0.2}


def simulate(default: np.ndarray, strength: float, seed: int = 7) -> dict:
    """Each signal shares one payment habit plus its own noise, so the three overlap the way real records would."""
    rng = np.random.default_rng(seed)
    habit = rng.normal(size=len(default)) - strength * default
    return {name: 1 / (1 + np.exp(-(CENTRES[name] + habit + rng.normal(size=len(default))))) for name in ALT_SIGNALS}
