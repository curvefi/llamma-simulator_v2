# Curve LLAMMA Simulator

## Installation

### Locally using uv

```
pip install uv
uv venv
uv sync
```

### Recommended python version
It's recommended to use pypy to do faster simulations

```
uv python install pypy@3.11
uv venv --python pypy@3.11
uv sync -p pypy@3.11
```

## Running simulations

### Import data

Add pair to simulator/settings.py and import price data

```
python manage.py import_data {pair_name} (i.e. BTCUSDT)
```

### Performing calculations

_PDF will be added with more detailed explanation_

1) Change parameters in python manage.py for `calculate` command

2) Run simulations

```
python manage.py calculate
```
Results automatically will be saved in results folder.

### Position initialization

Each replay anchors its synthetic position to the AMM's effective opening oracle,
after price limiting, rather than the opening market price. `position_shift`
lowers that anchor by the requested fraction. The existing band convention is
preserved: a four-band position occupies bands 1–4 in its newly constructed grid.
This is a liquidation simulation setup, not a reconstruction of the controller's
debt-dependent placement for a new loan.

Oracle and dynamic-fee memory are prepared from the full available history before
selecting a replay window. The opening state is restored before depositing and
valuing the position; its first update is not applied twice. The history starts
with zero fee memory because no earlier observations are available. Supplying
history before the windows of interest allows that memory to warm up.

For the EMA oracle, candle timestamps identify their opens and a close becomes
available only at the end of its candle (60 seconds by default). The first candle
is warm-up and is excluded from replay, with candle/oracle alignment preserved.
Custom oracle implementations must return values available at each candle open.

Raw loss compares final recovery with recovery immediately after initialization,
before trading. Calculator adjustments use the corresponding initial recovery
coefficient. Failed replays raise an error rather than contributing zero loss.
These initialization changes intentionally change results; existing result files
are not recalculated.

The existing `LendingAMM` constructor remains usable; an optional `oracle_state`
restores prepared history. Direct `calculate_loss()` calls without `initial_state`
start at the first supplied oracle with zero fee memory. Supply the matching
state from `oracle_states(...)` to retain earlier history. Ordinary
`Simulator.single_run()` calls prepare and select that state automatically.

Run the regression tests with `python -m unittest discover -s tests -v`.

### Separate scripts

Script ran for every pair is stored in `simulator/pairs` directory to save parameters used in calculations

```
export PYTHONPATH="${PYTHONPATH}:/path/to/your/directory"
python simulator/pairs/btcusd/calculate_a.py
```
