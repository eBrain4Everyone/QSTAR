#!/usr/bin/env python3
"""Idempotently require FlashJT minimum history at forecast time (epoch 3).

This prevents a forecast trained with fewer than min_history_runs from later
becoming eligible merely because more histories appeared by epoch 6.
"""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "qstar_new" / "common.py"
s = p.read_text()

old = '''            if epoch == 3 and self.enable_flashjt and assisted:\n                count = self.flashjt.fit_from_run_root(self.run_root)\n                if self.flashjt.model is not None:\n                    early = {e: actual_epoch_metrics[e] for e in (1, 2, 3)}\n                    forecast = self.flashjt.predict(early, self.metadata)\n                    atomic_json_dump({\n                        "seed": self.seed,\n                        "history_count": count,\n                        "metadata": self.metadata,\n                        "forecast": forecast,\n                    }, self.forecast_path)\n                    print(f"FlashJT forecast saved (history={count})", flush=True)\n                else:\n                    atomic_json_dump({\n                        "seed": self.seed,\n                        "history_count": count,\n                        "metadata": self.metadata,\n                        "forecast": None,\n                        "reason": "no_fitted_history",\n                    }, self.forecast_path)\n'''

new = '''            if epoch == 3 and self.enable_flashjt and assisted:\n                count = self.flashjt.fit_from_run_root(self.run_root)\n                enough_history = count >= self.flashjt.min_history_runs\n                if self.flashjt.model is not None and enough_history:\n                    early = {e: actual_epoch_metrics[e] for e in (1, 2, 3)}\n                    forecast = self.flashjt.predict(early, self.metadata)\n                    atomic_json_dump({\n                        "seed": self.seed,\n                        "history_count": count,\n                        "minimum_history_runs": self.flashjt.min_history_runs,\n                        "metadata": self.metadata,\n                        "forecast": forecast,\n                    }, self.forecast_path)\n                    print(f"FlashJT forecast saved (history={count})", flush=True)\n                else:\n                    reason = (\n                        f"insufficient_history_at_forecast:{count}<{self.flashjt.min_history_runs}"\n                        if self.flashjt.model is not None and not enough_history\n                        else "no_fitted_history"\n                    )\n                    forecast = None\n                    atomic_json_dump({\n                        "seed": self.seed,\n                        "history_count": count,\n                        "minimum_history_runs": self.flashjt.min_history_runs,\n                        "metadata": self.metadata,\n                        "forecast": None,\n                        "reason": reason,\n                    }, self.forecast_path)\n                    print(f"FlashJT forecast skipped: {reason}", flush=True)\n'''

if new in s:
    print("FlashJT forecast-time history gate already installed.")
elif old in s:
    p.write_text(s.replace(old, new))
    print(f"Patched {p}")
else:
    raise SystemExit("Expected FlashJT epoch-3 block not found; no changes made.")
