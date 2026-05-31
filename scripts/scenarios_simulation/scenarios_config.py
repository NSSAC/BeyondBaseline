#!/usr/bin/env python3
# scenarios_config.py
import pandas as pd

# ----------------- Global settings (not file paths) -----------------
DATE_FIELD_DEFAULT        = "date"
START_DATE_DEFAULT        = pd.to_datetime("2021-05-31")
MINIMUM_POOL_SIZE_DEFAULT = 50
GROUP_FEATURES            = ["age_group", "smh_race"]

# ----------------- Scenario table (1–8) -----------------
ROLL_W = 4
BATCH_FRAC_DEFAULT = 0.25
BATCH_CAP_DEFAULT  = 1000
MIN_PER_GROUP_DEFAULT = 10
BLEND_ALPHA = 0.5

SCENARIOS = [
# 1) CS–C(LL)
# { "id": 1, "name": "Scenario 1",
#   "target_mode": "linelist_dynamic",
#   "target_linelist_mode": "cumulative",
#   "target_linelist_window": None,
#   "decision_window_weeks": None,
#   "eval_metric": "kl_vs_linelist_cum",
#   "eval_window_weeks": None,
#   "batch_frac": BATCH_FRAC_DEFAULT, "batch_cap": BATCH_CAP_DEFAULT, "min_per_group": MIN_PER_GROUP_DEFAULT,
#   #"pool_mode": "history",
#   "no_replacement": True
# },

# 2) 1S–1(LL)
{ "id": 1, "name": "Scenario 1",
  "target_mode": "linelist_dynamic",
  "target_linelist_mode": "rolling",
  "target_linelist_window": 1,
  "decision_window_weeks": 1,
  "eval_metric": "per_stride_kl",
  "eval_window_weeks": None,
  "batch_frac": BATCH_FRAC_DEFAULT, "batch_cap": BATCH_CAP_DEFAULT, "min_per_group": MIN_PER_GROUP_DEFAULT,
  "pool_mode": "rolling", "pool_window_weeks": 1, "no_replacement": True
},

# 2) 4S–4(LL)
{ "id": 2, "name": "Scenario 2",
  "target_mode": "linelist_dynamic",
  "target_linelist_mode": "rolling",
  "target_linelist_window": ROLL_W,
  "decision_window_weeks": ROLL_W,
  "sampling_mode": "stride",
  "sampling_stride_weeks": ROLL_W,
  "eval_metric": "per_stride_kl",
  "eval_window_weeks": None,
  "batch_frac": BATCH_FRAC_DEFAULT, "batch_cap": BATCH_CAP_DEFAULT, "min_per_group": MIN_PER_GROUP_DEFAULT,
  "pool_mode": "rolling", "pool_window_weeks": 4, "no_replacement": True
},

# # 3) RS–C(LL)
# { "id": 3, "name": "Scenario 3",
#   "target_mode": "linelist_dynamic",
#   "target_linelist_mode": "cumulative",
#   "target_linelist_window": None,
#   "decision_window_weeks": ROLL_W,
#   "eval_metric": "kl_vs_linelist_cum",
#   "eval_window_weeks": None,
#   "batch_frac": 0.18, "batch_cap": BATCH_CAP_DEFAULT, "min_per_group": MIN_PER_GROUP_DEFAULT,
#   "pool_mode": "rolling", "pool_window_weeks": 4, "no_replacement": True
# },

# # 4) CS–C(LL,P)
# { "id": 4, "name": "Scenario 4",
#   "target_type": "blend", "blend_alpha": BLEND_ALPHA,
#   "target_linelist_mode": "cumulative",
#   "target_linelist_window": None,
#   "decision_window_weeks": None,
#   "eval_metric": "mean_kl_cum",
#   "eval_window_weeks": None,
#   "batch_frac": BATCH_FRAC_DEFAULT, "batch_cap": BATCH_CAP_DEFAULT, "min_per_group": MIN_PER_GROUP_DEFAULT,
#   #"pool_mode": "history",
#   "no_replacement": True
# },

# 3) 1S–1(LL,P)
{ "id": 3, "name": "Scenario 3",
  "target_type": "blend", "blend_alpha": BLEND_ALPHA,
  "target_linelist_mode": "rolling",
  "target_linelist_window": 1,
  "decision_window_weeks": 1,
  "eval_metric": "per_stride_kl",
  "eval_window_weeks": None,
  "batch_frac": BATCH_FRAC_DEFAULT, "batch_cap": BATCH_CAP_DEFAULT, "min_per_group": MIN_PER_GROUP_DEFAULT,
  "pool_mode": "rolling", "pool_window_weeks": 1, "no_replacement": True
},

# 4) 4S–4(LL,P)
{ "id": 4, "name": "Scenario 4",
  "target_type": "blend", "blend_alpha": BLEND_ALPHA,
  "target_linelist_mode": "rolling",
  "target_linelist_window": ROLL_W,
  "decision_window_weeks": ROLL_W,
  "sampling_mode": "stride",
  "sampling_stride_weeks": ROLL_W,
  "eval_metric": "per_stride_kl",
  "eval_window_weeks": None,
  "batch_frac": BATCH_FRAC_DEFAULT, "batch_cap": BATCH_CAP_DEFAULT, "min_per_group": MIN_PER_GROUP_DEFAULT,
  "pool_mode": "rolling", "pool_window_weeks": 4, "no_replacement": True
},

# 6) RS–C(LL,P)
# { "id": 6, "name": "Scenario 6",
#   "target_type": "blend", "blend_alpha": BLEND_ALPHA,
#   "target_linelist_mode": "cumulative",
#   "target_linelist_window": None,
#   "decision_window_weeks": ROLL_W,
#   "eval_metric": "mean_kl_cum",
#   "eval_window_weeks": None,
#   "batch_frac": BATCH_FRAC_DEFAULT, "batch_cap": BATCH_CAP_DEFAULT, "min_per_group": MIN_PER_GROUP_DEFAULT,
#   "pool_mode": "rolling", "pool_window_weeks": 4, "no_replacement": True
# },

# 7) CS–P
# { "id": 7, "name": "Scenario 7",
#   "target_mode": "population_static",
#   "decision_window_weeks": None,
#   "eval_metric": "kl_vs_population_cum",
#   "eval_window_weeks": None,
#   "batch_frac": BATCH_FRAC_DEFAULT, "batch_cap": BATCH_CAP_DEFAULT, "min_per_group": MIN_PER_GROUP_DEFAULT,
#   #"pool_mode": "history",
#   "no_replacement": True
# },

# 5) 1S–P
{ "id": 5, "name": "Scenario 5",
  "target_mode": "population_static",
  "decision_window_weeks": 1,
  "eval_metric": "per_stride_kl",
  "eval_window_weeks": None,
  "batch_frac": BATCH_FRAC_DEFAULT, "batch_cap": BATCH_CAP_DEFAULT, "min_per_group": MIN_PER_GROUP_DEFAULT,
  "pool_mode": "rolling", "pool_window_weeks": 1, "no_replacement": True
},

# 6) 4S–P
{ "id": 6, "name": "Scenario 6",
  "target_mode": "population_static",
  "decision_window_weeks": ROLL_W,
  "sampling_mode": "stride",
  "sampling_stride_weeks": ROLL_W,
  "eval_metric": "per_stride_kl",
  "eval_window_weeks": None,
  "batch_frac": BATCH_FRAC_DEFAULT, "batch_cap": BATCH_CAP_DEFAULT, "min_per_group": MIN_PER_GROUP_DEFAULT,
  "pool_mode": "rolling", "pool_window_weeks": 4, "no_replacement": True
},

]
