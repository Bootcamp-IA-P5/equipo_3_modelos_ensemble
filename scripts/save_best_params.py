import json

best = {
  "n_estimators": 70,
  "max_depth": 8,
  "num_leaves": 38,
  "learning_rate": 0.06062656751877782,
  "subsample": 0.8587250182875306,
  "colsample_bytree": 0.7148781133993373,
  "reg_alpha": 0.3727888495270412,
  "reg_lambda": 0.0004962578734028023
}

# Ajustes de tipo
for k in ("n_estimators", "max_depth", "num_leaves"):
    if k in best:
        best[k] = int(best[k])

payload = {"value": 0.3263464624672446, "params": best}

with open("optuna_best_params.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print("Saved optuna_best_params.json")