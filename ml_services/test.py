# test_bcs_explainer.py
from bcs_explainer import explain_bcs
res = explain_bcs("CCO", model_dir="./artifacts_bcs", logS_override= -1.23)
print(res)
