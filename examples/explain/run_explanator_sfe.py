import os
import pandas as pd
import numpy as np
from explain.pipeline import pipeline, process_overall_metrics_wo_emb
from explain.helpers import get_dirs
from tools import train_test


emb_import_paths = [
    "/home/arthurcgusmao/Projects/xkbc/algorithms/OpenKE/benchmarks/FB13/",
    "/home/arthurcgusmao/Projects/xkbc/algorithms/OpenKE/benchmarks/NELL186/",
]
overalls_output_path = "~/WHI_overalls_sfe.tsv"

# Call special functions when running SFE for a dataset (and not to explain an Embedding Model)
for path in emb_import_paths:
    pipeline(path, adapt_run_sfe_wo_emb=True)

metrics_dicts = process_overall_metrics_wo_emb(emb_import_paths)

pd.DataFrame(metrics_dicts).to_csv(overalls_output_path, sep='\t')
