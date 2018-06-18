from tools import pra_run

pra_run.extract_features(
    emb_import_path='./results/WN11/TransE/1527008113/',
    neg_rate=2,
    bern=True,
    feature_extractors=['pra'],
    cuda_device=1,
    use_ids=False,
    g_hat_info={'knn_k': 5},
    data_to_use='onefold'
)
