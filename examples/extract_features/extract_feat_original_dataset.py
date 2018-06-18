from tools import pra_run

pra_run.extract_features(
    emb_import_path=None,
    dataset_path='./benchmarks/NELL186',
    neg_rate=2,
    bern=True,
    feature_extractors=['pra'],
    use_ids=False,
)
