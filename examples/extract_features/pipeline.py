from tools import pra_run

def pipeline(emb_import_paths):
    for emb_import_path in emb_import_paths:
        pra_run.extract_features(
            emb_import_path=emb_import_path,
            neg_rate=2,
            bern=True,
            feature_extractors=['pra'],
            cuda_device=0,
            use_ids=False,
            g_hat_info=None,
            data_to_use='onefold'
        )
        pra_run.extract_features(
            emb_import_path=emb_import_path,
            neg_rate=2,
            bern=True,
            feature_extractors=['pra'],
            cuda_device=0,
            use_ids=False,
            g_hat_info={'knn_k': 3},
            data_to_use='onefold'
        )
        pra_run.extract_features(
            emb_import_path=emb_import_path,
            neg_rate=2,
            bern=True,
            feature_extractors=['pra'],
            cuda_device=0,
            use_ids=False,
            g_hat_info={'knn_k': 5},
            data_to_use='onefold'
        )

emb_import_paths = [
#    '/home/arthurcgusmao/Projects/xkbc/algorithms/OpenKE/results/WN11/TransE/1527008113',
    '/home/arthurcgusmao/Projects/xkbc/algorithms/OpenKE/results/FB13/TransE/1527033688',
    '/home/arthurcgusmao/Projects/xkbc/algorithms/OpenKE/results/NELL186/TransE/1526711822',
]
pipeline(emb_import_paths)

