# XKE (Explaining Knowledge Embedding models)

This is a heavily modified fork of [OpenKE](https://github.com/thunlp/OpenKE). This repository contains implementations of XKE-PRED and XKE-TRUE, as used in the paper:

- Arthur Colombini Gusmão, Alvaro Henrique Chaim Correia, Glauber De Bona, and Fabio Gagliardi Cozman. Interpreting Embedding Models of Knowledge Bases: A Pedagogical Approach. Proceedings of the ICML Workshop on Human Interpretability in Machine Learning, pp. 79-86, 2018. [Link](https://arxiv.org/abs/1806.09504)

## Disclaimer

Backward compatibility for past commits is not to be expected.

## License

This code was built from [this commit](https://github.com/arthurcgusmao/XKE/commit/99032f5fffa1644cd73e51e99a7347160b83d028) of [OpenKE](https://github.com/thunlp/OpenKE), which is distributed under the [MIT License](https://github.com/thunlp/OpenKE/blob/master/LICENSE). All code developed thereafter is distributed under the terms of the GNU General Public License, version 3 (or, at your choosing, any later version of that license). You can find the text of that license [here](https://github.com/arthurcgusmao/XKE/blob/xke-master/LICENSE). This code also has [PRA](https://github.com/arthurcgusmao/pra/tree/extract_features) as a dependency, which is also licensed under the GNU General Public License.

## Examples

Examples for running each stage of both XKE-PRED and XKE-TRUE can be found in the `examples/` directory:

- [`examples/emb_grid_search/`](https://github.com/arthurcgusmao/XKE/tree/xke-master/examples/emb_grid_search) Learn embedding models.
- [`examples/extract_features`](https://github.com/arthurcgusmao/XKE/tree/xke-master/examples/extract_features) Extract features.
- [`examples/explain`](https://github.com/arthurcgusmao/XKE/tree/xke-master/examples/explain) Learn an interpretable classifier (that uses features as input) and get its explanations.
