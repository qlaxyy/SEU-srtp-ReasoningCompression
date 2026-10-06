# Third-party notices and research attribution

Project-owned additions are licensed under Apache-2.0. Upstream copyright and licenses remain applicable to their source files and derivatives. Model weights and benchmark datasets are not included and remain subject to their providers' terms.

| Component | Source/version | Included scope | License |
|---|---|---|---|
| EasySteer | [ZJU-REAL/EasySteer](https://github.com/ZJU-REAL/EasySteer), reference commit `b771d3104f7f7c3181e7ce3b50fab565f8d08282` | Small Python-library snapshot and packaging metadata; local import/layout adaptations retained | Apache-2.0 |
| EasySteer vLLM fork | [ZJU-REAL/EasySteer-vllm-v1](https://github.com/ZJU-REAL/EasySteer-vllm-v1), base `6267ca0cfc9c6e93b1427d36b1d655821d6d6f9b` | 12 overlaid source files; full base fetched by setup | Apache-2.0 |
| ReBalance | [yu-lin-li/ReBalance](https://github.com/yu-lin-li/ReBalance); reference upstream commit `c7207ee1583e42170ac4f324c12345863b3d40f0` | Preserved parser/grader, mathematical normalization and adapted controller logic | MIT, Copyright 2026 Yulin Li |

License copies are in `licenses/`, `third_party/EasySteer/LICENSE` and `third_party/rebalance/LICENSE`. The included ReBalance snapshot was not globally certified identical to the reference upstream commit; preserved file hashes identify exactly what this release uses. Its grader also carries the upstream attribution to MATH, CRITIC, PRM800K, ToRA and DeepSeek-Math.

Files in `runtime/overlays/vllm/` are project-modified versions or additions against the stated base. They include the dynamic ReBalance state/controller, graph payload integration, sampler raw-confidence path and previously developed optional hooks. The public default leaves hybrid termination disabled. `runtime/expected_hashes.json` records the complete relevant Python runtime identity; `runtime/provenance.json` records the historical source hashes for the retained kernels and portable wrappers. This package is not an official mirror or release of any upstream project.

## Papers and method sources

- **Efficient Reasoning with Balanced Thinking** — ReBalance. See the [author repository](https://github.com/yu-lin-li/ReBalance) for its paper and citation.
- **EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering** — [arXiv:2509.25175](https://arxiv.org/abs/2509.25175).
- **SRQ paper** — [ACL Anthology, Findings 2026, paper 1507](https://aclanthology.org/2026.findings-acl.1507.pdf). SRQ motivates quality-based sample selection; the paper's PCA-CAA description does not fully specify this project's rank, pairing, normalization or online injection convention.
- **Improving Reasoning Performance in Large Language Models via Representation Engineering** — [arXiv:2504.19483](https://arxiv.org/abs/2504.19483), [author code](https://github.com/bertramhojer/improve-reasoning-iclr-2025), inspected at `d581643c38e8643499152377178ba6f190b71e73`. The project reviewed its `repana` PCA/CAA implementation. The released PCA16 formula is independently implemented and explicitly different from simply using the first principal component.

The combined method, fitted penalty scale and answer-stage adaptation should be attributed to this project in addition to the underlying methods. Reusing upstream infrastructure is not claimed as inventing ReBalance, EasySteer, vLLM, SRQ, CAA or PCA.
