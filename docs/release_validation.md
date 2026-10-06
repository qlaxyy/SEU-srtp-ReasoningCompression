# Release validation · 2026-10-07

The public package was assembled from the final successful 1.5B online and 7B v4 experiment implementations. A later research checkout contained 15 runtime files that differed from the frozen experiment manifest; the exact historical copies were recovered before export. Original research files and frozen result assets were left in place.

Completed release checks:

- **34 CPU tests passed**: input/gold separation, question identities, fitting overlap rejection, model-specific vectors and parameters, Cartesian-difference PCA equivalence, question-balanced calibration, bounded penalty arithmetic, lexical opening/phrase matching, natural phase transitions, off/zero behavior, request reordering and replay restoration.
- Both model-specific native runners were reached through exported public jobs in dry-run mode. All five method configurations were exercised without loading a GPU model.
- The pinned vLLM source plus public overlays and bundled EasySteer library matched **2,166 historical Python source hashes**. The overlay assembly was tested against a checkout of the stated upstream vLLM commit.
- The preserved parser/grader passed **three synthetic checks** for integer extraction, fractional equivalence and an incorrect answer. No benchmark outputs were regenerated for this check.
- Public Python sources were syntax-checked. The release tree was scanned for API-token patterns, private keys, deployment addresses, personal absolute paths and unexpected large files; none were found. Only the documented small vector/table binaries are included.

The CPU tensor checks used Windows, Python 3.9 and PyTorch 2.5.1 from an available local CPU-capable environment; source syntax/configuration checks additionally used Python 3.13. The **historical GPU environment is separately documented** as Linux/Python 3.12/PyTorch 2.11.0+cu129. These CPU checks do not certify a newly installed CUDA environment or reproduce benchmark accuracy. The release's automatic GPU engineering gates remain mandatory on the first actual evaluation.

Packaging changes consist of path/argument portability, sanitized job manifests, synthetic engineering fixtures, shared launch/scoring utilities and documentation. The retained numerical kernels/adapters have source-identity checks. `runtime/provenance.json` lists their historical hashes; `results/provenance.json` identifies the historical result reports by hash. The old public repository tree is replaced by a normal commit; its earlier Git history is preserved.
