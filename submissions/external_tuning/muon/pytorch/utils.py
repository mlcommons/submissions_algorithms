import logging


def _split_params_muon_adam(model):
    """Split parameters:
    - Muon: all matrix params (ndim ≥ 2) except embeddings
    - Adam: 1D params, all embeddings
    """
    ## too simplistic
    # params = [p for p in model.parameters() if p.requires_grad]
    # matrix_params = [p for p in params if p.ndim >= 2]
    # non_matrix_params = [p for p in params if p.ndim < 2]

    muon_params, adam_params = [], []
    muon_infos, adam_infos = [], []  # for logging only

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Assign embeddings to Adam (wmt, criteo)
        if "embedding" in n.lower():
            adam_params.append(p)
            adam_infos.append(f"{n} (ndim={p.ndim})")
        elif p.ndim >= 2:
            muon_params.append(p)
            muon_infos.append(f"{n} (ndim={p.ndim})")
        else:
            adam_params.append(p)
            adam_infos.append(f"{n} (ndim={p.ndim})")

    logging.info("Muon params:\n\t" + "\n\t".join(muon_infos))
    logging.info("Adam params:\n\t" + "\n\t".join(adam_infos))

    return muon_params, adam_params
