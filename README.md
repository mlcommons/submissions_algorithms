# MLCommons™ AlgoPerf: Training Algorithms Leaderboard

<br />
<p align="center">
<a href="#"><img width="600" img src="/.assets/mlc_logo.png" alt="MLCommons Logo"/></a>
</p>

This repository hosts the official rolling leaderboard for the [**AlgoPerf: Training Algorithms benchmark**](https://github.com/mlcommons/algorithmic-efficiency) by [**MLCommons**](https://mlcommons.org/).
The benchmark measures neural network training speedups due to algorithmic improvements in training algorithms.
The leaderboard tracks the aggregate performance of different algorithms on a variety of [workloads](https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#workloads) and under two different [tuning rulesets](https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#tuning).

> [!NOTE]  
> **If you want to submit to the AlgoPerf benchmark, please open a PR with your submission. The AlgoPerf working group will review your submission and potentially evaluate your submission on all workloads. For more details, see the [How to Submit](#how-to-submit) section.**

## Live Leaderboards

> **Leaderboard Version:** 0.6  
> **Last Updated:** 2025-03-24 15:07 UTC  
> **Using Benchmark Version:** [latest](https://github.com/mlcommons/algorithmic-efficiency)

> [!TIP]
> The leaderboard of the first AlgoPerf competition with more entries can be found [here](./previous_leaderboards/algoperf_v05/README.md).

### External Tuning Ruleset Leaderboard

_In the external tuning ruleset, submission must provide workload-agnostic hyperparameter search spaces and they will get_ $5$ _tuning trials per workload sampled from this search space._

<!-- BEGIN EXTERNAL TUNING LEADERBOARD -->

| **Rank** | **Submission**                                                                                                                                                                                                                                                                                                                                                                                                              | **Authors**                                                                                  | **Affiliation** | **Framework** | **Logs**                             | **Score**  |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------- | ------------- | ------------------------------------ | ---------- |
| 1.       | <details><summary>[**Distributed Shampoo**](submissions/external_tuning/shampoo/)</summary>Based on the Distributed Shampoo algorithm of [Anil et al. (2020)](https://arxiv.org/abs/2002.09018) with an implementation tailored to leverage PyTorch performance optimizations. See [Shi et al. (2023)](https://arxiv.org/abs/2309.06497) for details. The submission uses a list of five hyperparameter settings.</details> | Hao-Jun Shi, Tsung-Hsien Lee, Anna Cai, Shintaro Iwasaki, Wenyin Fu, Yuchen Hao, Mike Rabbat | Meta Platforms  | PyTorch       | [💾](logs/external_tuning/shampoo/)  | **0.6244** |
| 2.       | <details><summary>[**_Baseline_**](submissions/external_tuning/baseline/)</summary>Baseline using NadamW ([Dozat, 2016](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ); [Loshchilov & Hutter, 2019](https://openreview.net/forum?id=Bkg6RiCqY7)) and a linear learning rate warmup followed by a cosine decay ([Dahl et al., 2023](https://arxiv.org/abs/2306.07179)).</details>                                     |                                                                                              |                 | JAX           | [💾](logs/external_tuning/baseline/) | **0.4590** |

<!-- END EXTERNAL TUNING LEADERBOARD -->

### Self-Tuning Ruleset Leaderboard

_In the self-tuning ruleset, submissions must be completely hyperparameter-free._

> [!NOTE]
> The first self-tuning submissions are currently being scored.

<!-- BEGIN SELF-TUNING LEADERBOARD -->

<!-- | **Rank** | **Submission**                                                                                                                                                                                                                                                                                                   | **Authors**                                                                                                          | **Affiliation**            | **Framework** | **Logs**                                    | **Score**  |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------- | ------------- | ------------------------------------------- | ---------- |
| 1.       | <details><summary>[**Schedule Free AdamW v2**](submissions/self_tuning/schedule_free_adamw_v2/)</summary>A self-tuning version of Schedule Free AdamW ([Defazio et al., 2024](https://openreview.net/forum?id=0XeNkkENuI)) using a single hyperparameter configuration.</details>                                      | Alice Yang, Aaron Defazio, Konstantin Mishchenko                                                                     | Meta AI, Samsung AI        | PyTorch       | [💾](logs/self_tuning/schedule_free_adamw_v2/) | **0.8542** |
| 2.       | <details><summary>[**_Baseline_**](submissions/self_tuning/baseline/)</summary>Baseline using NadamW, a linear learning rate warmup followed by a cosine decay, and a single hyperparameter point ([Dahl et al., 2023](https://arxiv.org/abs/2306.07179)).</details>                                             |                                                                                                                      |                            | JAX           | [💾](logs/self_tuning/baseline/)            | **0.8194** | -->

<!-- END SELF-TUNING LEADERBOARD -->

## How to Submit

To submit your algorithm for evaluation on the AlgoPerf leaderboard, please follow these steps:

1. **Implement your algorithm in the AlgoPerf API:** Have a look at our [Getting Started Guide](https://github.com/mlcommons/algorithmic-efficiency/blob/main/GETTING_STARTED.md) and the [Technical Documentation](https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md).
2. **Create a Pull Request:** Fork this repository, create a new branch and add your submission code to a new folder within either `submissions/external_tuning/` or `submissions/self_tuning`. Open a pull request (PR) to the `evaluation` branch of this repository. Make sure to fill out the PR template asking for information such as submission name, authors, affiliations, etc.
3. **PR Review and Evaluation:** The AlgoPerf working group will review your PR. Based on our available resources and the perceived potential of the method, it will be selected for a free evaluation and merged into the `evaluation` branch. The working group will run your submission on all workloads and push the results, as well as the updated leaderboard, to the `main`branch.

## Scoring

The code that computes this leaderboard lives in [`scoring/`](./scoring/). Given a
directory of submission logs (such as those under [`previous_leaderboards/`](./previous_leaderboards/)),
it computes the performance profiles, time-to-target, AlgoPerf benchmark scores, and
speedups used in the tables above. This code was moved here from the
[`scoring/` directory of the algorithmic-efficiency repository](https://github.com/mlcommons/algorithmic-efficiency)
so that the repository that hosts the leaderboard also owns the code that produces it.

### Installation

The scoring code depends on the [`algoperf`](https://github.com/mlcommons/algorithmic-efficiency)
package for the workload registry and target metrics. We recommend a fresh conda
environment:

```bash
conda create -n algoperf-scoring python=3.11 -y && conda activate algoperf-scoring
pip install -e .          # installs the scoring tooling and the algoperf dependency
```

### Regenerating the leaderboard

```bash
# External tuning ruleset
python -m scoring.score_submissions \
  --submission_directory previous_leaderboards/algoperf_v06/logs/external_tuning \
  --compute_performance_profiles \
  --output_dir scoring_results_external_tuning

# Self-tuning ruleset (add --self_tuning_ruleset)
python -m scoring.score_submissions \
  --submission_directory previous_leaderboards/algoperf_v06/logs/self_tuning \
  --compute_performance_profiles \
  --self_tuning_ruleset \
  --output_dir scoring_results_self_tuning
```

See the [scoring methodology](https://github.com/mlcommons/algorithmic-efficiency/blob/main/docs/DOCUMENTATION.md#scoring)
in the benchmark documentation for details on how scores are computed.

## Citation

If you use the _AlgoPerf benchmark_ in your research, please consider citing our paper.

> [Dahl, Schneider, Nado, et al.<br/> > **Benchmarking Neural Network Training Algorithms**<br/> > _arXiv 2306.07179_](http://arxiv.org/abs/2306.07179)

```bibtex
@Misc{Dahl2023AlgoPerf,
  title         = {{Benchmarking Neural Network Training Algorithms}},
  author        = {Dahl, George E. and Schneider, Frank and Nado, Zachary and Agarwal, Naman and Sastry, Chandramouli Shama and Hennig, Philipp and Medapati, Sourabh and Eschenhagen, Runa and Kasimbeg, Priya and Suo, Daniel and Bae, Juhan and Gilmer, Justin and Peirson, Abel L. and Khan, Bilal and Anil, Rohan and Rabbat, Mike and Krishnan, Shankar and Snider, Daniel and Amid, Ehsan and Chen, Kongtao and Maddison, Chris J. and Vasudev, Rakshith and Badura, Michal and Garg, Ankush and Mattson, Peter},
  year          = {2023},
  archiveprefix = {arXiv},
  eprint        = {2306.07179},
}
```

If you use the results from the first _AlgoPerf competition_, please consider citing the results paper, as well as the relevant submissions:

> [Kasimbeg, Schneider, Eschenhagen, et al.<br/> > **Accelerating neural network training: An analysis of the AlgoPerf competition**<br/>
> ICLR 2025](https://openreview.net/forum?id=CtM5xjRSfm)

```bibtex
@inproceedings{Kasimbeg2025AlgoPerfResults,
title           = {Accelerating neural network training: An analysis of the {AlgoPerf} competition},
author          = {Kasimbeg, Priya and Schneider, Frank and Eschenhagen, Runa and Bae, Juhan and Sastry, Chandramouli Shama and Saroufim, Mark and Boyuan, Feng and Wright, Less and Yang, Edward Z. and Nado, Zachary and Medapati, Sourabh and Hennig, Philipp and Rabbat, Michael and Dahl, George E.},
booktitle       = {The Thirteenth International Conference on Learning Representations},
year            = {2025},
url             = {https://openreview.net/forum?id=CtM5xjRSfm}
}
```
