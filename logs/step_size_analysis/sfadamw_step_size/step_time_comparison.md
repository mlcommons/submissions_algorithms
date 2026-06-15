# Step Execution Time Comparison of Schedule Free Adamw in Markdown (ms/step)

| Optimizer | criteo1tb | fastmri | finewebedu_lm | imagenet_resnet | imagenet_vit | librispeech_conformer | librispeech_deepspeech | ogbg | wmt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| JAX Schedule-Free v1 | 1507.4 ± 402.5 | 252.4 ± 30.2 | 224.7 ± 0.2 | 288.4 ± 16.8 | 618.4 ± 185.1 | 2591.7 ± 78.4 | 1285.0 ± 7.6 | 199.5 ± 2.4 | 131.9 ± 0.2 |
| JAX Schedule-Free v2 | 903.6 ± 6.6 | 195.4 ± 9.7 | 225.7 ± 1.0 | 457.5 ± 42.2 | 512.4 ± 25.6 | 2731.3 ± 50.3 | 2781.9 ± 3.6 | 196.4 ± 1.1 | 132.2 ± 0.3 |
| PyTorch Schedule-Free v1 | 1248.2 ± 375.1 | 90.8 ± 16.8 | 390.8 ± 0.4 | 755.1 ± 0.4 | 797.9 ± 4.0 | 640.6 ± 5.9 | 442.4 ± 3.4 | 228.5 ± 1.3 | 148.2 ± 0.6 |
| PyTorch Schedule-Free v2 | 896.2 ± 33.2 | 78.7 ± 6.8 | 395.6 ± 0.3 | 780.5 ± 76.9 | 876.3 ± 30.6 | 669.2 ± 1.7 | 514.2 ± 27.7 | 217.4 ± 5.8 | 140.4 ± 0.1 |

## LaTeX Source Code

```latex
\begin{table*}[t]
\centering
\caption{Step Execution Time Comparison (milliseconds per step) across different workloads.}
\label{tab:step_time_comparison}
\begin{tabular}{lrrrrrrrrr}
\toprule
Optimizer & criteo1tb & fastmri & finewebedu\_lm & imagenet\_resnet & imagenet\_vit & librispeech\_conformer & librispeech\_deepspeech & ogbg & wmt \\
\midrule
JAX Schedule-Free v1 & $1507.4 \pm 402.5$ & $252.4 \pm 30.2$ & $224.7 \pm 0.2$ & $288.4 \pm 16.8$ & $618.4 \pm 185.1$ & $2591.7 \pm 78.4$ & $1285.0 \pm 7.6$ & $199.5 \pm 2.4$ & $131.9 \pm 0.2$ \\
JAX Schedule-Free v2 & $903.6 \pm 6.6$ & $195.4 \pm 9.7$ & $225.7 \pm 1.0$ & $457.5 \pm 42.2$ & $512.4 \pm 25.6$ & $2731.3 \pm 50.3$ & $2781.9 \pm 3.6$ & $196.4 \pm 1.1$ & $132.2 \pm 0.3$ \\
PyTorch Schedule-Free v1 & $1248.2 \pm 375.1$ & $90.8 \pm 16.8$ & $390.8 \pm 0.4$ & $755.1 \pm 0.4$ & $797.9 \pm 4.0$ & $640.6 \pm 5.9$ & $442.4 \pm 3.4$ & $228.5 \pm 1.3$ & $148.2 \pm 0.6$ \\
PyTorch Schedule-Free v2 & $896.2 \pm 33.2$ & $78.7 \pm 6.8$ & $395.6 \pm 0.3$ & $780.5 \pm 76.9$ & $876.3 \pm 30.6$ & $669.2 \pm 1.7$ & $514.2 \pm 27.7$ & $217.4 \pm 5.8$ & $140.4 \pm 0.1$ \\
\bottomrule
\end{tabular}
\end{table*}
```
