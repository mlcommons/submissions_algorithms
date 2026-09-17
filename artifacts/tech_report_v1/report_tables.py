"""Shared LaTeX table rendering for standard, step, and relaxed results."""

from artifacts.tech_report_v1.report_utils import latex_name

WORKLOAD_LATEX = {
  'criteo1tb': r'\criteo',
  'fastmri': r'\fastmri',
  'finewebedu_lm': r'\finewebedu',
  'imagenet_resnet': r'\resnet',
  'imagenet_vit': r'\vit',
  'librispeech_conformer': r'\conformer',
  'librispeech_deepspeech': r'\deepspeech',
  'ogbg': r'\ogbg',
  'wmt': r'\wmt',
}


def scores_to_latex(
  scores_df, caption='AlgoPerf Self-Tuning Leaderboard', label='tab:scores'
):
  """
  Render a leaderboard scores DataFrame as a LaTeX table.
  Submission names come from the DataFrame index (pretty-named), rendered
  via their `latex_name()` \\newcommand macro.
  Scores are formatted to 4 decimal places; best score is bolded.
  """
  df = scores_df.sort_values('score', ascending=False).copy()
  df['rank'] = range(1, len(df) + 1)

  best_score = df['score'].iloc[0]

  rows = []
  for rank, (name, row) in enumerate(df.iterrows(), start=1):
    score_str = f'{row["score"]:.4f}'
    if row['score'] == best_score:
      score_str = r'\textbf{' + score_str + '}'
    rows.append(f'    {rank} & {latex_name(name)} & {score_str} \\\\')

  body = '\n'.join(rows)

  latex = (
    r'\begin{table}[h]' + '\n'
    r'  \centering' + '\n'
    r'  \caption{' + caption + '}\n'
    r'  \label{' + label + '}\n'
    r'  \begin{tabular}{rlr}' + '\n'
    r'    \toprule' + '\n'
    r'    Rank & Submission & Score \\' + '\n'
    r'    \midrule' + '\n' + body + '\n'
    r'    \bottomrule' + '\n'
    r'  \end{tabular}' + '\n'
    r'\end{table}'
  )
  return latex


def workload_table(frame, *, caption, label, bold=None):
  """Render formatted workload values; align optional emphasis by row/column name."""
  headers = ' & '.join(WORKLOAD_LATEX.get(w, w) for w in frame.columns)
  rows = []
  for name, row in frame.iterrows():
    cells = []
    for workload, value in row.items():
      cell = str(value).replace('%', r'\%')
      if bold is not None and bold.loc[name, workload]:
        cell = r'\textbf{' + cell + '}'
      cells.append(cell)
    rows.append(
      '      ' + latex_name(name) + ' & ' + ' & '.join(cells) + r' \\'
    )
  return '\n'.join(
    [
      r'\begin{table}[htbp]',
      r'  \centering',
      r'  \caption{' + caption + '}',
      r'  \label{' + label + '}',
      r'  \resizebox{\textwidth}{!}{%',
      r'  \begin{tabular}{' + 'l' + 'r' * len(frame.columns) + '}',
      r'    \toprule',
      r'    Submission & ' + headers + r' \\',
      r'    \midrule',
      *rows,
      r'    \bottomrule',
      r'  \end{tabular}%',
      r'  }',
      r'\end{table}',
    ]
  )
