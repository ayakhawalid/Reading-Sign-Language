"""Write pandas DataFrames as LaTeX `tabular` fragments for the reports."""

from pathlib import Path


def df_to_tex(df, path, float_fmt="%.3f", column_format=None):
    """Write `df` as a bare tabular (no table wrapper); the report sets captions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tex = df.to_latex(index=False, escape=True,
                      float_format=lambda v: float_fmt % v,
                      column_format=column_format)
    path.write_text(tex, encoding="utf-8")
    return path
