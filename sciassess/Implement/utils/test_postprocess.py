from pytest import mark

import pandas as pd

from sciassess.Implement.utils.postprocess import extract_table


table_expected = pd.DataFrame([["1a", "80 nM"], ["2b", "60 nM"]],
                              columns=pd.MultiIndex.from_tuples([("Compound", ""), ("Affinities", "protein")]))
table_info = {"index": ("Compound", ""), "compare_fields": [("Affinities", "protein")]}


@mark.parametrize(
    "s, sample, expected",
    [
        ("xxx\n```csv\nCompound,Affinities\n,protein\n1a,80 nM\n2b,60 nM\n```\nhahaha", table_info, table_expected),
        ("Compound,Affinities\n,protein\n1a,80 nM\n2b,60 nM", table_info, table_expected)
    ],
)
def test_extract_table(s: str, sample: dict, expected: pd.DataFrame):
    extracted = extract_table(s, **sample)
    assert extracted.equals(expected)
