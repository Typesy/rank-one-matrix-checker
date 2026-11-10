"""Regression fixtures from the inspected final LZ source and its audit.

Literal source discrepancies stay labeled as reproduction cases, never
silently adopted as a mathematical classification oracle.
"""
import json
from pathlib import Path

import pytest

from rankone.field import Field
from rankone.fibre import fibre_check,verify_fibre
from rankone.linalg import elimination,flatten


AUDIT=Path(__file__).resolve().parent/'fixtures'
POSITIVE=json.loads((AUDIT/'source_fixture_checks.json').read_text())
REPRODUCTION=json.loads((AUDIT/'quadric_source_checks.json').read_text())


@pytest.mark.parametrize('record',POSITIVE+REPRODUCTION,
                         ids=[f"q{r['q']}-{r['label']}" for r in POSITIVE+REPRODUCTION])
def test_inspected_source_fixtures_preserve_independence_and_no_extra_contract(record):
    f=Field(record['q'],[1,1,1] if record['q']==4 else None)
    matrices=[[[f.mul(x,y) for y in u] for x in u] for u in record['vectors']]
    problem={'field':f.metadata(),'generators':matrices}
    assert elimination([flatten(a) for a in matrices],f)[0]==record['span_rank']
    checked=fibre_check(problem)
    assert checked['status']==('VALID' if record['valid'] else 'INVALID')
    if not record['valid']:
        assert checked['reason']=='dependent_generators'
    else:
        assert checked['stats']['rank_one_directions']==record['k']
    assert verify_fibre(checked)['verified'] is True


@pytest.mark.parametrize('q,d,expected',[(2,2,'VALID'),(3,2,'INVALID'),(3,3,'VALID')])
def test_diagonal_plus_all_ones_construction_has_the_stated_dimension_guard(q,d,expected):
    matrices=[]
    for i in range(d):
        a=[[0]*d for _ in range(d)];a[i][i]=1;matrices.append(a)
    matrices.append([[1]*d for _ in range(d)])
    checked=fibre_check({'field':{'order':q},'generators':matrices})
    assert checked['status']==expected
    assert verify_fibre(checked)['verified'] is True
