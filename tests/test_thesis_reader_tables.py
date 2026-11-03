"""Reader layout must preserve results and keep merged labels on their pages."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/thesis'))
from export_fixed_statistics import grouped_cells, longtable, source_note
from resolve_method_clarifications import BASELINES


def test_merging_preserves_scores_and_respects_parent_groups():
    rows = [['A', 'Old', '.6'], ['A', 'Old', '.7'], ['B', 'Old', '.8'], ['B', 'New', '.9']]
    cells, ends = grouped_cells(rows, (0,1))
    assert cells[0][0] == r'\multirow[t]{2}{=}{A}'
    assert cells[0][1] == r'\multirow[t]{2}{=}{Old}'
    assert cells[2][1] == 'Old'  # must not span classifiers
    assert [r[-1] for r in cells] == [r[-1] for r in rows]
    assert ends == {1,3}
    assert rows[1][0] == 'A'  # input evidence is unchanged


def test_table_keeps_outer_group_together_and_credit_separate():
    text = longtable('Scores', 'tab:test', ['Model', 'Score'], ['.4','.5'],
                     [['A','.6'],['A','.7'],['B','.8']], 'Notes.', merge_columns=(0,))
    assert r' & .6 \\*' in text
    assert r' & .7 \\' in text
    assert r'(資料來源：本研究實驗結果整理)\par Notes.' in text
    assert source_note('資料來源：本研究整理。說明。') == r'(資料來源：本研究整理)\par 說明。'


def test_primary_results_use_one_final_tuned_record_per_classifier():
    names = [name for name, _ in BASELINES]
    assert names == ['XGBoost', 'LightGBM', 'Random Forest', 'Logistic Regression', 'TabM']
    assert len(set(names)) == len(names)
    assert 'rerun_20260406_gridplus_n24/tuned/' in BASELINES[0][1]
    assert all('/tuned/' in path for _,path in BASELINES)
