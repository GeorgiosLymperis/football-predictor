import pandas as pd
import pytest

from match_predict.training.poisson_model import prepare_df


def _matches():
    return pd.DataFrame({'team1': ['A', 'Promoted'], 'team2': ['B', 'A'], 'score1': [1, 0], 'score2': [0, 2]})


def test_prepare_df_indexes_teams():
    out = prepare_df(_matches(), ['A', 'B', 'Promoted'])
    assert out['home_team_index'].tolist() == [0, 2]
    assert out['away_team_index'].tolist() == [1, 0]


def test_prepare_df_rejects_teams_missing_from_the_team_list():
    # Before this check, 'Promoted' got a NaN index that was later cast to 0,
    # silently training its matches as if team 'A' had played them.
    with pytest.raises(ValueError, match='Promoted'):
        prepare_df(_matches(), ['A', 'B'])
