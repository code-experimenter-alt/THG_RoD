"""Regression for dialogue-prefix versus actual-speaker identity."""
import pytest

from tools.prepare_iemocap_disjoint import speaker_of


@pytest.mark.parametrize("uid,speaker", [
    ("Ses01F_impro01_M000", "Ses01M"),
    ("Ses02M_script01_1_F012", "Ses02F"),
    ("Ses05F_impro02_F003", "Ses05F"),
])
def test_actual_speaker_uses_utterance_suffix(uid, speaker):
    assert speaker_of(uid) == speaker


def test_missing_identity_is_not_inferred():
    with pytest.raises(ValueError):
        speaker_of("unknown")
