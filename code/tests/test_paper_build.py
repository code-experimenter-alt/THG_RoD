import pytest
from tools.verify_paper_build import minimum_margins, validate_margins, main_page_limit


def test_one_inch_ink_margins_pass():
    margins = minimum_margins(612, 792, [(73, 73, 539, 719)])
    assert margins == dict(left=73, bottom=73, right=73, top=73)
    validate_margins(margins)


@pytest.mark.parametrize("box", [(49, 73, 539, 719), (73, 49, 539, 719),
                                 (73, 73, 563, 719), (73, 73, 539, 743)])
def test_on_page_but_margin_violation_is_rejected(box):
    with pytest.raises(AssertionError, match="one-inch"):
        validate_margins(minimum_margins(612, 792, [box]))


def test_rrq_is_not_a_fourteen_page_revision():
    assert main_page_limit("initial") == main_page_limit("resubmission") == 10
    assert main_page_limit("revision") == 14
