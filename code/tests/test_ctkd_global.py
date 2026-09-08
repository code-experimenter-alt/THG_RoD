import math

import pytest
import torch

from src.ctkd import GlobalTemperature, curriculum_magnitude
from src.losses import kd_loss
from src.routing import mode_needs_teacher_logits


@pytest.mark.parametrize("epoch,expected", [(0,0), (5,.5), (10,1), (15,1)])
def test_cosine_curriculum(epoch, expected):
    assert curriculum_magnitude(epoch) == pytest.approx(expected)


def test_reversal_preserves_student_gradient_and_reverses_temperature():
    model = GlobalTemperature().double()
    student = torch.tensor([[1., -.3, .4], [.2, .5, -.1]], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([[.1, 1.2, -.5], [.8, -.2, .4]], dtype=torch.float64)
    y = torch.tensor([1, 0])
    epoch = 5
    learned = kd_loss(student, teacher, y, .7, model(epoch))
    learned.backward()
    student_grad, raw_grad = student.grad.clone(), model.raw.grad.clone()
    ordinary_raw = torch.ones(1, dtype=torch.float64, requires_grad=True)
    ordinary_student = student.detach().clone().requires_grad_(True)
    reference = kd_loss(ordinary_student, teacher, y, .7, 1 + 20 * ordinary_raw.sigmoid())
    reference.backward()
    torch.testing.assert_close(student_grad, ordinary_student.grad)
    torch.testing.assert_close(raw_grad, -curriculum_magnitude(epoch) * ordinary_raw.grad)
    assert abs(raw_grad.item()) > 1e-8
    assert model(1).item() == pytest.approx(1 + 20/(1+math.exp(-1)))
    assert mode_needs_teacher_logits("ctkd_global")
