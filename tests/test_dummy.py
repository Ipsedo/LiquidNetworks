# -*- coding: utf-8 -*-
from typing import Any

import pytest

from liquid_netorks.dummy import identity


@pytest.mark.parametrize("var", [1, "Azerty", True, print])
def test_identity(var: Any) -> None:
    var_identity = identity(var)

    assert isinstance(var_identity, type(var))
    assert var == var_identity
