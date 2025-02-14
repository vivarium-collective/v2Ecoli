"""
tests that vivarium-ecoli equilibrium process update is the same as saved wcEcoli updates
"""

import pytest

from ecoli.processes.equilibrium import Equilibrium
from migration.migration_utils import run_and_compare


@pytest.mark.master
def test_equilibrium_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, Equilibrium, layer=1)
        run_and_compare(initial_time, Equilibrium, layer=1, operons=False)


if __name__ == "__main__":
    test_equilibrium_migration()
