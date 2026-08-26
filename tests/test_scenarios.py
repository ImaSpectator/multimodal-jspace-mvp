import pytest

from backend.app.scenario_generator import generate_scenario, list_domains
from backend.app.schemas import ScenarioControls
from backend.app.simulator import run_generated_scenario


@pytest.mark.parametrize("domain", list_domains())
def test_every_domain_autoruns_with_conflict(domain):
    scenario = generate_scenario(ScenarioControls(domain=domain, difficulty="hard", include_conflict=True, seed=1234))
    result = run_generated_scenario(scenario, capacity_k=5, preserve_conflicts=True)
    assert result.evaluation.action_correct, (domain, result.evaluation.notes)
    assert result.evaluation.conflict_detected, domain
    assert result.evaluation.critical_evidence_retention == 1.0, (domain, result.evaluation.notes)
    assert result.evaluation.score == 100.0, (domain, result.evaluation.notes)


@pytest.mark.parametrize("domain", list_domains())
def test_every_domain_autoruns_without_conflict(domain):
    scenario = generate_scenario(ScenarioControls(domain=domain, difficulty="medium", include_conflict=False, seed=4321))
    result = run_generated_scenario(scenario, capacity_k=6, preserve_conflicts=True)
    assert not result.evaluation.conflict_detected, domain
    assert result.evaluation.critical_evidence_retention == 1.0, (domain, result.evaluation.notes)
    assert result.evaluation.action_correct, (domain, result.evaluation.notes)
