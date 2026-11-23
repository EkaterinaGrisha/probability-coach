import pytest

from coach import simulations
from coach.data import load_problems
from coach.simulations import (SIMULATIONS, base_rate, birthday, gamblers_fallacy, monty_hall, regression_to_mean,
                               simpson, small_samples, two_children)


@pytest.fixture(autouse=True)
def plots_to_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(simulations, "OUTPUT_DIR", tmp_path)


def test_monty_hall():
    assert monty_hall(trials=20_000)["switch_win_rate"] == pytest.approx(2 / 3, abs=0.02)
    assert monty_hall(n_doors=100)["switch_win_rate"] == pytest.approx(0.99, abs=0.01)
    random_host = monty_hall(host_knows=False, trials=20_000)
    assert random_host["switch_win_rate"] == pytest.approx(0.5, abs=0.03)
    assert random_host["discarded"] == pytest.approx(20_000 / 3, rel=0.05)


def test_base_rate():
    result = base_rate(prevalence=0.001, sensitivity=0.99, false_positive_rate=0.01)
    assert result["population"] == 100_000
    assert result["p_sick_if_positive"] == pytest.approx(0.0902, abs=1e-4)
    dna = base_rate(prevalence=2e-7, sensitivity=1.0, false_positive_rate=1e-6, population=5_000_000)
    assert dna["p_sick_if_positive"] == pytest.approx(1 / 6, abs=1e-3)


def test_birthday():
    assert birthday(23)["any_pair"] == pytest.approx(0.5073, abs=1e-4)
    assert birthday(23)["pairs"] == 253
    assert birthday(57)["any_pair"] == pytest.approx(0.9901, abs=1e-4)
    assert birthday(23, mode="my_birthday")["my_birthday"] == pytest.approx(0.0586, abs=1e-4)


def test_gamblers_fallacy():
    assert gamblers_fallacy(5)["after_streak"] == pytest.approx(0.5, abs=0.02)


def test_small_samples():
    hospitals = small_samples([15, 45], more_than=0.6)["samples"]
    assert [s["min_count"] for s in hospitals] == [10, 28]
    assert hospitals[0]["exact"] == pytest.approx(0.1509, abs=1e-4)
    coins = small_samples([10, 100], more_than=0.69)["samples"]
    assert [s["min_count"] for s in coins] == [7, 70]


def test_regression_to_mean():
    result = regression_to_mean(reliability=0.5, share=0.1)
    assert result["top_second"] < result["top_first"]
    assert result["bottom_second"] > result["bottom_first"]
    # при надёжности 0.5 отрыв от среднего во второй попытке в среднем вдвое меньше
    assert result["top_second"] - 60 == pytest.approx((result["top_first"] - 60) / 2, abs=1)


def test_two_children():
    assert two_children("any_boy")["p_both_boys"] == pytest.approx(1 / 3, abs=0.01)
    assert two_children("older_boy")["p_both_boys"] == pytest.approx(1 / 2, abs=0.01)


def test_simpson():
    rates = simpson({"А": {"x": [81, 87], "y": [192, 263]}, "Б": {"x": [234, 270], "y": [55, 80]}})["rates"]
    assert rates["А"]["x"] > rates["Б"]["x"] and rates["А"]["y"] > rates["Б"]["y"]
    assert rates["А"]["всего"] < rates["Б"]["всего"]


@pytest.mark.parametrize("problem", [p for p in load_problems() if p["simulation"]], ids=lambda p: p["id"])
def test_problem_params_fit_simulation(problem):
    SIMULATIONS[problem["simulation"]](**problem["sim_params"])
