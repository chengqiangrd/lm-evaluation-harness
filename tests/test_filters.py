import importlib.util
from pathlib import Path

import pytest

from lm_eval.filters.extraction import MultiChoiceRegexFilter
from lm_eval.filters.transformation import SPANFilter


def test_multi_choice_regex_all_empty_capture_groups_falls_back_to_choice_text():
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"()()",
        ignore_case=True,
        ignore_punctuation=True,
    )

    resps = [["alpha"]]
    docs = [{"choices": ["alpha", "beta"]}]

    assert filt.apply(resps, docs) == [["(A)"]]


def test_multi_choice_regex_all_empty_capture_groups_falls_back_to_bare_letter():
    filt = MultiChoiceRegexFilter(regex_pattern=r"()()")

    resps = [[": B"]]
    docs = [{"choices": ["alpha", "beta"]}]

    assert filt.apply(resps, docs) == [["(B)"]]


def test_format_span_normalizes_label_only():
    # Labels are normalized, but entity text containing label-words as
    # substrings (e.g. "Company", "Country", "George") must be left intact.
    filt = SPANFilter()
    resps = [["ORGANIZATION: Shell Company $ LOCATION: Country Club $ PERSON: George"]]

    assert filt.apply(resps, [{}]) == [
        ["org: shell company $ loc: country club $ per: george"]
    ]


def test_multi_choice_regex_prefix_choice_does_not_shadow_longer_choice():
    # When one choice's text is a prefix of another, naming the longer choice in the
    # response must map to the longer choice's letter. Regression: the fallback regex
    # joined choices in list order, and leftmost-alternation let the shorter prefix
    # ("Guilty") shadow "Guilty of Romance", returning (A) instead of (B).
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"()()",
        ignore_case=True,
        ignore_punctuation=True,
    )

    resps = [["the answer is Guilty of Romance"]]
    docs = [{"choices": ["Guilty", "Guilty of Romance"]}]

    assert filt.apply(resps, docs) == [["(B)"]]


def test_multi_choice_regex_prefix_fix_holds_under_task_config():
    # Every task that uses this filter passes group_select=-1 and a "(\\([A-Z]\\))"
    # pattern, not the defaults; the shorter choice named alone must still win.
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        ignore_case=True,
        ignore_punctuation=True,
    )
    docs = [{"choices": ["Guilty", "Guilty of Romance"]}]

    assert filt.apply([["the answer is Guilty of Romance"]], docs) == [["(B)"]]
    assert filt.apply([["the answer is Guilty"]], docs) == [["(A)"]]


@pytest.mark.parametrize("choices", [["-5", "5"], ["15", "1.5"]])
@pytest.mark.parametrize("reverse_choices", [False, True])
def test_multi_choice_regex_preserves_colliding_choice_punctuation(
    choices, reverse_choices
):
    if reverse_choices:
        choices = list(reversed(choices))
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        ignore_case=True,
        ignore_punctuation=True,
    )
    resps = [[f"The answer is {choice}" for choice in choices]]

    assert filt.apply(resps, [{"choices": choices}]) == [["(A)", "(B)"]]


@pytest.mark.parametrize("choices", [["-5", "5"], ["5", "-5"]])
@pytest.mark.parametrize(
    "group_select,selected_choice", [(0, "-5"), (1, "5"), (-1, "5")]
)
def test_multi_choice_regex_resolves_only_the_selected_match(
    choices, group_select, selected_choice
):
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=group_select,
        ignore_punctuation=True,
    )
    target = f"({chr(ord('A') + choices.index(selected_choice))})"

    assert filt.apply([["Not -5; the answer is 5."]], [{"choices": choices}]) == [
        [target]
    ]


def test_multi_choice_regex_preserves_primary_regex_priority_with_collisions():
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        ignore_punctuation=True,
    )

    assert filt.apply(
        [["The answer is (B), not -5.", "(A) or (B): -5"]],
        [{"choices": ["-5", "5"]}],
    ) == [["(B)", "(B)"]]


def test_multi_choice_regex_retains_configured_ignores_when_resolving_collisions():
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        ignore_punctuation=True,
        regexes_to_ignore=[r" USD"],
    )

    assert filt.apply(
        [["The answer is -5 USD", "Not -5 USD; the answer is 5 USD."]],
        [{"choices": ["-5 USD", "5 USD"]}],
    ) == [["(A)", "(B)"]]


def test_multi_choice_regex_preserves_shape_and_unambiguous_normalization():
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        ignore_case=True,
        ignore_punctuation=True,
        fallback="unmatched",
    )

    assert filt.apply(
        [["ALPHA!", "beta", "unknown"], ["-5", "5", "unknown"]],
        [{"choices": ["alpha", "beta"]}, {"choices": ["-5", "5"]}],
    ) == [["(A)", "(B)", "unmatched"], ["(A)", "(B)", "unmatched"]]


@pytest.mark.parametrize(
    "choices,response,options",
    [
        (["Na", "NA"], "Na", {"ignore_case": True}),
        (["NA", "Na"], "Na", {"ignore_case": True}),
        (
            ["-5", "5"],
            "-5",
            {"ignore_punctuation": True, "regexes_to_ignore": [r"-"]},
        ),
        (
            ["5", "-5"],
            "-5",
            {"ignore_punctuation": True, "regexes_to_ignore": [r"-"]},
        ),
        (["-5", "-5", "5"], "-5", {"ignore_punctuation": True}),
        (["alpha", "alpha"], "alpha", {}),
        (["-5", "5."], "5", {"ignore_punctuation": True}),
        (["-5", "5-"], "-5-", {"ignore_punctuation": True}),
    ],
)
def test_multi_choice_regex_ambiguous_choices_use_existing_fallbacks(
    choices, response, options
):
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        fallback="ambiguous",
        **options,
    )

    assert filt.apply(
        [[response, response + ": A", response + " (B)"]], [{"choices": choices}]
    ) == [["ambiguous", "(A)", "(B)"]]


def test_multi_choice_regex_duplicate_choices_do_not_hide_a_distinct_choice():
    filt = MultiChoiceRegexFilter(regex_pattern=r"(\([A-Z]\))", ignore_punctuation=True)

    assert filt.apply([["5"]], [{"choices": ["-5", "-5", "5"]}]) == [["(C)"]]


@pytest.mark.parametrize(
    "choices,options",
    [
        (["!!!", "beta"], {"ignore_punctuation": True}),
        (["alpha", "beta"], {"regexes_to_ignore": [r"alpha"]}),
        (["!", "?"], {"ignore_punctuation": True}),
        (["", "beta"], {}),
    ],
)
def test_multi_choice_regex_empty_normalized_choice_keeps_existing_fallback(
    choices, options
):
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        fallback="unmatched",
        **options,
    )

    assert filt.apply([["unknown", ": B"]], [{"choices": choices}]) == [
        ["unmatched", "(B)"]
    ]


@pytest.mark.parametrize("group_select,target", [(0, "(A)"), (-1, "(B)")])
def test_multi_choice_regex_keeps_normalized_match_selection_with_collisions(
    group_select, target
):
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=group_select,
        ignore_punctuation=True,
    )

    assert filt.apply(
        [["foobar; the answer is -5"]], [{"choices": ["foo-bar", "-5", "5"]}]
    ) == [[target]]


@pytest.mark.parametrize("choices", [["-5", "5"], ["15", "1.5"]])
def test_multi_choice_regex_resolves_collisions_inside_surrounding_punctuation(choices):
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        ignore_punctuation=True,
    )

    assert filt.apply(
        [[f"The answer is ({choices[0]}).", f"The answer is [{choices[1]}]."]],
        [{"choices": choices}],
    ) == [["(A)", "(B)"]]


@pytest.mark.parametrize("variant", ["zeroshot", "cot_zeroshot"])
def test_bbh_multi_choice_regex_prefix_choice_does_not_shadow_longer_choice(variant):
    # bbh keeps its own copy of MultiChoiceRegexFilter, wired via
    # `!function utils.MultiChoiceRegexFilter`, so it needs the same longest-first
    # ordering. Mirrors the bbh_zeroshot_movie_recommendation filter config.
    path = Path(__file__).parent.parent / f"lm_eval/tasks/bbh/{variant}/utils.py"
    spec = importlib.util.spec_from_file_location(f"bbh_{variant}_utils", path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    filt = utils.MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=0,
        ignore_case=True,
        ignore_punctuation=True,
    )
    docs = [
        {
            "input": "Find a movie similar to Batman Begins:\nOptions:\n"
            "(A) Batman\n(B) Batman Returns\n(C) Alien\n(D) Titanic"
        }
    ]

    assert filt.apply([["The answer is Batman Returns"]], docs) == [["(B)"]]
    assert filt.apply([["The answer is Batman"]], docs) == [["(A)"]]
