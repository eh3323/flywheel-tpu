#!/usr/bin/env python3
# Copyright 2026 heavyball-research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pick a pull request's TPU tests, and summarize a TPU test run.

    python .github/ci/tpu_tests.py select --changed-configs changed.json \\
        --changed-files files.txt --out selected.txt --summary summary.md
    python .github/ci/tpu_tests.py report --junit junit.xml \\
        [--collected collected.txt] --summary summary.md

select reads the configurations whose IR differs from the pull request's
merge base (scripts/mlir_snapshot.py compare --changed) and the files the
pull request changes, and writes the pytest arguments to run, one per line:
- the tests of every API one of whose configurations changed (API_TESTS);
- every test file the pull request adds or changes, and every test file that
  imports a changed module under tests/, directly or through another one.
An API that API_TESTS does not name is an error, so a configuration added to
mlir_snapshot.py for a new entry point comes with its tests here. So is a
test file under tests/ that no API names: otherwise it would run only when a
pull request changes it, and weekly.

report writes a Markdown summary of a JUnit file: the counts, the tests that
failed, and with --collected (pytest --collect-only -q) the tests the run
never reached, as a weekly run that hits the end of its window leaves. It
exits 1 if a test failed or was not reached.
"""

import argparse
import ast
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTS_DIR = "tests"
SOFTMAX = f"{TESTS_DIR}/softmax_attention"
PREFILL = f"{SOFTMAX}/test_prefill.py"
REGRESSION = f"{SOFTMAX}/test_regression.py"
PAGED = f"{SOFTMAX}/test_paged.py"
PAGE = f"{SOFTMAX}/test_page.py"
# Test files that test several APIs, split by test function below. A test
# function of theirs that no group names runs with every softmax group.
MIXED_FILES = (PREFILL, REGRESSION)
# Failed tests listed in a report, past which only the count is given.
MAX_LISTED = 50


def functions(path: str, *names: str) -> list[str]:
    return [f"{path}::{name}" for name in names]


DENSE = [
    *functions(
        PREFILL, "test_flash_attn_output", "test_flash_attn_race_condition"
    ),
    *functions(
        REGRESSION,
        "test_flash_attn_overflow_guard_recovery",
        "test_flash_attn_window_size_crosses_kv_compute_tiles",
        "test_flash_attn_rect_short_q_long_kv",
        "test_flash_attn_rect_fully_masked_q_block",
        "test_flash_attn_array_softmax_scale",
        "test_flash_attn_func_attention_sinks",
        "test_flash_attn_func_attention_sinks_head_pairs",
        "test_attention_sinks_are_forward_only",
        "test_attention_sinks_far_above_the_scores",
        "test_kernel_rejects_misshaped_sinks",
        "test_attention_sinks_after_guard_recovery",
        "test_traced_sinks_compile_once",
    ),
]
# Note: the paged prefill is flash_attn_varlen_func with a block table, so
# test_paged.py tests varlen as well as the kvcache.
VARLEN = [
    *functions(PREFILL, "test_flash_attn_varlen_output"),
    *functions(
        REGRESSION,
        "test_flash_attn_varlen_func_feeds_the_program",
        "test_flash_attn_varlen_padding_rows_are_zero",
        "test_flash_attn_varlen_ragged_last_block",
        "test_flash_attn_varlen_attention_sinks",
        "test_attention_minus_inf_sinks_are_no_sink",
        "test_flash_attn_varlen_attention_sinks_gpt_oss",
        "test_varlen_padding_rows_ignore_sinks",
        "test_flash_attn_varlen_guard_replay_unroll",
        "test_guard_replay_unroll_rejects_bad_values",
        "test_flash_attn_varlen_clamped_window_first_pass",
    ),
    PAGED,
]
KVCACHE = [
    *functions(PREFILL, "test_flash_attn_kvcache"),
    *functions(
        REGRESSION,
        "test_decode_attention_sinks_after_guard_recovery",
        "test_flash_attn_with_kvcache_recovers_when_later_block_changes_anchor",
        "test_kvcache_compute_fragments_handle_masked_prefix_and"
        "_boundary_append",
        "test_kvcache_contiguous_cache_across_compute_fragments",
        "test_kvcache_contiguous_cache_attention_sinks",
        "test_flash_attn_with_kvcache_appends_step_after_step",
        "test_flash_attn_with_kvcache_block_table_changes_under_one_executable",
        "test_flash_attn_with_kvcache_pads_block_table_to_whole_blocks",
    ),
    PAGED,
    f"{SOFTMAX}/test_decode_packed.py",
    f"{SOFTMAX}/test_dispatch.py",
    PAGE,
]
KV_WRITE = [PAGE, f"{TESTS_DIR}/test_internal_exports.py"]
GDN = [f"{TESTS_DIR}/linear_attention/test_gdn.py"]
KDA = [f"{TESTS_DIR}/linear_attention/test_kda.py"]
SOFTMAX_GROUPS = (DENSE, VARLEN, KVCACHE)
# The tests of each api mlir_snapshot.py's configurations name.
API_TESTS = {
    "flash_attn_func": DENSE,
    "_flash_attn_func_pcp_causal_offset": DENSE,
    "jax.vjp(flash_attn_func)": DENSE,
    "flash_attn_varlen_func": VARLEN,
    "flash_attn_with_kvcache": KVCACHE,
    "write_kv_cache_pages": KV_WRITE,
    "flash_mla_decode": [f"{SOFTMAX}/test_mla_decode.py"],
    "flash_mla_prefill": [f"{SOFTMAX}/test_mla_prefill.py"],
    "mla_cache_write": [f"{SOFTMAX}/test_mla_cache_write.py"],
    "fused_conv1d_gdn": GDN,
    "fused_conv1d_gdn_replay": GDN,
    "fused_conv1d_kda": KDA,
    "fused_conv1d_kda_replay": KDA,
}


def test_functions(path: str) -> list[str]:
    """The module-level test functions of a test file."""
    tree = ast.parse((REPO_ROOT / path).read_text("utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]


def unassigned_functions() -> list[str]:
    """The test functions of MIXED_FILES that no group names."""
    named = {test for group in SOFTMAX_GROUPS for test in group}
    return [
        test
        for path in MIXED_FILES
        for test in functions(path, *test_functions(path))
        if test not in named
    ]


def unmapped_test_files() -> list[str]:
    """The test files under tests/ that API_TESTS names neither whole nor by
    function."""
    named = {
        test.split("::")[0] for tests in API_TESTS.values() for test in tests
    }
    return sorted(
        path
        for path in (
            str(file.relative_to(REPO_ROOT))
            for file in (REPO_ROOT / TESTS_DIR).rglob("test_*.py")
        )
        if path not in named
    )


def module_name(path: pathlib.Path) -> str:
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def imported_modules(path: pathlib.Path) -> set[str]:
    """The modules a file imports, with every name it imports from one, since
    `from tests.a import b` may import the module tests.a.b."""
    tree = ast.parse(path.read_text("utf-8"))
    package = module_name(path).rsplit(".", 1)[0]
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0]
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


def importing_test_files(changed: set[str]) -> set[str]:
    """The test files that are, or import directly or through other modules
    under tests/, a changed module under tests/."""
    files = sorted((REPO_ROOT / TESTS_DIR).rglob("*.py"))
    imports = {module_name(path): imported_modules(path) for path in files}
    reached = {module_name(REPO_ROOT / path) for path in changed}
    while True:
        more = {
            module
            for module, names in imports.items()
            if module not in reached and names & reached
        }
        if not more:
            break
        reached |= more
    return {
        str(path.relative_to(REPO_ROOT))
        for path in files
        if path.name.startswith("test_") and module_name(path) in reached
    }


def select(
    changed_configs: dict[str, str], changed_files: list[str]
) -> tuple[list[str], list[str]]:
    """The pytest arguments to run, and the Markdown lines saying why."""
    unknown = sorted(set(changed_configs.values()) - set(API_TESTS))
    if unknown:
        raise ValueError(
            f"No tests for api {unknown}: add them to API_TESTS in"
            f" {pathlib.Path(__file__).relative_to(REPO_ROOT)}."
        )
    unmapped = unmapped_test_files()
    if unmapped:
        raise ValueError(
            f"No api in API_TESTS names {unmapped}: add each to the tests of"
            " the api it tests, in"
            f" {pathlib.Path(__file__).relative_to(REPO_ROOT)}, so that pull"
            " requests that change that api run it."
        )
    unassigned = unassigned_functions()
    selected = set()
    lines = []
    by_api = {}
    for name, api in sorted(changed_configs.items()):
        by_api.setdefault(api, []).append(name)
    for api, names in sorted(by_api.items()):
        tests = API_TESTS[api]
        if any(tests is group for group in SOFTMAX_GROUPS):
            tests = [*tests, *unassigned]
        selected.update(tests)
        lines.append(
            f"- `{api}`: the IR of {', '.join(f'`{n}`' for n in names)}"
            " changed."
        )
    changed_tests = {
        path
        for path in changed_files
        if path.startswith(f"{TESTS_DIR}/")
        and path.endswith(".py")
        and (REPO_ROOT / path).exists()
    }
    by_file = sorted(importing_test_files(changed_tests))
    if by_file:
        selected.update(by_file)
        lines.append(
            "- Test files changed or importing a changed module: "
            + ", ".join(f"`{path}`" for path in by_file)
            + "."
        )
    if unassigned and by_api:
        lines.append(
            "- Test functions no group in tpu_tests.py names, run with every"
            " softmax group: "
            + ", ".join(f"`{test}`" for test in unassigned)
            + "."
        )
    # Note: a test function of a file that runs whole would run twice.
    whole_files = {test for test in selected if "::" not in test}
    tests = sorted(
        test
        for test in selected
        if "::" not in test or test.split("::")[0] not in whole_files
    )
    return tests, lines


def run_select(args: argparse.Namespace) -> None:
    report = json.loads(args.changed_configs.read_text("utf-8"))
    if not report["comparable"]:
        raise ValueError(
            "The two snapshots were taken in different environments, so no"
            " IR is comparable."
        )
    changed_files = args.changed_files.read_text("utf-8").split()
    tests, lines = select(report["configs"], changed_files)
    args.out.write_text("".join(f"{test}\n" for test in tests), "utf-8")
    summary = ["## TPU tests", ""]
    if tests:
        summary += [
            f"{len(tests)} test files or functions run:",
            "",
            *lines,
            "",
            "<details><summary>pytest arguments</summary>",
            "",
            *(f"    {test}" for test in tests),
            "",
            "</details>",
        ]
    else:
        summary.append(
            "No kernel's IR changed and no test file changed, so no TPU test"
            " runs."
        )
    args.summary.write_text("\n".join(summary) + "\n", "utf-8")
    print("\n".join(summary))


def node_id(classname: str, name: str) -> str:
    """A JUnit test case's pytest node id: its classname is the module's
    dotted path, then any test classes."""
    parts = classname.split(".")
    for split in range(len(parts), 0, -1):
        path = REPO_ROOT.joinpath(*parts[:split]).with_suffix(".py")
        if path.exists():
            return "::".join(
                ["/".join(parts[:split]) + ".py", *parts[split:], name]
            )
    return "::".join([classname.replace(".", "/") + ".py", name])


def run_report(args: argparse.Namespace) -> int:
    root = ET.parse(args.junit).getroot()
    outcomes = {}
    seconds = 0.0
    for suite in root.iter("testsuite"):
        seconds += float(suite.get("time", 0))
    for case in root.iter("testcase"):
        # Note: an interrupted run ends with a test case of no name.
        if not case.get("name"):
            continue
        test = node_id(case.get("classname", ""), case.get("name", ""))
        if case.find("failure") is not None or case.find("error") is not None:
            outcomes[test] = "failed"
        elif case.find("skipped") is not None:
            outcomes.setdefault(test, "skipped")
        else:
            outcomes.setdefault(test, "passed")
    failed = sorted(test for test, kind in outcomes.items() if kind == "failed")
    counts = {
        kind: sum(outcome == kind for outcome in outcomes.values())
        for kind in ("passed", "failed", "skipped")
    }
    summary = [
        "## TPU test results",
        "",
        (
            f"{counts['passed']} passed, {counts['failed']} failed,"
            f" {counts['skipped']} skipped in {seconds / 3600:.1f} h."
        ),
    ]
    not_reached = []
    if args.collected is not None:
        collected = [
            line.strip()
            for line in args.collected.read_text("utf-8").splitlines()
            if "::" in line
        ]
        not_reached = [test for test in collected if test not in outcomes]
        if not_reached:
            summary.append(
                f"{len(not_reached)} of {len(collected)} collected tests were"
                " not reached."
            )
    if failed:
        summary += ["", "Failed:", ""]
        summary += [f"- `{test}`" for test in failed[:MAX_LISTED]]
        if len(failed) > MAX_LISTED:
            summary.append(f"- and {len(failed) - MAX_LISTED} more")
    if not_reached:
        files = {}
        for test in not_reached:
            path = test.split("::")[0]
            files[path] = files.get(path, 0) + 1
        summary += ["", "Not reached, by file:", ""]
        summary += [f"- `{path}`: {count}" for path, count in files.items()]
    args.summary.write_text("\n".join(summary) + "\n", "utf-8")
    print("\n".join(summary))
    return 1 if failed or not_reached else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser(
        "select", help="write the pytest arguments of a pull request's tests"
    )
    select_parser.add_argument(
        "--changed-configs", type=pathlib.Path, required=True
    )
    select_parser.add_argument(
        "--changed-files",
        type=pathlib.Path,
        required=True,
        help="the paths the pull request changes, one per line",
    )
    select_parser.add_argument("--out", type=pathlib.Path, required=True)
    select_parser.add_argument("--summary", type=pathlib.Path, required=True)
    report_parser = subparsers.add_parser(
        "report", help="summarize a JUnit file as Markdown"
    )
    report_parser.add_argument("--junit", type=pathlib.Path, required=True)
    report_parser.add_argument("--collected", type=pathlib.Path)
    report_parser.add_argument("--summary", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        run_select(args)
    else:
        sys.exit(run_report(args))


if __name__ == "__main__":
    main()
