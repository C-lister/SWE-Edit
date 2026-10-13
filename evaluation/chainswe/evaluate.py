"""Score model patches with strict sequence validation.

Sequential mode validates cumulative model patches. Interleaved mode validates
prior gold fixes followed by the current model patch in a fresh container.
Plain git apply, bug reproduction, explicit test coverage, and regression checks
are included here. The read-only pytest observer lives in test_reporter.py
alongside this file.

Requires the docker and unidiff Python packages, the Docker CLI, and a running
Docker daemon. Keep evaluate.py and test_reporter.py together; no curation
modules are required.
"""

import argparse
import copy
import hashlib
import json
import logging
import re
import shlex
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import docker
from unidiff import PatchSet

EVALUATION_DIR = Path(__file__).resolve().parent

VALIDATOR_VERSION = "strict-sequential-v16"
CONTAINER_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTEST_XDIST_AUTO_NUM_WORKERS": "2",
}
# Load the neighboring plugin once so the fingerprint matches the code installed in containers.
TEST_REPORTER_SOURCE = (EVALUATION_DIR / "test_reporter.py").read_text(encoding="utf-8")
EVALUATOR_VERSION = "strict-model-sequences-v2-standalone"
EVALUATOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
VALIDATOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes() + TEST_REPORTER_SOURCE.encode("utf-8")).hexdigest()


class ValidationFailure(Exception):
    def __init__(self, stage, reason, **details):
        super().__init__(reason)
        self.stage, self.reason, self.details = stage, reason, details


def setup_logger(log_dir="logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("filter")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    for handler in (logging.FileHandler(Path(log_dir) / "filter.log"), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def dexec(container, cmd, timeout=600, stdin_data=None, workdir=None):
    full = ["docker", "exec"]
    if workdir:
        full += ["-w", workdir]
    if stdin_data is not None:
        full.append("-i")
    full += [container.id, "bash", "-c", cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout, input=stdin_data)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as exc:

        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

        return 124, decode(exc.stdout), decode(exc.stderr) + "\nTIMEOUT"


def checked_exec(container, cmd, stage, workdir=None, timeout=600):
    ec, out, err = dexec(container, cmd, timeout, workdir=workdir)
    if ec:
        raise ValidationFailure(stage, "command_failed", command=cmd, exit_code=ec, output=(out + err)[-6000:])
    return out


def write_to_container(container, content, path):
    return dexec(container, "cat > " + shlex.quote(path), stdin_data=content)[0] == 0


def pull_and_start_container(client, image, run_id, logger):
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        logger.info("Pulling %s", image)
        client.images.pull(image)
    container = client.containers.create(
        image=image,
        name=f"swe-filter-{run_id[:35]}-{uuid.uuid4().hex[:8]}",
        user="root",
        detach=True,
        entrypoint="",
        command=["tail", "-f", "/dev/null"],
        mem_limit="6g",
        shm_size="1g",
        nano_cpus=2_000_000_000,
        pids_limit=1024,
        environment=CONTAINER_ENVIRONMENT,
        labels={"continuous-swe.validator": VALIDATOR_VERSION},
    )
    try:
        container.start()
    except Exception:
        container.remove(force=True)
        raise
    return container


def detect_repo_dir(client, image, test_cmds, source):
    match = re.search(r"(?:^|&&)\s*cd\s+(/[^\s&|;]+)", test_cmds or "")
    if match:
        return match.group(1)
    wd = client.images.get(image).attrs.get("Config", {}).get("WorkingDir")
    return wd or ("/app" if source == "swebench-pro" else "/testbed")


def apply_patch(container, patch_text, patch_path, label, bug_id, repo_dir, logger):
    """Apply exactly the supplied patch. No fallbacks or implicit repairs."""
    if not patch_text:
        return True, None
    if not write_to_container(container, patch_text, patch_path):
        return False, "could not write patch"
    ec, out, err = dexec(container, "git apply " + shlex.quote(patch_path), workdir=repo_dir)
    if ec:
        logger.warning("%s %s: %s", bug_id, label, (err or out)[-1500:])
    return ec == 0, err or out


def canonical_test_id(test_id):
    test_id = test_id.strip().removeprefix("./")
    match = re.fullmatch(r"(\w+) \(([^)]+)\)", test_id)
    if match:
        name, owner = match.groups()
        return owner if owner.endswith("." + name) else owner + "." + name
    return test_id


PRIORITY = {"PASSED": 0, "XPASS": 0, "SKIPPED": 1, "XFAIL": 1, "FAILED": 3, "ERROR": 4}
PASS_STATUSES = {"PASSED", "XPASS"}


def add_result(results, test_id, status):
    key = canonical_test_id(test_id)
    if PRIORITY[status] >= PRIORITY.get(results.get(key), -1):
        results[key] = status


def classify_unittest_errors(output, results):
    """Recognize test-body exceptions without accepting fixture failures."""
    evidence = {}
    lifecycle = {
        "setUp", "tearDown", "setUpClass", "tearDownClass",
        "setUpModule", "tearDownModule", "asyncSetUp", "asyncTearDown",
        "_callSetUp", "_callTearDown", "_callCleanup", "doCleanups",
        "doClassCleanups", "doModuleCleanups",
    }
    for block in re.split(r"^={6,}\s*$", output, flags=re.MULTILINE):
        header = re.search(r"^ERROR: (\w+) \(([^)]+)\)(?: .*)?$", block, re.MULTILINE)
        if header is None:
            continue
        name, owner = header.groups()
        key = canonical_test_id(f"{name} ({owner})")
        frames = re.findall(r'^  File "[^"]+", line \d+, in (\w+)\s*$', block, re.MULTILINE)
        body_error = (
            "Traceback (most recent call last):" in block
            and name in frames
            and not lifecycle.intersection(frames)
        )
        evidence.setdefault(key, []).append(body_error)
    for key, body_errors in evidence.items():
        # A test can fail in both its body and teardown. Every error must be
        # accounted for as a body exception before it can be a bug failure.
        if results.get(key) == "ERROR" and all(body_errors):
            results[key] = "FAILED"


def parse_test_results(output):
    """Fallback for verbose non-pytest runners; absent tests remain absent."""
    results = {}
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    unittest_statuses = {
        "ok": "PASSED", "FAIL": "FAILED", "ERROR": "ERROR",
        "expected failure": "XFAIL", "unexpected success": "XPASS",
    }
    descriptions = {}
    description_lines = set()
    # TextTestRunner prints a test's short description on a second line when
    # it has a docstring. Preserve its full identity and the emitted description
    # used by some benchmark declarations. A shared description is evidence
    # only when every test that printed it has the same observed status.
    for match in re.finditer(
        r"^(\w+ \([^)]+\))\n([^\n]+?)\s+\.\.\.\s+"
        r"(ok|FAIL|ERROR|skipped[^\n]*|expected failure|unexpected success)$",
        output,
        re.MULTILINE,
    ):
        test_id, description, raw = match.groups()
        key = canonical_test_id(test_id)
        add_result(results, key, unittest_statuses.get(raw, "SKIPPED"))
        descriptions.setdefault(description.strip(), set()).add(key)
        description_lines.add(match.group(0).splitlines()[1].strip())
    for line in output.splitlines():
        line = line.strip()
        if line in description_lines:
            continue
        # unittest's aggregate status is not an individual test result.
        if re.fullmatch(
            r"FAILED \((?:failures|errors|skipped|expected failures|unexpected successes)=\d+"
            r"(?:, (?:failures|errors|skipped|expected failures|unexpected successes)=\d+)*\)",
            line,
        ):
            continue
        match = re.match(r"^(PASSED|FAILED|ERROR) (.+)$", line)
        if match:
            status, test_id = match.groups()
            depth = 0
            for offset, char in enumerate(test_id):
                if depth == 0 and test_id.startswith(" - ", offset):
                    test_id = test_id[:offset]
                    break
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth = max(0, depth - 1)
            add_result(results, test_id, status)
            continue
        match = re.match(
            r"^(\w+ \([^)]+\))\s+.*?\.\.\.\s+(ok|FAIL|ERROR|skipped.*|expected failure|unexpected success)$", line
        )
        if match:
            test_id, raw = match.groups()
            status = unittest_statuses.get(raw, "SKIPPED")
            add_result(results, test_id, status)
            continue
        match = re.match(r"^(.+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s+\[.*\])?$", line)
        if match:
            test_id, status = match.groups()
            add_result(results, test_id, status)
    classify_unittest_errors(output, results)
    for description, owners in descriptions.items():
        statuses = {results[owner] for owner in owners}
        if len(statuses) == 1:
            add_result(results, description, statuses.pop())
    return results


def parse_failing_tests(output):
    return [key for key, value in parse_test_results(output).items() if value in ("FAILED", "ERROR")]


def read_observer_report(text):
    results, errors, sessions = {}, [], []
    for line in text.splitlines():
        event = json.loads(line)
        if event["kind"] == "collection_error":
            errors.append(event["nodeid"])
        elif event["kind"] == "session":
            sessions.append(event)
        elif event["kind"] == "test":
            add_result(results, event["nodeid"], event["status"])
            # pytest may select a nested rootdir. This exact alias comes from
            # the observer's rootdir and repository path, never fuzzy matching.
            if event.get("repo_nodeid"):
                add_result(results, event["repo_nodeid"], event["status"])
    return results, errors, sessions


def run_tests(container, bug, repo_dir, args, audit_dir, phase):
    report_path = f"/tmp/swe-results-{uuid.uuid4().hex}.jsonl"
    if not write_to_container(container, TEST_REPORTER_SOURCE, "/tmp/continuous_swe_test_reporter.py"):
        raise ValidationFailure(phase, "observer_install_failed")
    cmd = bug["test_cmds"]
    # Read-only observer: no changes to selection, assertions, skips or exits.
    wrapped = (
        f"export CONTINUOUS_SWE_REPORT={shlex.quote(report_path)}; "
        f"export CONTINUOUS_SWE_REPO_ROOT={shlex.quote(repo_dir)}; "
        'export PYTHONPATH="/tmp${PYTHONPATH:+:$PYTHONPATH}"; '
        'export PYTEST_PLUGINS="continuous_swe_test_reporter${PYTEST_PLUGINS:+,$PYTEST_PLUGINS}"; ' + cmd
    )
    head = checked_exec(container, "git rev-parse HEAD", phase, repo_dir).strip()
    ec, stdout, stderr = dexec(container, wrapped, timeout=args.test_timeout, workdir=repo_dir)
    output = stdout + "\n" + stderr
    (audit_dir / f"{phase}.log").write_text(output)
    rc, observed, _ = dexec(container, "cat " + shlex.quote(report_path), workdir=repo_dir)
    if rc == 0:
        (audit_dir / f"{phase}.events.jsonl").write_text(observed)
        results, errors, sessions = read_observer_report(observed)
        if any(s["exitstatus"] not in (0, 1) for s in sessions):
            errors.append("pytest session did not complete normally")
    else:
        results, errors, sessions = parse_test_results(output), [], []
    run = {
        "phase": phase,
        "command": cmd,
        "exit_code": ec,
        "results": results,
        "collection_errors": errors,
        "observer": rc == 0,
    }
    (audit_dir / f"{phase}.json").write_text(json.dumps(run, indent=2) + "\n")
    if ec == 124:
        raise ValidationFailure(phase, "test_timeout")
    if rc == 0 and not sessions:
        raise ValidationFailure(phase, "incomplete_test_session", exit_code=ec)
    if errors:
        raise ValidationFailure(phase, "collection_error", errors=errors[:20])
    if not results:
        raise ValidationFailure(phase, "no_per_test_evidence", exit_code=ec)
    if any(v == "ERROR" for v in results.values()):
        raise ValidationFailure(phase, "test_setup_or_teardown_error")
    after = checked_exec(container, "git rev-parse HEAD", phase, repo_dir).strip()
    # Untracked files inside submodules are not changes to tracked source.
    # Keep detecting modified tracked content and changed submodule commits.
    changed = checked_exec(
        container, "git diff HEAD --ignore-submodules=untracked --name-only", phase, repo_dir
    ).strip()
    if after != head or changed:
        raise ValidationFailure(phase, "tests_modified_tracked_repository", files=changed)
    return run


def required_test_members(results, test):
    """Resolve only the benchmark's whitespace-truncated pytest parameter IDs.

    Some source reports split node IDs at whitespace. Match that exact transform,
    and require the whole group; never select an arbitrary matching parameter.
    Complete IDs and non-parameterized names still require exact identity.
    """
    key = canonical_test_id(test)
    if key in results:
        return [key]
    # Parameter values themselves can contain pytest's :: separator.
    parameter_id = key.split("::", 1)[-1]
    if (
        "::" in key
        and "[" in parameter_id
        # A trailing bracket can close a literal value such as [foo], while
        # the outer parameter ID was still truncated at the following space.
        and (not key.endswith("]") or parameter_id.count("[") > parameter_id.count("]"))
        and key.split() == [key]
    ):
        return sorted(
            node for node in results
            if node.endswith("]") and node.split(maxsplit=1)[0] == key
        )
    return []


def required_tests_passed(results, tests):
    return all(
        (members := required_test_members(results, test))
        and all(results[node] in PASS_STATUSES for node in members)
        for test in tests
    )


def observed_tests_passed(results, tests):
    return all(results.get(test) in PASS_STATUSES for test in tests)


def require_observed_passed(results, tests, phase):
    bad = {test: results.get(test, "MISSING") for test in tests if results.get(test) not in PASS_STATUSES}
    if bad:
        raise ValidationFailure(phase, "observed_tests_not_passed", tests=dict(list(bad.items())[:50]), count=len(bad))


def require_status(results, tests, expected, phase):
    # Non-strict XPASS means the test actually passed. Strict XPASS is reported
    # as FAILED by pytest and also gives the command a failing exit status.
    accepted = PASS_STATUSES if expected == "PASSED" else {expected}
    bad, expansions = {}, {}
    for test in tests:
        members = required_test_members(results, test)
        if not members:
            bad[test] = "MISSING"
            continue
        if members != [canonical_test_id(test)]:
            expansions[test] = members
        rejected = {node: results[node] for node in members if results[node] not in accepted}
        if rejected:
            bad[test] = rejected if test in expansions else next(iter(rejected.values()))
    if bad:
        raise ValidationFailure(
            phase, "required_tests_not_" + expected.lower(), tests=dict(list(bad.items())[:50]), count=len(bad)
        )
    return expansions


def commit(container, message, repo_dir, patch_text):
    # Preserve untracked test outputs as untracked. Only the supplied patch's
    # files belong in this commit, including additions, deletions and renames.
    paths = sorted(
        {
            path.split("/", 1)[-1]
            for file in PatchSet(patch_text)
            for path in (file.source_file, file.target_file)
            if path and path != "/dev/null"
        }
    )
    if paths:
        checked_exec(container, "git add -A -- " + shlex.join(paths), "commit", repo_dir)
    checked_exec(
        container,
        "git -c core.hooksPath=/dev/null commit --allow-empty -m " + shlex.quote(message),
        "commit",
        repo_dir,
    )


def sequence_fingerprint(seq):
    return hashlib.sha256(json.dumps(seq, sort_keys=True).encode()).hexdigest()


def validator_fingerprint():
    return VALIDATOR_SHA256


def process_sequence(seq, client, logger, args):
    cid = seq["continuous_id"]
    audit_dir = Path(args.audit_dir) / cid
    audit_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "continuous_id": cid,
        "base_commit": seq["base_commit"],
        "input_sha256": sequence_fingerprint(seq),
        "validator_version": VALIDATOR_VERSION,
        "validator_sha256": validator_fingerprint(),
        "container_resources": {"memory": "6g", "shared_memory": "1g", "cpus": 2},
        "container_environment": CONTAINER_ENVIRONMENT,
        "started_at": datetime.now(UTC).isoformat(),
        "steps": [],
    }
    container = None
    try:
        if not seq.get("docker_image") or not seq.get("bug_fixes"):
            raise ValidationFailure("metadata", "missing_image_or_bugs")
        container = pull_and_start_container(client, seq["docker_image"], cid, logger)
        repo_dir = detect_repo_dir(
            client, seq["docker_image"], seq["bug_fixes"][0].get("test_cmds", ""), seq.get("source")
        )
        report["repo_dir"] = repo_dir
        checked_exec(container, "git config user.email filter@swe && git config user.name Filter", "setup", repo_dir)
        checked_exec(
            container, "git reset --hard " + shlex.quote(seq["base_commit"]) + " && git clean -fd", "setup", repo_dir
        )
        updated_bugs, observed_by_bug = [], []
        for i, original in enumerate(seq["bug_fixes"]):
            bug = dict(original)
            bug_id = bug["swebench_instance_id"]
            report["current_bug"] = bug_id
            if not bug.get("patch") or not bug.get("test_cmds"):
                raise ValidationFailure(f"{i}-metadata", "missing_gold_patch_or_test_command")
            for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                if not isinstance(bug.get(key, []), list):
                    raise ValidationFailure(f"{i}-metadata", "invalid_test_list", field=key)
            ok, error = apply_patch(
                container, bug.get("test_patch", ""), f"/tmp/test-{i}.patch", "test_patch", bug_id, repo_dir, logger
            )
            if not ok:
                raise ValidationFailure(f"{i}-test_patch", "patch_does_not_apply", error=error)
            commit(container, f"test-{i}-{bug_id}", repo_dir, bug.get("test_patch", ""))
            before = run_tests(container, bug, repo_dir, args, audit_dir, f"{i}-before")
            if before["exit_code"] == 0:
                raise ValidationFailure(f"{i}-before", "bug_not_reproduced")
            failed = [t for t, status in before["results"].items() if status == "FAILED"]
            if not failed:
                raise ValidationFailure(f"{i}-before", "no_failing_test_evidence")
            declared_f2p, declared_p2p = bug.get("FAIL_TO_PASS", []), bug.get("PASS_TO_PASS", [])
            f2p_expansions = require_status(before["results"], declared_f2p, "FAILED", f"{i}-before-F2P")
            p2p_expansions = require_status(before["results"], declared_p2p, "PASSED", f"{i}-before-P2P")
            ok, error = apply_patch(
                container, bug["patch"], f"/tmp/gold-{i}.patch", "gold_patch", bug_id, repo_dir, logger
            )
            if not ok:
                raise ValidationFailure(f"{i}-gold_patch", "patch_does_not_apply", error=error)
            commit(container, f"gold-{i}-{bug_id}", repo_dir, bug["patch"])
            after = run_tests(container, bug, repo_dir, args, audit_dir, f"{i}-after")
            if after["exit_code"] != 0 or any(v in ("FAILED", "ERROR") for v in after["results"].values()):
                raise ValidationFailure(f"{i}-after", "test_command_failed", exit_code=after["exit_code"])
            # Derive missing labels from actual transitions; never drop declared tests.
            f2p = sorted(set(declared_f2p) | set(failed))
            p2p = sorted(set(declared_p2p) | {t for t, v in before["results"].items() if v in PASS_STATUSES})
            require_status(after["results"], f2p + p2p, "PASSED", f"{i}-after-required")
            observed = [t for t, v in before["results"].items() if v == "FAILED" or v in PASS_STATUSES]
            require_observed_passed(after["results"], observed, f"{i}-after-observed")
            bug["FAIL_TO_PASS"], bug["PASS_TO_PASS"] = f2p, p2p
            regression_runs = []
            for j, prior in enumerate(updated_bugs):
                required = prior["FAIL_TO_PASS"] + prior["PASS_TO_PASS"]
                if required_tests_passed(after["results"], required) and observed_tests_passed(
                    after["results"], observed_by_bug[j]
                ):
                    regression_runs.append({"bug": j, "covered_by": f"{i}-after"})
                    continue
                regression = run_tests(container, prior, repo_dir, args, audit_dir, f"{i}-regression-{j}")
                if regression["exit_code"] != 0 or any(
                    v in ("FAILED", "ERROR") for v in regression["results"].values()
                ):
                    raise ValidationFailure(f"{i}-regression-{j}", "earlier_test_command_failed")
                require_status(regression["results"], required, "PASSED", f"{i}-regression-{j}")
                require_observed_passed(regression["results"], observed_by_bug[j], f"{i}-regression-{j}")
                regression_runs.append({"bug": j, "covered_by": f"{i}-regression-{j}"})
            report["steps"].append(
                {"bug": bug_id, "f2p_count": len(f2p), "p2p_count": len(p2p), "regression_runs": regression_runs,
                 "test_id_expansions": {"FAIL_TO_PASS": f2p_expansions, "PASS_TO_PASS": p2p_expansions}}
            )
            updated_bugs.append(bug)
            observed_by_bug.append(observed)
            logger.info("%s step %s/%s validated", cid, i + 1, len(seq["bug_fixes"]))
        validated = dict(seq, bug_fixes=updated_bugs)
        validated.pop("validation", None)
        report.update(status="passed", output_sha256=sequence_fingerprint(validated))
        validated["validation"] = {
            "validator_version": VALIDATOR_VERSION,
            "validator_sha256": report["validator_sha256"],
            "input_sha256": report["input_sha256"],
            "output_sha256": report["output_sha256"],
            "audit_path": str(audit_dir / "report.json"),
        }
        return "passed", validated
    except ValidationFailure as exc:
        report.update(status="skipped", stage=exc.stage, reason=exc.reason, details=exc.details)
        logger.warning("%s: %s / %s", cid, exc.stage, exc.reason)
        return "skipped", report
    except Exception as exc:
        report.update(status="error", reason=type(exc).__name__, details={"error": str(exc)})
        logger.exception("%s failed", cid)
        return "skipped", report
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        (audit_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                logger.exception("Could not remove validation container %s", container.id)


def load_jsonl(path):
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]



def evaluation_identity(seq, args):
    """Identify the prediction, policy, mode, and timeout behind a result."""
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_sha256": EVALUATOR_SHA256,
        "validator_version": VALIDATOR_VERSION,
        "validator_sha256": validator_fingerprint(),
        "mode": args.mode,
        "test_timeout": args.test_timeout,
        "prediction_sha256": sequence_fingerprint(seq),
    }


def _run_validation(seq, client, logger, args, run_dir):
    opts = SimpleNamespace(test_timeout=args.test_timeout, audit_dir=run_dir)
    process_sequence(seq, client, logger, opts)
    # process_sequence writes the report, including its finish timestamp, for
    # success and failure. Keep the raw phase reports and observer events too.
    report_path = run_dir / seq["continuous_id"] / "report.json"
    return json.loads(report_path.read_text()), report_path


def _unmet(results, labels, expected):
    unmet = []
    for label in labels:
        members = required_test_members(results, label)
        if not members or any(results[node] not in expected for node in members):
            unmet.append(label)
    return unmet


def _result_from_report(seq, index, report, report_path):
    bug = seq["bug_fixes"][index]
    completed = len(report["steps"])
    row = {
        "instance_id": bug["swebench_instance_id"],
        "bug_index": index,
        "resolved": index < completed,
        "test_exit_code": None,
        "f2p_broken": [],
        "p2p_broken": [],
        "other_failures": [],
        "audit_path": str(report_path),
    }
    if row["resolved"]:
        row["test_exit_code"] = 0
        row["validation_step"] = report["steps"][index]
        return row

    row.update(
        reason=report.get("reason", "validation_failed"),
        failure_stage=report.get("stage", "validation"),
        details=report.get("details", {}),
    )
    if index > completed:
        row["reason"] = "prior_step_failed"
        row["blocked_by"] = seq["bug_fixes"][completed]["swebench_instance_id"]
        return row

    stage = row["failure_stage"]
    before = stage.startswith(f"{index}-before")
    phase = f"{index}-before" if before else f"{index}-after"
    checked_bug = bug
    regression = re.fullmatch(rf"{index}-regression-(\d+)", stage)
    if regression:
        phase = stage
        checked_bug = seq["bug_fixes"][int(regression.group(1))]
        row["regression_of"] = checked_bug["swebench_instance_id"]
    run_path = report_path.parent / f"{phase}.json"
    if run_path.exists():
        run = json.loads(run_path.read_text())
        row["test_exit_code"] = run["exit_code"]
        results = run["results"]
        row["f2p_broken"] = _unmet(
            results,
            checked_bug.get("FAIL_TO_PASS", []),
            {"FAILED"} if before else PASS_STATUSES,
        )
        row["p2p_broken"] = _unmet(results, checked_bug.get("PASS_TO_PASS", []), PASS_STATUSES)
        required = {
            node
            for label in checked_bug.get("FAIL_TO_PASS", []) + checked_bug.get("PASS_TO_PASS", [])
            for node in required_test_members(results, label)
        }
        row["other_failures"] = sorted(
            node for node, status in results.items() if status in {"FAILED", "ERROR"} and node not in required
        )[:20]
    return row


def evaluate_sequence(seq, client, logger, args):
    """Return a result for every bug, including steps blocked by an earlier failure."""
    if args.mode not in {"sequential", "interleaved"}:
        raise ValueError(f"Unknown evaluation mode: {args.mode}")
    bugs = seq["bug_fixes"]
    if not bugs:
        raise ValueError("Sequence has no bug fixes")
    identity = evaluation_identity(seq, args)
    run_dir = Path(args.audit_dir) / f"{args.mode}-{identity['prediction_sha256'][:12]}-{uuid.uuid4().hex}"
    candidate = copy.deepcopy(seq)
    candidate.pop("validation", None)

    if args.mode == "sequential":
        for bug in candidate["bug_fixes"]:
            bug["patch"] = bug.get("model_patch", "")
        report, path = _run_validation(candidate, client, logger, args, run_dir)
        rows = [_result_from_report(candidate, i, report, path) for i in range(len(bugs))]
    else:
        rows = []
        for i in range(len(bugs)):
            # Prior steps use their gold patches and undergo the complete
            # validator, including their own before/after and regression tests.
            # A prior failure prevents scoring the current patch as resolved.
            prefix = copy.deepcopy(candidate)
            prefix["bug_fixes"] = prefix["bug_fixes"][: i + 1]
            prefix["bug_fixes"][i]["patch"] = bugs[i].get("model_patch", "")
            report, path = _run_validation(prefix, client, logger, args, run_dir / str(i))
            rows.append(_result_from_report(prefix, i, report, path))
    for row in rows:
        row["continuous_id"] = seq["continuous_id"]
        row["evaluation"] = identity
    return rows


def load_previous_results(path, args):
    """Refuse to mix old permissive scores or other policies into this output."""
    previous = {}
    if not path.exists():
        return previous
    policy = evaluation_identity({}, args)
    policy.pop("prediction_sha256")
    for row in load_jsonl(path):
        metadata = row.get("evaluation", {})
        if any(metadata.get(key) != value for key, value in policy.items()):
            raise ValueError(
                f"{path} contains results from a different evaluator, mode, or timeout. "
                "Use a new --output path; those scores cannot be reused for this run."
            )
        previous.setdefault(row["continuous_id"], []).append(row)
    return previous


def sequence_is_complete(seq, rows, args):
    """Resume only a complete set of results for the exact current prediction."""
    bugs = seq["bug_fixes"]
    if not bugs or len(rows) != len(bugs):
        return False
    ordered = sorted(rows, key=lambda row: row.get("bug_index", -1))
    expected = evaluation_identity(seq, args)
    return all(
        row.get("bug_index") == i
        and row.get("instance_id") == bug["swebench_instance_id"]
        and row.get("evaluation") == expected
        and isinstance(row.get("resolved"), bool)
        for i, (row, bug) in enumerate(zip(ordered, bugs, strict=True))
    )


def write_results(path, records):
    """Replace complete sequence groups atomically, including resumed partial runs."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as output:
            for rows in records.values():
                for row in rows:
                    output.write(json.dumps(row) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=EVALUATION_DIR / "outputs/interleaved.jsonl")
    parser.add_argument("--output", type=Path, default=EVALUATION_DIR / "outputs/results.jsonl")
    parser.add_argument("--audit-dir", type=Path, default=EVALUATION_DIR / "audit")
    parser.add_argument("--mode", choices=["sequential", "interleaved"], default="interleaved")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test-timeout", type=int, default=900)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--continuous-id", type=str, default=None)
    args = parser.parse_args()
    if args.workers < 1 or args.test_timeout < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers, --test-timeout, and --limit must be positive")
    logger = setup_logger(args.audit_dir)
    sequences = load_jsonl(args.predictions)
    if len({s["continuous_id"] for s in sequences}) != len(sequences):
        parser.error("Predictions contain duplicate continuous_id values")
    try:
        records = load_previous_results(args.output, args)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    todo = []
    for seq in sequences:
        cid = seq["continuous_id"]
        if args.continuous_id is not None and cid != args.continuous_id:
            continue
        if sequence_is_complete(seq, records.get(cid, []), args):
            continue
        todo.append(seq)
        if args.limit is not None and len(todo) >= args.limit:
            break
    logger.info("Evaluating %s sequences with %s workers (%s)", len(todo), args.workers, args.mode)

    def worker(seq):
        client = docker.from_env(timeout=600)
        try:
            return evaluate_sequence(seq, client, logger, args)
        finally:
            client.close()

    completed = resolved = instances = passed_sequences = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, seq): seq for seq in todo}
        for future in as_completed(futures):
            seq = futures[future]
            try:
                rows = future.result()
            except Exception:
                logger.exception("Evaluation could not complete for %s", seq["continuous_id"])
                records.pop(seq["continuous_id"], None)
                write_results(args.output, records)
                continue
            records[seq["continuous_id"]] = rows
            write_results(args.output, records)
            completed += 1
            instances += len(rows)
            count = sum(row["resolved"] for row in rows)
            resolved += count
            passed_sequences += count == len(rows)
            logger.info("%s: %s/%s bugs resolved", seq["continuous_id"], count, len(rows))
    logger.info(
        "Completed %s/%s sequences; %s/%s bugs resolved; %s sequences fully resolved",
        completed,
        len(todo),
        resolved,
        instances,
        passed_sequences,
    )
    if completed != len(todo):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
