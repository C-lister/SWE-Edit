import json
import os

_depth = 0
_roots = []
_enabled = []


def pytest_configure(config):
    owner = os.environ.get("CONTINUOUS_SWE_REPORT_OWNER")
    config._continuous_swe_observe = owner is None or owner == str(os.getpid()) or hasattr(config, "workerinput")
    if owner is None:
        os.environ["CONTINUOUS_SWE_REPORT_OWNER"] = str(os.getpid())


def pytest_sessionstart(session):
    global _depth
    _depth += 1
    _roots.append(os.path.abspath(str(session.config.rootdir)))
    _enabled.append(session.config._continuous_swe_observe)


def _record(event):
    # Exclude nested fixture sessions and unrelated child pytest invocations.
    if _depth != 1 or not _enabled[0]:
        return
    repo = os.environ.get("CONTINUOUS_SWE_REPO_ROOT")
    if repo and event.get("nodeid") and _roots:
        filename, separator, test = event["nodeid"].partition("::")
        relative = os.path.relpath(os.path.join(_roots[0], filename), repo)
        if relative != os.pardir and not relative.startswith(os.pardir + os.sep):
            event["repo_nodeid"] = relative.replace(os.sep, "/") + separator + test
    path = os.environ.get("CONTINUOUS_SWE_REPORT")
    if path:
        data = (json.dumps(event) + "\n").encode("utf-8")
        fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)


def pytest_runtest_logreport(report):
    if report.failed:
        status = "FAILED" if report.when == "call" else "ERROR"
    elif report.skipped:
        status = "XFAIL" if hasattr(report, "wasxfail") else "SKIPPED"
    elif report.when == "call" and report.passed:
        status = "XPASS" if hasattr(report, "wasxfail") else "PASSED"
    else:
        return
    _record({"kind": "test", "nodeid": report.nodeid, "status": status, "when": report.when})


def pytest_collectreport(report):
    if report.failed:
        _record({"kind": "collection_error", "nodeid": report.nodeid})


def pytest_sessionfinish(session, exitstatus):
    global _depth
    _record({"kind": "session", "exitstatus": int(exitstatus), "collected": session.testscollected})
    _depth -= 1
    _roots.pop()
    _enabled.pop()
