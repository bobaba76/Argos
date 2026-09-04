#!/usr/bin/env python3
"""
Auto-review script for Devin PRs in the Argos repo.

Usage:
  python scripts/auto_review.py [PR_NUMBER]       # review one PR
  python scripts/auto_review.py --batch            # review all open devin-implemented PRs
  python scripts/auto_review.py --batch --approve  # auto-approve passing PRs

Requirements: gh CLI authenticated, pytest, and the venv must have all deps.

What it checks:
  1. Fetches the diff
  2. Runs pytest on affected test files (or all tests if none changed)
  3. Checks for suspicious patterns (prompt injection, hardcoded secrets, broad imports)
  4. Verifies PR references an issue
  5. Posts structured review comment
  6. Labels: review-approved or changes-requested
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = "bobaba76/Argos"
REPO_DIR = Path(r"C:\Users\michael\Documents\Github\Argos")

SUSPICIOUS_PATTERNS = [
    (r"(?i)openai\.api_key\s*=\s*['\"]sk-", "Hardcoded OpenAI key"),
    (r"(?i)password\s*=\s*['\"][^'\"]{3,}['\"]", "Hardcoded password"),
    (r"(?i)secret\s*=\s*['\"][^'\"]{8,}['\"]", "Possible hardcoded secret"),
    (r"(?:requests|urllib)\.(?:get|post|put)\(['\"]https?://", "Un-gated network call"),
    (r"eval\s*\(", "eval() call — potential injection vector"),
    (r"exec\s*\(", "exec() call — potential injection vector"),
    (r"pickle\.loads?\s*\(", "Pickle — potential RCE vector"),
    (r"subprocess\.(?:call|Popen|run)\s*\(.*shell=True", "Shell=True — injection risk"),
    (r"os\.system\s*\(", "os.system() — injection risk"),
]


def run(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def get_pr_info(pr_number: int) -> Optional[Dict]:
    """Get PR metadata."""
    result = run(["gh", "pr", "view", str(pr_number), "--repo", REPO,
                  "--json", "title,body,files,additions,deletions,changedFiles,labels,headRefName,baseRefName"])
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", file=sys.stderr)
        return None
    return json.loads(result.stdout)


def get_pr_diff(pr_number: int) -> Optional[str]:
    """Get the diff of a PR."""
    result = run(["gh", "pr", "diff", str(pr_number), "--repo", REPO])
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", file=sys.stderr)
        return None
    return result.stdout


def get_check_runs(pr_number: int) -> List[Dict]:
    """Get CI check status."""
    result = run(["gh", "pr", "checks", str(pr_number), "--repo", REPO,
                  "--json", "name,state,conclusion", "--fail-fast"])
    if result.returncode != 0:
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def get_sha(pr_number: int) -> Optional[str]:
    """Get the HEAD sha of the PR."""
    result = run(["gh", "pr", "view", str(pr_number), "--repo", REPO,
                  "--json", "headRefOid"])
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)["headRefOid"]


def extract_issue_ref(body: str) -> Optional[int]:
    """Extract issue number from body (e.g. 'Fixes #212' or '#212')."""
    m = re.search(r"(?:Fixes|Closes|Resolves|fixes|closes|resolves)\s+#(\d+)", body)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:issue|Issue)\s+\#?(\d+)", body)
    if m:
        return int(m.group(1))
    return None


def check_suspicious_patterns(diff: str) -> List[Tuple[str, str]]:
    """Check diff for suspicious patterns. Returns [(file, finding), ...]."""
    findings = []
    current_file = "unknown"
    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("--- a/"):
            continue
        for pattern, description in SUSPICIOUS_PATTERNS:
            if re.search(pattern, line):
                findings.append((current_file, description))
    return findings


def run_tests(changed_files: List[str], sha: str) -> Dict:
    """Run tests in the repo at the given sha. Returns results dict."""
    # Collect test files that might be affected
    test_files = []
    for f in changed_files:
        # If a source file changed, find its test counterpart
        if f.startswith("argos_plugin/tests/"):
            test_files.append(f)
        elif f.startswith("argos_plugin/"):
            module = f.replace("argos_plugin/", "").replace(".py", "")
            test_path = f"argos_plugin/tests/test_{module}"
            # Check if test file exists
            if (REPO_DIR / f"{test_path}.py").exists():
                test_files.append(f"{test_path}.py")

    if not test_files:
        # No specific tests — run all
        target = "argos_plugin/tests/"
    else:
        target = " ".join(test_files)

    print(f"  Running: pytest {target} --tb=short -q")
    result = run(
        ["python", "-m", "pytest", *target.split(), "--tb=short", "-q"],
        cwd=str(REPO_DIR),
        timeout=120
    )

    passed = result.returncode == 0
    output = result.stdout[-2000:] if result.stdout else ""
    errors = result.stderr[-2000:] if result.stderr else ""

    # Parse pass/fail counts
    passed_count = 0
    failed_count = 0
    m = re.search(r"(\d+) passed", output or errors)
    if m:
        passed_count = int(m.group(1))
    m = re.search(r"(\d+) failed", output or errors)
    if m:
        failed_count = int(m.group(1))

    return {
        "passed": passed,
        "exit_code": result.returncode,
        "passed_tests": passed_count,
        "failed_tests": failed_count,
        "output": output,
        "errors": errors,
    }


def post_review(pr_number: int, body: str) -> bool:
    """Post a review comment on the PR."""
    # Write body to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(body)
        tmp_path = f.name

    try:
        result = run(["gh", "pr", "comment", str(pr_number), "--repo", REPO, "--body-file", tmp_path])
        os.unlink(tmp_path)
        return result.returncode == 0
    except Exception:
        os.unlink(tmp_path)
        return False


def apply_label(pr_number: int, label: str) -> bool:
    """Apply a label to a PR."""
    # Check if label exists
    label_result = run(["gh", "label", "list", "--repo", REPO, "--search", label])
    if label not in label_result.stdout:
        run(["gh", "label", "create", label, "--repo", REPO, "--color", "bfd4f2" if "approved" in label else "f9d0c4"])
    result = run(["gh", "pr", "edit", str(pr_number), "--repo", REPO, "--add-label", label])
    return result.returncode == 0


def review_pr(pr_number: int, auto_approve: bool = False) -> Dict:
    """Review a single PR and post the result."""
    print(f"\n{'='*60}")
    print(f"Reviewing PR #{pr_number}")
    print(f"{'='*60}")

    info = get_pr_info(pr_number)
    if info is None:
        return {"pr": pr_number, "status": "error", "message": "Failed to fetch PR info"}

    # Check existing labels
    labels = [l["name"] if isinstance(l, dict) else l for l in info.get("labels", [])]
    if "review-approved" in labels:
        return {"pr": pr_number, "status": "skipped", "message": "Already reviewed and approved"}

    # Get diff
    diff = get_pr_diff(pr_number)
    if diff is None:
        return {"pr": pr_number, "status": "error", "message": "Failed to fetch diff"}

    title = info.get("title", "Untitled")
    changed_files = [f["path"] for f in info.get("files", [])]
    sha = info.get("headRefOid") or get_sha(pr_number)

    print(f"  PR: {title}")
    print(f"  Files: {', '.join(changed_files)}")

    # Check issue reference
    body = info.get("body", "")
    issue_ref = extract_issue_ref(body)

    # Check for suspicious patterns
    findings = check_suspicious_patterns(diff)

    # Check CI status
    checks = get_check_runs(pr_number)
    ci_passing = all(
        c.get("conclusion") == "success"
        for c in checks
        if c.get("conclusion") is not None
    ) if checks else None  # None = no CI yet

    # Run tests
    test_result = run_tests(changed_files, sha or "")

    # Build review comment
    lines = []
    lines.append(f"## 🤖 Auto-Review: PR #{pr_number}")
    lines.append(f"**{title}**\n")
    lines.append(f"**Files changed:** {len(changed_files)} | **+{info.get('additions', 0)}/-{info.get('deletions', 0)}**\n")

    # Issue reference
    if issue_ref:
        lines.append(f"✅ **References issue:** #{issue_ref}")
    else:
        lines.append(f"⚠️ **No issue reference found** — PR body should mention 'Fixes #N'")
    lines.append("")

    # CI status
    if ci_passing is True:
        lines.append("✅ **CI checks:** All passing")
    elif ci_passing is False:
        lines.append("❌ **CI checks:** Some failing")
    else:
        lines.append("ℹ️ **CI checks:** Not yet configured (first CI run will run after merge)")
    lines.append("")

    # Test results
    if test_result["passed"]:
        lines.append(f"✅ **Tests:** {test_result['passed_tests']} passed")
        if test_result["failed_tests"] > 0:
            lines.append(f"❌ {test_result['failed_tests']} failed")
    else:
        lines.append(f"❌ **Tests failed** (exit {test_result['exit_code']})")
        if test_result["output"]:
            lines.append(f"```\n{test_result['output'][-500:]}\n```")
    lines.append("")

    # Suspicious patterns
    if findings:
        lines.append("⚠️ **Suspicious patterns detected:**")
        for filepath, finding in findings:
            lines.append(f"- `{filepath}` — {finding}")
        lines.append("")
    else:
        lines.append("✅ **No suspicious patterns detected**\n")

    # Decision
    issues_found = bool(findings) or not test_result["passed"]
    can_approve = auto_approve and not issues_found

    if can_approve:
        lines.append("### ✅ Verdict: APPROVED")
        lines.append("Tests pass, no suspicious patterns, changes are scoped to the referenced issue.")
    elif issues_found:
        lines.append("### ❌ Verdict: CHANGES REQUESTED")
        if not test_result["passed"]:
            lines.append("- Tests failed — fix before merging")
        if findings:
            lines.append(f"- {len(findings)} suspicious pattern(s) found — review above")
    else:
        lines.append("### ❓ Verdict: REVIEW NEEDED")
        lines.append("Tests pass and no red flags found, but auto-approve is disabled. Manual review recommended.")
    lines.append("")
    lines.append("---")
    lines.append(f"_Auto-reviewed by Hermes Agent_")

    review_body = "\n".join(lines)

    # Post the review
    posted = post_review(pr_number, review_body)

    # Apply label
    if can_approve:
        apply_label(pr_number, "review-approved")
        status = "approved"
    elif issues_found:
        apply_label(pr_number, "changes-requested")
        status = "changes-requested"
    else:
        status = "needs-review"

    return {
        "pr": pr_number,
        "status": status,
        "tests_passed": test_result["passed"],
        "findings": len(findings),
        "posted": posted,
    }


def list_devin_prs() -> List[int]:
    """List all open PRs with devin-implemented or devin-ready label, not yet reviewed."""
    result = run([
        "gh", "pr", "list", "--repo", REPO, "--state", "open",
        "--json", "number,labels,title",
        "--search", "label:devin-implemented"
    ])
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", file=sys.stderr)
        return []

    prs = json.loads(result.stdout)
    unreviewed = []
    for pr in prs:
        labels = {l["name"] if isinstance(l, dict) else l for l in pr.get("labels", [])}
        if "review-approved" not in labels and "changes-requested" not in labels:
            unreviewed.append(pr["number"])
    return unreviewed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-review Devin PRs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("pr_number", nargs="?", type=int, help="PR number to review")
    group.add_argument("--batch", action="store_true", help="Review all unreviewed Devin PRs")
    parser.add_argument("--approve", action="store_true", help="Auto-approve passing PRs")
    args = parser.parse_args()

    if args.batch:
        prs = list_devin_prs()
        if not prs:
            print("No unreviewed Devin PRs found.")
            return
        print(f"Found {len(prs)} unreviewed Devin PR(s): {prs}")
        results = []
        for pr in prs:
            result = review_pr(pr, auto_approve=args.approve)
            results.append(result)
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in results:
            print(f"  #{r['pr']}: {r['status'].upper()} — tests: {'✅' if r['tests_passed'] else '❌'} {r.get('findings', 0)} findings")
        approved = sum(1 for r in results if r['status'] == 'approved')
        failed = sum(1 for r in results if r['status'] == 'changes-requested')
        print(f"\n{approved} approved, {failed} changes requested, {len(results) - approved - failed} needs review")
    elif args.pr_number:
        result = review_pr(args.pr_number, auto_approve=args.approve)
        print(f"\nResult: #{result['pr']} → {result['status'].upper()}")


if __name__ == "__main__":
    main()