# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom resource: the AWS-provided Nori plugin associated with a domain.

An optional plugin package exists once per OpenSearch engine version, so
``on_event`` looks up the ``analysis-nori`` ZIP-PLUGIN package of the
domain's version and makes its id the physical id: an engine version change
then replaces the resource (the new version's package is associated, and
CloudFormation deletes the old one in the update's cleanup). ``is_complete``
drives the association, a blue/green deployment of the domain, to ACTIVE.

Every step reads the domain's associations first, so a package associated by
hand (or by an engine upgrade) is left as it is. A Delete dissociates only
when a stack update removed or replaced the resource: a stack deletion or a
rollback leaves the package, so a retained domain keeps its nori indices
usable. A dissociation fails while an index still uses the analyzer.

Runs as Lambda functions behind the CDK custom-resource provider framework
(``on_event`` starts a change, ``is_complete`` is polled until it is done),
with only the boto3 of the Lambda runtime.
"""

from __future__ import annotations

import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

PACKAGE_TYPE = "ZIP-PLUGIN"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# The stack status in which CloudFormation deletes the resources a stack
# update removed or replaced. A Delete in any other status (stack deletion,
# a rollback) leaves the package associated.
UPDATE_CLEANUP = "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS"

# Errors that clear by themselves: the next poll tries again.
RETRYABLE_ERRORS = frozenset(
    {"ConflictException", "Throttling", "ThrottlingException", "TooManyRequests"}
)


def client(service: str) -> Any:
    """A boto3 client of the function's region (patched in tests)."""
    return boto3.client(service)  # type: ignore[call-overload]


def error_code(error: ClientError) -> str:
    return str(error.response.get("Error", {}).get("Code", ""))


def dissociates_on_delete(event: dict[str, Any]) -> bool:
    """Whether a Delete should dissociate: only when a stack update removed
    or replaced the resource."""
    (stack,) = client("cloudformation").describe_stacks(StackName=event["StackId"])[
        "Stacks"
    ]
    status = stack["StackStatus"]
    if status != UPDATE_CLEANUP:
        logger.info("Stack status %s: leaving the package associated", status)
        return False
    return True


class PackageError(RuntimeError):
    """The package cannot be found, associated or dissociated."""


def on_event(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    if event["RequestType"] == "Delete":
        return {}
    props = event["ResourceProperties"]
    package_id = find_package(props["PackageName"], props["EngineVersion"])
    logger.info(
        "Package %s is %s for %s",
        package_id,
        props["PackageName"],
        props["EngineVersion"],
    )
    return {"PhysicalResourceId": package_id, "Data": {"PackageId": package_id}}


def is_complete(event: dict[str, Any], _context: Any) -> dict[str, bool]:
    domain = event["ResourceProperties"]["DomainName"]
    package_id = event["PhysicalResourceId"]
    try:
        if event["RequestType"] == "Delete":
            done = not dissociates_on_delete(event) or _dissociated(domain, package_id)
        else:
            done = _associated(domain, package_id)
    except ClientError as error:
        if error_code(error) not in RETRYABLE_ERRORS:
            raise
        logger.info("Retrying after %s", error_code(error))
        done = False
    return {"IsComplete": done}


def find_package(name: str, engine_version: str) -> str:
    """The id of the AWS-provided plugin package ``name`` for ``engine_version``."""
    opensearch = client("opensearch")
    filters = [
        {"Name": "PackageName", "Value": [name]},
        {"Name": "PackageType", "Value": [PACKAGE_TYPE]},
        {"Name": "EngineVersion", "Value": [engine_version]},
    ]
    packages: list[dict[str, Any]] = []
    request: dict[str, Any] = {"Filters": filters}
    while True:
        page = opensearch.describe_packages(**request)
        packages += page.get("PackageDetailsList", [])
        if not page.get("NextToken"):
            break
        request["NextToken"] = page["NextToken"]
    ids = sorted({p["PackageID"] for p in packages if p.get("PackageName") == name})
    if len(ids) != 1:
        raise PackageError(
            f"Expected one {PACKAGE_TYPE} package {name!r} for {engine_version}, "
            f"found {ids or 'none'} (aws opensearch describe-packages): the "
            "region may not offer it for this engine version"
        )
    return str(ids[0])


def _association(domain: str, package_id: str) -> dict[str, Any] | None:
    opensearch = client("opensearch")
    request: dict[str, Any] = {"DomainName": domain}
    while True:
        page = opensearch.list_packages_for_domain(**request)
        for details in page.get("DomainPackageDetailsList", []):
            if details["PackageID"] == package_id:
                return dict(details)
        if not page.get("NextToken"):
            return None
        request["NextToken"] = page["NextToken"]


def _failure(details: dict[str, Any]) -> str:
    error = details.get("ErrorDetails") or {}
    return f"{error.get('ErrorType', '')}: {error.get('ErrorMessage', '')}".strip(": ")


def _associated(domain: str, package_id: str) -> bool:
    details = _association(domain, package_id)
    status = details["DomainPackageStatus"] if details else None
    if status == "ACTIVE":
        return True
    if status in ("ASSOCIATION_FAILED", "DISSOCIATION_FAILED"):
        assert details is not None
        raise PackageError(
            f"Package {package_id} on domain {domain} is {status} "
            f"({_failure(details)}); dissociate it "
            "(aws opensearch dissociate-package) and deploy again"
        )
    if status is None:
        # One blue/green deployment at a time: wait out a running change.
        state = client("opensearch").describe_domain(DomainName=domain)
        state = state["DomainStatus"]
        if state.get("Processing") or state.get("UpgradeProcessing"):
            logger.info("Domain %s is processing a change; waiting", domain)
            return False
        logger.info("Associating package %s with domain %s", package_id, domain)
        client("opensearch").associate_package(PackageID=package_id, DomainName=domain)
    return False  # ASSOCIATING, DISSOCIATING (associated again next), or just asked


def _dissociated(domain: str, package_id: str) -> bool:
    try:
        details = _association(domain, package_id)
    except ClientError as error:
        if error_code(error) == "ResourceNotFoundException":
            return True  # the domain is gone
        raise
    if details is None:
        return True
    status = details["DomainPackageStatus"]
    if status == "DISSOCIATION_FAILED":
        raise PackageError(
            f"Dissociating package {package_id} from domain {domain} failed "
            f"({_failure(details)}): an index still using the analyzer blocks "
            "it; rebuild the Korean indices with another analyzer first"
        )
    if status in ("ACTIVE", "ASSOCIATION_FAILED"):
        logger.info("Dissociating package %s from domain %s", package_id, domain)
        client("opensearch").dissociate_package(PackageID=package_id, DomainName=domain)
    return False
