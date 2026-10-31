# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Nori package handlers against stubbed AWS clients (no AWS calls).

botocore's Stubber checks every request against the service model, so the
parameters the handlers send are the ones the real APIs take.
"""

from __future__ import annotations

import datetime
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "custom_resources"))

import nori_package  # noqa: E402

_STACK_ID = "arn:aws:cloudformation:us-west-2:111111111111:stack/GraphRagStorage/1"
_DOMAIN = "opensearch-domain"
_PACKAGE = "G123456789"


class _Clients:
    """One stubbed client per service, as ``nori_package.client`` hands them out."""

    def __init__(self) -> None:
        self.stubs: dict[str, Stubber] = {}
        self.clients: dict[str, Any] = {}

    def __call__(self, service: str) -> Any:
        return self.clients[service]

    def stub(self, service: str) -> Stubber:
        if service not in self.stubs:
            client = boto3.client(
                service,
                region_name="us-west-2",
                aws_access_key_id="testing",
                aws_secret_access_key="testing",
            )
            self.clients[service] = client
            self.stubs[service] = Stubber(client)
            self.stubs[service].activate()
        return self.stubs[service]


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Clients]:
    clients = _Clients()
    monkeypatch.setattr(nori_package, "client", clients)
    for service in ("cloudformation", "opensearch"):
        clients.stub(service)
    yield clients
    for stub in clients.stubs.values():
        stub.assert_no_pending_responses()


def _stack_status(aws: _Clients, status: str) -> None:
    aws.stub("cloudformation").add_response(
        "describe_stacks",
        {
            "Stacks": [
                {
                    "StackName": "GraphRagStorage",
                    "CreationTime": datetime.datetime(2026, 1, 1),
                    "StackStatus": status,
                }
            ]
        },
        {"StackName": _STACK_ID},
    )


def _event(request: str, **props: str) -> dict[str, Any]:
    return {
        "RequestType": request,
        "StackId": _STACK_ID,
        "PhysicalResourceId": props.pop("physical_id", ""),
        "ResourceProperties": props,
    }


def _nori_event(request: str) -> dict[str, Any]:
    return _event(
        request,
        DomainName=_DOMAIN,
        EngineVersion="OpenSearch_2.13",
        PackageName="analysis-nori",
        physical_id=_PACKAGE,
    )


_FILTERS = [
    {"Name": "PackageName", "Value": ["analysis-nori"]},
    {"Name": "PackageType", "Value": ["ZIP-PLUGIN"]},
    {"Name": "EngineVersion", "Value": ["OpenSearch_2.13"]},
]


def _packages(*ids: str, next_token: str | None = None) -> dict[str, Any]:
    page: dict[str, Any] = {
        "PackageDetailsList": [
            {
                "PackageID": package_id,
                "PackageName": "analysis-nori",
                "PackageType": "ZIP-PLUGIN",
                "EngineVersion": "OpenSearch_2.13",
            }
            for package_id in ids
        ]
    }
    if next_token:
        page["NextToken"] = next_token
    return page


def _domain_packages(aws: _Clients, status: str | None, **error: str) -> None:
    details: list[dict[str, Any]] = [
        {"PackageID": "F000000000", "DomainPackageStatus": "ACTIVE"}
    ]
    if status:
        entry: dict[str, Any] = {"PackageID": _PACKAGE, "DomainPackageStatus": status}
        if error:
            entry["ErrorDetails"] = error
        details.append(entry)
    aws.stub("opensearch").add_response(
        "list_packages_for_domain",
        {"DomainPackageDetailsList": details},
        {"DomainName": _DOMAIN},
    )


def _domain_processing(aws: _Clients, processing: bool) -> None:
    aws.stub("opensearch").add_response(
        "describe_domain",
        {
            "DomainStatus": {
                "DomainId": "111111111111/opensearch-domain",
                "DomainName": _DOMAIN,
                "ARN": "arn:aws:es:us-west-2:111111111111:domain/opensearch-domain",
                "ClusterConfig": {},
                "Processing": processing,
                "UpgradeProcessing": False,
            }
        },
        {"DomainName": _DOMAIN},
    )


def test_nori_on_event_resolves_the_package_of_the_engine_version(
    aws: _Clients,
) -> None:
    opensearch = aws.stub("opensearch")
    opensearch.add_response(
        "describe_packages", _packages(next_token="t1"), {"Filters": _FILTERS}
    )
    opensearch.add_response(
        "describe_packages",
        _packages(_PACKAGE),
        {"Filters": _FILTERS, "NextToken": "t1"},
    )
    result = nori_package.on_event(_nori_event("Create"), None)
    assert result == {"PhysicalResourceId": _PACKAGE, "Data": {"PackageId": _PACKAGE}}


@pytest.mark.parametrize("ids", [(), (_PACKAGE, "G999999999")], ids=["none", "two"])
def test_nori_on_event_needs_exactly_one_package(
    aws: _Clients, ids: tuple[str, ...]
) -> None:
    aws.stub("opensearch").add_response(
        "describe_packages", _packages(*ids), {"Filters": _FILTERS}
    )
    with pytest.raises(nori_package.PackageError, match="OpenSearch_2.13"):
        nori_package.on_event(_nori_event("Update"), None)


def test_nori_on_event_delete_keeps_the_physical_id(aws: _Clients) -> None:
    assert nori_package.on_event(_nori_event("Delete"), None) == {}


def test_nori_associates_a_missing_package_on_an_idle_domain(aws: _Clients) -> None:
    _domain_packages(aws, None)
    _domain_processing(aws, False)
    aws.stub("opensearch").add_response(
        "associate_package",
        {},
        {"PackageID": _PACKAGE, "DomainName": _DOMAIN},
    )
    assert nori_package.is_complete(_nori_event("Create"), None) == {
        "IsComplete": False
    }


def test_nori_waits_for_a_running_domain_change(aws: _Clients) -> None:
    _domain_packages(aws, None)
    _domain_processing(aws, True)
    assert nori_package.is_complete(_nori_event("Update"), None) == {
        "IsComplete": False
    }


@pytest.mark.parametrize(
    ("status", "done"),
    [("ASSOCIATING", False), ("DISSOCIATING", False), ("ACTIVE", True)],
)
def test_nori_completes_when_the_association_is_active(
    aws: _Clients, status: str, done: bool
) -> None:
    _domain_packages(aws, status)
    assert nori_package.is_complete(_nori_event("Create"), None) == {"IsComplete": done}


def test_nori_fails_on_a_failed_association(aws: _Clients) -> None:
    _domain_packages(
        aws, "ASSOCIATION_FAILED", ErrorType="PluginError", ErrorMessage="no room"
    )
    with pytest.raises(nori_package.PackageError, match="PluginError: no room"):
        nori_package.is_complete(_nori_event("Create"), None)


def test_nori_retries_a_conflicting_change(aws: _Clients) -> None:
    _domain_packages(aws, None)
    _domain_processing(aws, False)
    aws.stub("opensearch").add_client_error(
        "associate_package", service_error_code="ConflictException"
    )
    assert nori_package.is_complete(_nori_event("Create"), None) == {
        "IsComplete": False
    }


def test_nori_raises_other_errors(aws: _Clients) -> None:
    aws.stub("opensearch").add_client_error(
        "list_packages_for_domain", service_error_code="AccessDeniedException"
    )
    with pytest.raises(Exception, match="AccessDeniedException"):
        nori_package.is_complete(_nori_event("Create"), None)


@pytest.mark.parametrize(
    "stack_status",
    ["DELETE_IN_PROGRESS", "UPDATE_ROLLBACK_IN_PROGRESS", "ROLLBACK_IN_PROGRESS"],
)
def test_nori_delete_outside_an_update_cleanup_keeps_the_package(
    aws: _Clients, stack_status: str
) -> None:
    _stack_status(aws, stack_status)
    # No OpenSearch call: the Stubber fails on an unexpected one.
    assert nori_package.is_complete(_nori_event("Delete"), None) == {"IsComplete": True}


def test_nori_delete_in_an_update_cleanup_dissociates(aws: _Clients) -> None:
    _stack_status(aws, nori_package.UPDATE_CLEANUP)
    _domain_packages(aws, "ACTIVE")
    aws.stub("opensearch").add_response(
        "dissociate_package", {}, {"PackageID": _PACKAGE, "DomainName": _DOMAIN}
    )
    assert nori_package.is_complete(_nori_event("Delete"), None) == {
        "IsComplete": False
    }
    _stack_status(aws, nori_package.UPDATE_CLEANUP)
    _domain_packages(aws, None)
    assert nori_package.is_complete(_nori_event("Delete"), None) == {"IsComplete": True}


def test_nori_delete_of_a_gone_domain_completes(aws: _Clients) -> None:
    _stack_status(aws, nori_package.UPDATE_CLEANUP)
    aws.stub("opensearch").add_client_error(
        "list_packages_for_domain", service_error_code="ResourceNotFoundException"
    )
    assert nori_package.is_complete(_nori_event("Delete"), None) == {"IsComplete": True}


def test_nori_failed_dissociation_names_the_cause(aws: _Clients) -> None:
    _stack_status(aws, nori_package.UPDATE_CLEANUP)
    _domain_packages(aws, "DISSOCIATION_FAILED", ErrorMessage="index uses nori")
    with pytest.raises(nori_package.PackageError, match="index uses nori"):
        nori_package.is_complete(_nori_event("Delete"), None)
