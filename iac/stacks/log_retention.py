# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Log group retention from a number of days."""

from __future__ import annotations

from aws_cdk import aws_logs as logs


def set_retention(log_group: logs.ILogGroup, days: int) -> None:
    """Keep ``log_group``'s events ``days`` days (a CloudWatch Logs retention
    value, checked by ``DeploymentConfig``). The ``RetentionDays`` enum cannot
    be built from a number in Python, so the CloudFormation property is set
    directly."""
    cfn_log_group = log_group.node.default_child
    assert isinstance(cfn_log_group, logs.CfnLogGroup)
    cfn_log_group.retention_in_days = days
