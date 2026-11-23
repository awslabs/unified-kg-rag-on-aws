# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The AWS session every client of a config defaults to.

One owner for "the session from ``config.aws``": the configured profile and
``aws.region_name``. A client of a service in another region (Bedrock, in
``aws.bedrock_region``) names its region when it is built. The session never
falls back to ``AWS_DEFAULT_REGION`` or the profile's region where the config
names one (``docs/guides/configure.md``). The AWS SDK is imported on the
first call, so importing this module loads none of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import boto3

    from unified_kg_rag.domain.models import Config


def config_boto_session(config: Config) -> boto3.Session:
    """A new session on ``config.aws.profile_name`` and
    ``config.aws.region_name``. Building it makes no AWS call."""
    import boto3

    return boto3.Session(
        profile_name=config.aws.profile_name, region_name=config.aws.region_name
    )
