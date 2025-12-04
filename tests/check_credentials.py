#!/usr/bin/env python3
"""
Check current AWS credentials and configuration
"""

import boto3
import os
from pathlib import Path


def check_credentials():
    """Check AWS credentials being used"""
    print("=== AWS Credentials Check ===\n")

    # 1. Environment variables
    print("1. Environment Variables:")
    env_vars = ['AWS_PROFILE', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
                'AWS_SESSION_TOKEN', 'AWS_DEFAULT_REGION']
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            if 'KEY' in var or 'TOKEN' in var:
                print(f"   {var}: ***{value[-4:]}")  # Show only last 4 chars
            else:
                print(f"   {var}: {value}")
        else:
            print(f"   {var}: Not set")

    # 2. AWS Config files
    print("\n2. AWS Config Files:")
    config_file = Path.home() / '.aws' / 'config'
    credentials_file = Path.home() / '.aws' / 'credentials'

    print(f"   Config file: {config_file.exists()}")
    print(f"   Credentials file: {credentials_file.exists()}")

    # 3. Profile configuration
    profile_name = 'sct_aiml_dev'
    print(f"\n3. Profile Configuration ('{profile_name}'):")

    try:
        # Read config file for profile
        if config_file.exists():
            with open(config_file, 'r') as f:
                content = f.read()
                if f'[profile {profile_name}]' in content:
                    print("   Profile found in config")
                    # Extract profile section
                    start = content.find(f'[profile {profile_name}]')
                    end = content.find('\n[', start + 1)
                    if end == -1:
                        end = len(content)
                    profile_section = content[start:end]

                    # Show relevant lines (hiding sensitive data)
                    for line in profile_section.split('\n'):
                        if line.strip() and not line.startswith('['):
                            if 'role_arn' in line:
                                parts = line.split('=')
                                if len(parts) > 1:
                                    arn = parts[1].strip()
                                    print(f"   role_arn: {arn}")
                            elif 'sso_' in line:
                                print(f"   {line.strip()}")
                else:
                    print("   Profile NOT found in config")
    except Exception as e:
        print(f"   Error reading config: {e}")

    # 4. Current session details
    print("\n4. Current Session Details:")
    try:
        session = boto3.Session(profile_name=profile_name)

        # Get credentials
        credentials = session.get_credentials()
        if credentials:
            print(f"   Access Key: ***{credentials.access_key[-4:]}")
            print(f"   Secret Key: {'***' if credentials.secret_key else 'None'}")
            print(f"   Session Token: {'***' + credentials.token[-4:] if credentials.token else 'None'}")
            print(f"   Method: {credentials.method}")

        # Get caller identity
        sts = session.client('sts')
        identity = sts.get_caller_identity()

        print(f"\n   Account: {identity['Account']}")
        print(f"   UserId: {identity['UserId']}")
        print(f"   Arn: {identity['Arn']}")

        # Parse the ARN to get role info
        arn_parts = identity['Arn'].split(':')
        if len(arn_parts) >= 6:
            resource = arn_parts[5]
            if 'assumed-role' in resource:
                role_parts = resource.split('/')
                if len(role_parts) >= 2:
                    print(f"   Role Name: {role_parts[1]}")

    except Exception as e:
        print(f"   Error getting session details: {e}")

    # 5. Region configuration
    print("\n5. Region Configuration:")
    try:
        session = boto3.Session(profile_name=profile_name)
        print(f"   Profile region: {session.region_name}")
        print(f"   Bedrock region: us-west-2 (from config)")
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    check_credentials()