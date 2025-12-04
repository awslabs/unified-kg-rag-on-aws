#!/usr/bin/env python3
"""
Fixed test script for Bedrock Haiku 3.5 model access
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
import json
from aws_graphrag.aws.bedrock import BedrockLanguageModelFactory
from aws_graphrag.models.config import (
    Config, AWSConfig, BedrockConfig, LanguageModelId
)


def test_sso_login_status():
    """Check SSO login status"""
    print("\n=== Checking SSO Login Status ===")

    profile_name = 'sct_aiml_dev'

    try:
        session = boto3.Session(profile_name=profile_name)
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ SSO is active")
        print(f"Account: {identity['Account']}")
        print(f"User/Role: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"❌ SSO login required: {e}")
        print(f"\nRun: aws sso login --profile {profile_name}")
        return False


def test_us_prefixed_model():
    """Test with us. prefixed model ID (found in inference profiles)"""
    print("\n=== Testing us. Prefixed Model ID ===")

    profile_name = 'sct_aiml_dev'
    region_name = 'us-west-2'

    try:
        session = boto3.Session(profile_name=profile_name)
        bedrock = session.client('bedrock-runtime', region_name=region_name)

        # Use the us. prefixed model ID found in inference profiles
        model_id = 'us.anthropic.claude-3-5-haiku-20241022-v1:0'
        print(f"Testing model: {model_id}")

        body = {
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "max_tokens": 10,
            "anthropic_version": "bedrock-2023-05-31"
        }

        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )

        result = json.loads(response['body'].read())
        print(f"✅ us. prefixed model SUCCESS!")
        print(f"Response: {result.get('content', [{}])[0].get('text', 'No text')}")
        return True

    except Exception as e:
        print(f"❌ us. prefixed model FAILED: {type(e).__name__}: {e}")
        return False


def test_bedrock_factory_fixed():
    """Test BedrockLanguageModelFactory with proper config"""
    print("\n\n=== Testing BedrockLanguageModelFactory (Fixed) ===")

    try:
        # Create proper config structure
        aws_config = AWSConfig(
            profile_name='sct_aiml_dev',
            region_name='ap-northeast-2',
            bedrock=BedrockConfig(
                region_name='us-west-2'
            )
        )

        config = Config(aws=aws_config)

        print(f"Config profile: {config.aws.profile_name}")
        print(f"Config Bedrock region: {config.aws.bedrock.region_name}")

        # Create factory
        factory = BedrockLanguageModelFactory(config)
        print(f"Created factory with region: {factory.region_name}")

        # Get model
        model_id = LanguageModelId.CLAUDE_V3_5_HAIKU
        print(f"\nGetting model: {model_id.value}")

        model = factory.get_model(model_id, max_tokens=10)
        print(f"✅ Model created successfully!")
        print(f"Model type: {type(model).__name__}")

        # Test invoke
        print("\nTesting model invoke...")
        response = model.invoke("Say hello in one word")
        print(f"✅ Model invoke SUCCESS!")
        print(f"Response: {response.content}")
        return True

    except Exception as e:
        print(f"❌ Factory test FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bedrock_converse_api():
    """Test with Bedrock Converse API"""
    print("\n\n=== Testing Bedrock Converse API ===")

    profile_name = 'sct_aiml_dev'
    region_name = 'us-west-2'

    try:
        session = boto3.Session(profile_name=profile_name)
        bedrock = session.client('bedrock-runtime', region_name=region_name)

        # Try with Converse API
        model_id = 'us.anthropic.claude-3-5-haiku-20241022-v1:0'
        print(f"Testing Converse API with model: {model_id}")

        response = bedrock.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "Say hello in one word"}]
                }
            ],
            inferenceConfig={
                "maxTokens": 10,
                "temperature": 0
            }
        )

        output_text = response['output']['message']['content'][0]['text']
        print(f"✅ Converse API SUCCESS!")
        print(f"Response: {output_text}")
        return True

    except Exception as e:
        print(f"❌ Converse API FAILED: {type(e).__name__}: {e}")
        return False


def check_model_access():
    """Check which models we have access to"""
    print("\n\n=== Checking Model Access ===")

    profile_name = 'sct_aiml_dev'
    region_name = 'us-west-2'

    try:
        session = boto3.Session(profile_name=profile_name)
        bedrock = session.client('bedrock', region_name=region_name)

        # List foundation models
        response = bedrock.list_foundation_models()
        models = response.get('modelSummaries', [])

        # Filter for Claude models
        claude_models = [m for m in models if 'claude' in m['modelId'].lower()]

        print(f"Found {len(claude_models)} Claude models")

        # Check for Haiku 3.5
        haiku_35_models = [m for m in claude_models if 'haiku' in m['modelId'] and '3-5' in m['modelId']]

        if haiku_35_models:
            print("\nHaiku 3.5 models available:")
            for model in haiku_35_models:
                print(f"  - {model['modelId']}")
                print(f"    Name: {model.get('modelName', 'N/A')}")
                print(f"    Status: {model.get('modelLifecycle', {}).get('status', 'N/A')}")
        else:
            print("\n❌ No Haiku 3.5 models found in list")

        return True

    except Exception as e:
        print(f"❌ Model access check FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Bedrock Haiku 3.5 Test Suite (Fixed)")
    print("=" * 60)

    # Check SSO first
    if not test_sso_login_status():
        print("\n⚠️  Please login with SSO first")
        return

    tests = [
        ("Check Model Access", check_model_access),
        ("us. Prefixed Model", test_us_prefixed_model),
        ("Bedrock Converse API", test_bedrock_converse_api),
        ("Bedrock Factory", test_bedrock_factory_fixed),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")

    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()