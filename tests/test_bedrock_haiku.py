#!/usr/bin/env python3
"""
Test script for Bedrock Haiku 3.5 model access
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
from aws_graphrag.aws.bedrock import BedrockLanguageModelFactory
from aws_graphrag.core import get_config
from aws_graphrag.models import Config, LanguageModelId


def test_bedrock_haiku_direct():
    """Test direct Bedrock API call with Haiku 3.5"""
    print("\n=== Testing Direct Bedrock API Call ===")

    profile_name = 'sct_aiml_dev'
    region_name = 'us-west-2'

    try:
        # Create boto3 session
        session = boto3.Session(profile_name=profile_name)
        print(f"Created boto3 session with profile: {profile_name}")

        # Verify credentials
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        print(f"Current identity: {identity['Arn']}")

        # Create Bedrock client
        bedrock = session.client('bedrock-runtime', region_name=region_name)
        print(f"Created Bedrock client for region: {region_name}")

        # Test standard model ID
        model_id = 'anthropic.claude-3-5-haiku-20241022-v1:0'
        print(f"\nTesting model: {model_id}")

        body = {
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "max_tokens": 10,
            "anthropic_version": "bedrock-2023-05-31"
        }

        import json
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )

        result = json.loads(response['body'].read())
        print(f"✅ Direct API call SUCCESS!")
        print(f"Response: {result.get('content', [{}])[0].get('text', 'No text')}")

    except Exception as e:
        print(f"❌ Direct API call FAILED: {type(e).__name__}: {e}")
        return False

    return True


def test_bedrock_factory():
    """Test BedrockLanguageModelFactory with Haiku 3.5"""
    print("\n\n=== Testing BedrockLanguageModelFactory ===")

    try:
        # Create minimal config
        config = Config(
            aws=Config.aws.__class__(
                profile_name='sct_aiml_dev',
                region_name='ap-northeast-2',
                bedrock=Config.aws.bedrock.__class__(
                    region_name='us-west-2'
                )
            )
        )

        print(f"Config profile: {config.aws.profile_name}")
        print(f"Config Bedrock region: {config.aws.bedrock.region_name}")

        # Create factory
        factory = BedrockLanguageModelFactory(config)
        print(f"Created factory with region: {factory.region_name}")
        print(f"Boto session profile: {factory.boto_session.profile_name}")

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

    except Exception as e:
        print(f"❌ Factory test FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_cross_region_models():
    """Test cross-region model IDs"""
    print("\n\n=== Testing Cross-Region Model IDs ===")

    profile_name = 'sct_aiml_dev'
    region_name = 'us-west-2'

    try:
        session = boto3.Session(profile_name=profile_name)
        bedrock = session.client('bedrock-runtime', region_name=region_name)

        # Test different model ID formats
        test_model_ids = [
            'anthropic.claude-3-5-haiku-20241022-v1:0',  # Standard
            'us.anthropic.claude-3-5-haiku-20241022-v1:0',  # Regional
            'global.anthropic.claude-3-5-haiku-20241022-v1:0'  # Global
        ]

        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "anthropic_version": "bedrock-2023-05-31"
        }

        for model_id in test_model_ids:
            print(f"\nTesting model ID: {model_id}")
            try:
                import json
                response = bedrock.invoke_model(
                    modelId=model_id,
                    body=json.dumps(body)
                )
                print(f"✅ {model_id} - SUCCESS")
            except Exception as e:
                print(f"❌ {model_id} - FAILED: {e}")

    except Exception as e:
        print(f"❌ Cross-region test setup FAILED: {e}")
        return False

    return True


def test_inference_profiles():
    """Check available inference profiles"""
    print("\n\n=== Checking Inference Profiles ===")

    profile_name = 'sct_aiml_dev'
    region_name = 'us-west-2'

    try:
        session = boto3.Session(profile_name=profile_name)
        bedrock = session.client('bedrock', region_name=region_name)

        response = bedrock.list_inference_profiles(
            maxResults=1000,
            typeEquals='SYSTEM_DEFINED'
        )

        profiles = response.get('inferenceProfileSummaries', [])
        print(f"Found {len(profiles)} inference profiles")

        # Filter for Haiku profiles
        haiku_profiles = [p for p in profiles if 'haiku' in p['inferenceProfileId'].lower()]

        if haiku_profiles:
            print("\nHaiku-related profiles:")
            for profile in haiku_profiles:
                print(f"  - {profile['inferenceProfileId']}")
        else:
            print("\nNo Haiku-related profiles found")

    except Exception as e:
        print(f"❌ Inference profile check FAILED: {e}")
        return False

    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Bedrock Haiku 3.5 Test Suite")
    print("=" * 60)

    # Set environment variable if needed
    if 'AWS_PROFILE' not in os.environ:
        os.environ['AWS_PROFILE'] = 'sct_aiml_dev'
        print(f"Set AWS_PROFILE={os.environ['AWS_PROFILE']}")

    tests = [
        ("Direct Bedrock API", test_bedrock_haiku_direct),
        ("Bedrock Factory", test_bedrock_factory),
        ("Cross-Region Models", test_cross_region_models),
        ("Inference Profiles", test_inference_profiles)
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