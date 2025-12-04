#!/usr/bin/env python3
"""
Debug Bedrock access issues
"""

import boto3
import json
from datetime import datetime


def debug_bedrock_access():
    """Comprehensive Bedrock access debugging"""
    profile_name = 'sct_aiml_dev'
    region_name = 'us-west-2'

    print("=== Bedrock Access Debugging ===\n")

    session = boto3.Session(profile_name=profile_name)

    # 1. Check Bedrock endpoint
    print("1. Bedrock Endpoints:")
    bedrock = session.client('bedrock', region_name=region_name)
    bedrock_runtime = session.client('bedrock-runtime', region_name=region_name)

    print(f"   Bedrock endpoint: {bedrock._endpoint}")
    print(f"   Bedrock Runtime endpoint: {bedrock_runtime._endpoint}")

    # 2. Get model details
    print("\n2. Model Details:")
    try:
        response = bedrock.get_foundation_model(
            # modelIdentifier='anthropic.claude-3-5-haiku-20241022-v1:0'
            modelIdentifier='us.anthropic.claude-opus-4-1-20250805-v1:0'
        )
        model_details = response.get('modelDetails', {})
        print(f"   Model ID: {model_details.get('modelId')}")
        print(f"   Model Name: {model_details.get('modelName')}")
        print(f"   Provider: {model_details.get('providerName')}")
        print(f"   Status: {model_details.get('modelLifecycle', {}).get('status')}")

        # Check inference types
        inference_types = model_details.get('inferenceTypesSupported', [])
        print(f"   Inference Types: {inference_types}")

        # Check input/output modalities
        input_modalities = model_details.get('inputModalities', [])
        output_modalities = model_details.get('outputModalities', [])
        print(f"   Input Modalities: {input_modalities}")
        print(f"   Output Modalities: {output_modalities}")

    except Exception as e:
        print(f"   Error getting model details: {e}")

    # 3. List model access
    print("\n3. Model Access Status:")
    try:
        # This API might not exist, but let's try
        response = bedrock.list_model_customization_jobs()
        print(f"   Customization jobs: {len(response.get('modelCustomizationJobSummaries', []))}")
    except:
        print("   Model customization API not available")

    # 4. Test different request formats
    print("\n4. Testing Different Request Formats:")

    test_cases = [
        {
            "name": "Standard invoke_model",
            "method": "invoke_model",
            "body": {
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "anthropic_version": "bedrock-2023-05-31"
            }
        },
        {
            "name": "Without anthropic_version",
            "method": "invoke_model",
            "body": {
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            }
        },
        {
            "name": "Old format (prompt)",
            "method": "invoke_model",
            "body": {
                "prompt": "\n\nHuman: Hi\n\nAssistant:",
                "max_tokens": 10
            }
        },
        {
            "name": "Converse API",
            "method": "converse",
            "body": None  # Special handling
        }
    ]

    for test in test_cases:
        print(f"\n   Testing: {test['name']}")
        try:
            if test['method'] == 'invoke_model':
                response = bedrock_runtime.invoke_model(
                    modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0',
                    # modelId='us.anthropic.claude-opus-4-1-20250805-v1:0',
                    # modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
                    body=json.dumps(test['body'])
                )
                print(f"     ✅ SUCCESS")
            elif test['method'] == 'converse':
                response = bedrock_runtime.converse(
                    modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0',
                    # modelId='us.anthropic.claude-opus-4-1-20250805-v1:0',
                    # modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
                    messages=[{"role": "user", "content": [{"text": "Hi"}]}],
                    inferenceConfig={"maxTokens": 10}
                )
                print(f"     ✅ SUCCESS")
        except Exception as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown') if hasattr(e, 'response') else 'Unknown'
            error_msg = str(e)
            print(f"     ❌ FAILED: {error_code}")
            if 'access' in error_msg.lower():
                print(f"        Full error: {error_msg}")

    # 5. Check IAM permissions
    print("\n5. IAM Permission Check:")
    try:
        iam = session.client('iam')
        sts = session.client('sts')

        # Get current role
        identity = sts.get_caller_identity()
        role_name = identity['Arn'].split('/')[-2]

        print(f"   Current role: {role_name}")

        # Try to simulate policy
        try:
            response = iam.simulate_principal_policy(
                PolicySourceArn=identity['Arn'],
                ActionNames=['bedrock:InvokeModel'],
                ResourceArns=['arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0']
            )
            results = response.get('EvaluationResults', [])
            for result in results:
                print(f"   Action: {result.get('EvalActionName')}")
                print(f"   Decision: {result.get('EvalDecision')}")
        except Exception as e:
            print(f"   Cannot simulate policy: {e}")

    except Exception as e:
        print(f"   IAM check error: {e}")

    # 6. List all inference profiles
    print("\n6. Available Inference Profiles:")
    try:
        response = bedrock.list_inference_profiles(maxResults=100)
        profiles = response.get('inferenceProfileSummaries', [])

        haiku_profiles = [p for p in profiles if 'haiku' in p.get('inferenceProfileId', '').lower() and '3-5' in p.get('inferenceProfileId', '')]

        for profile in haiku_profiles:
            profile_id = profile.get('inferenceProfileId')
            print(f"\n   Profile: {profile_id}")
            print(f"   Status: {profile.get('status')}")
            print(f"   Type: {profile.get('type')}")

            # Test this profile
            print(f"   Testing access...")
            try:
                response = bedrock_runtime.invoke_model(
                    modelId=profile_id,
                    body=json.dumps({
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 10,
                        "anthropic_version": "bedrock-2023-05-31"
                    })
                )
                print(f"   ✅ This profile WORKS!")
            except Exception as e:
                print(f"   ❌ Failed: {type(e).__name__}")

    except Exception as e:
        print(f"   Error listing profiles: {e}")


if __name__ == "__main__":
    debug_bedrock_access()