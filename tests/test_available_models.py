#!/usr/bin/env python3
"""
Test available Claude models
"""

import boto3
import json


def test_all_claude_models():
    """Test all available Claude models"""
    profile_name = 'sct_aiml_dev'
    region_name = 'us-west-2'

    session = boto3.Session(profile_name=profile_name)
    bedrock = session.client('bedrock', region_name=region_name)
    bedrock_runtime = session.client('bedrock-runtime', region_name=region_name)

    # Get list of Claude models
    response = bedrock.list_foundation_models()
    models = response.get('modelSummaries', [])
    claude_models = [m for m in models if 'claude' in m['modelId'].lower()]

    print(f"Found {len(claude_models)} Claude models\n")

    body = {
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
        "anthropic_version": "bedrock-2023-05-31"
    }

    # Test each model
    working_models = []
    for model in claude_models:
        model_id = model['modelId']
        print(f"Testing: {model_id}")

        try:
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )
            print(f"  ✅ {model_id} - WORKS!")
            working_models.append(model_id)
        except Exception as e:
            error_type = type(e).__name__
            if "AccessDenied" in error_type:
                print(f"  ❌ {model_id} - No access")
            else:
                print(f"  ❌ {model_id} - {error_type}")

    print("\n" + "="*60)
    print(f"Working models ({len(working_models)}):")
    for model_id in working_models:
        print(f"  - {model_id}")

    return working_models


if __name__ == "__main__":
    test_all_claude_models()