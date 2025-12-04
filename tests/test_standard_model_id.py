#!/usr/bin/env python3
"""
Test with standard model ID (without cross-region prefix)
"""

import boto3
import json


def test_standard_model_id():
    """Test with standard model ID"""
    print("=== Testing Standard Model ID ===")

    profile_name = 'sct_aiml_dev'
    region_name = 'us-west-2'

    session = boto3.Session(profile_name=profile_name)
    bedrock = session.client('bedrock-runtime', region_name=region_name)

    # Use standard model ID (without us. prefix)
    model_id = 'anthropic.claude-3-5-haiku-20241022-v1:0'
    print(f"Testing model: {model_id}")

    body = {
        "messages": [{"role": "user", "content": "Say hello in one word"}],
        "max_tokens": 10,
        "anthropic_version": "bedrock-2023-05-31"
    }

    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )

        result = json.loads(response['body'].read())
        print(f"✅ Standard model ID SUCCESS!")
        print(f"Response: {result.get('content', [{}])[0].get('text', 'No text')}")
        return True

    except Exception as e:
        print(f"❌ Standard model ID FAILED: {type(e).__name__}: {e}")

        # Try with Converse API
        print("\nTrying with Converse API...")
        try:
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
            print(f"✅ Converse API with standard ID SUCCESS!")
            print(f"Response: {output_text}")
            return True

        except Exception as e2:
            print(f"❌ Converse API also FAILED: {type(e2).__name__}: {e2}")
            return False


if __name__ == "__main__":
    test_standard_model_id()