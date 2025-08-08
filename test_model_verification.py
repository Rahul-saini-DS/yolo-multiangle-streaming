#!/usr/bin/env python3
"""
Model Verification Test Script
Demonstrates that the selected model is actually being used
"""

from pipeline import Inference, discover_models, get_device_config
import sys

def test_model_selection():
    """Test that model selection actually uses the chosen model"""
    
    print("🔍 Model Selection Verification Test")
    print("=" * 50)
    
    # Discover available models
    models = discover_models()
    if not models:
        print("❌ No models found in 'models' directory")
        return
    
    print(f"📁 Found {len(models)} models:")
    for i, model in enumerate(models):
        print(f"   {i+1}. {model['name']} ({model['format']}) - {model['task']}")
    
    print("\n" + "=" * 50)
    
    # Test each model to verify it's actually loaded
    device_config = get_device_config(prefer_gpu=True)
    
    for model in models[:2]:  # Test first 2 models to avoid long runtime
        print(f"\n🧪 Testing model: {model['name']} ({model['format']})")
        print("-" * 30)
        
        try:
            # Create inference instance
            inference = Inference(
                model_path=model['path'],
                imgsz=640,
                conf=0.5,
                device=device_config['device'],
                half=device_config['half']
            )
            
            # Verify model usage
            verification = inference.verify_model_usage()
            
            # Check that the correct model is loaded
            loaded_model_name = verification['model_info']['filename']
            expected_model_name = model['name']
            
            if expected_model_name in loaded_model_name:
                print(f"✅ VERIFIED: Model '{expected_model_name}' is actually loaded")
                print(f"   📄 Loaded file: {loaded_model_name}")
                print(f"   🏷️ Format: {verification['model_info']['format']}")
                print(f"   🎯 Task: {verification['model_info']['task']}")
                print(f"   🔧 Device: {verification['device']}")
            else:
                print(f"❌ MISMATCH: Expected '{expected_model_name}' but loaded '{loaded_model_name}'")
            
            # Cleanup
            inference.cleanup()
            
        except Exception as e:
            print(f"❌ Failed to load model {model['name']}: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Model verification test completed!")


if __name__ == "__main__":
    test_model_selection()
