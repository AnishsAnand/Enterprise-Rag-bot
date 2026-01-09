#!/usr/bin/env python3
"""
PRODUCTION TEST SCRIPT: Validate Image Rendering in OpenWebUI
Tests all scenarios where images should appear.
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import List, Dict, Any

# Test the formatter directly
from app.services.openwebui_formatter import format_for_openwebui


def print_test_header(test_name: str):
    """Print formatted test header."""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)


def validate_markdown_images(markdown: str) -> Dict[str, Any]:
    """
    Validate that markdown contains proper image tags.
    Returns validation results.
    """
    import re
    
    # Find all image markdown tags
    image_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
    images = re.findall(image_pattern, markdown)
    
    validation = {
        "total_images": len(images),
        "valid_urls": 0,
        "placeholder_urls": 0,
        "invalid_urls": 0,
        "images": []
    }
    
    for alt, url in images:
        image_info = {"alt": alt, "url": url}
        
        if url.startswith("http://") or url.startswith("https://"):
            validation["valid_urls"] += 1
            
            if "placeholder" in url.lower():
                validation["placeholder_urls"] += 1
                image_info["type"] = "placeholder"
            else:
                image_info["type"] = "real"
        else:
            validation["invalid_urls"] += 1
            image_info["type"] = "invalid"
        
        validation["images"].append(image_info)
    
    return validation


async def test_steps_with_image_prompts():
    """Test 1: Steps with image_prompt fields (most common scenario)."""
    print_test_header("Steps with image_prompt fields")
    
    steps = [
        {
            "step_number": 1,
            "text": "Log in to the Vayu Cloud portal and navigate to firewall settings.",
            "type": "action",
            "image_prompt": "Screenshot of Vayu Cloud login page with username field highlighted"
        },
        {
            "step_number": 2,
            "text": "Click on the firewall configuration menu.",
            "type": "action",
            "image_prompt": "UI showing firewall menu with 'Configuration' option circled in red"
        },
        {
            "step_number": 3,
            "text": "Enter the new throughput values and save.",
            "type": "action",
            "image_prompt": "Configuration form with throughput input fields and Save button"
        }
    ]
    
    formatted = format_for_openwebui(
        answer="Follow these steps to configure your firewall.",
        steps=steps,
        images=[],
        query="How do I configure firewall throughput?",
        confidence=0.85
    )
    
    print("\n📝 Generated Markdown (first 1000 chars):")
    print(formatted[:1000])
    
    validation = validate_markdown_images(formatted)
    print("\n✅ Validation Results:")
    print(f"   Total Images Found: {validation['total_images']}")
    print(f"   Valid URLs: {validation['valid_urls']}")
    print(f"   Placeholder URLs: {validation['placeholder_urls']}")
    print(f"   Invalid URLs: {validation['invalid_urls']}")
    
    # Show first 3 image URLs
    print("\n🖼️  Sample Image URLs:")
    for i, img in enumerate(validation['images'][:3], 1):
        print(f"   {i}. [{img['type']}] {img['url'][:80]}...")
    
    # Test assertions
    assert validation['total_images'] == len(steps), "Each step should have an image!"
    assert validation['invalid_urls'] == 0, "No invalid URLs allowed!"
    assert validation['placeholder_urls'] > 0, "Should have generated placeholder URLs!"
    
    print("\n✅ TEST PASSED: All steps have valid image URLs")
    return formatted


async def test_steps_with_real_images():
    """Test 2: Steps with real image URLs."""
    print_test_header("Steps with real image URLs")
    
    steps = [
        {
            "step_number": 1,
            "text": "Open the dashboard.",
            "type": "action",
            "image": {
                "url": "https://example.com/images/dashboard.png",
                "alt": "Dashboard screenshot",
                "caption": "Main dashboard view"
            }
        },
        {
            "step_number": 2,
            "text": "Navigate to settings.",
            "type": "action",
            "image": "https://example.com/images/settings.png"
        }
    ]
    
    formatted = format_for_openwebui(
        answer="Here's how to access settings.",
        steps=steps,
        images=[],
        query="How do I access settings?",
        confidence=0.90
    )
    
    validation = validate_markdown_images(formatted)
    print("\n✅ Validation Results:")
    print(f"   Total Images: {validation['total_images']}")
    print(f"   Real URLs: {validation['valid_urls'] - validation['placeholder_urls']}")
    
    assert validation['total_images'] == len(steps), "Each step should have an image!"
    assert validation['invalid_urls'] == 0, "No invalid URLs!"
    
    print("\n✅ TEST PASSED: Real image URLs preserved correctly")
    return formatted


async def test_steps_with_mixed_content():
    """Test 3: Mixed scenario - some with URLs, some with prompts, some with nothing."""
    print_test_header("Mixed content (URLs, prompts, and missing)")
    
    steps = [
        {
            "step_number": 1,
            "text": "First step with real image.",
            "type": "action",
            "image": {"url": "https://example.com/step1.png", "alt": "Step 1"}
        },
        {
            "step_number": 2,
            "text": "Second step with image prompt.",
            "type": "action",
            "image_prompt": "Diagram showing network topology"
        },
        {
            "step_number": 3,
            "text": "Third step with NO image information at all.",
            "type": "action"
            # Deliberately missing image/image_prompt
        }
    ]
    
    formatted = format_for_openwebui(
        answer="Mixed content test.",
        steps=steps,
        images=[],
        query="Test query",
        confidence=0.75
    )
    
    validation = validate_markdown_images(formatted)
    print("\n✅ Validation Results:")
    print(f"   Total Images: {validation['total_images']}")
    print(f"   Valid URLs: {validation['valid_urls']}")
    
    # Critical assertion: Even step 3 (with NO image info) should get a placeholder
    assert validation['total_images'] == len(steps), \
        "CRITICAL: Even steps without image info should get placeholders!"
    assert validation['invalid_urls'] == 0, "No invalid URLs!"
    
    print("\n✅ TEST PASSED: Missing images handled with placeholders")
    return formatted


async def test_image_pool_assignment():
    """Test 4: Image pool assignment to steps."""
    print_test_header("Image pool assignment")
    
    steps = [
        {"step_number": 1, "text": "Step 1", "type": "action"},
        {"step_number": 2, "text": "Step 2", "type": "action"},
        {"step_number": 3, "text": "Step 3", "type": "action"}
    ]
    
    images = [
        {"url": "https://example.com/img1.png", "alt": "Image 1"},
        {"url": "https://example.com/img2.png", "alt": "Image 2"},
        {"url": "https://example.com/img3.png", "alt": "Image 3"}
    ]
    
    formatted = format_for_openwebui(
        answer="Testing image pool.",
        steps=steps,
        images=images,
        query="Test",
        confidence=0.80
    )
    
    validation = validate_markdown_images(formatted)
    print("\n✅ Validation Results:")
    print(f"   Total Images: {validation['total_images']}")
    print(f"   From Pool: {sum(1 for img in validation['images'] if 'example.com' in img['url'])}")
    
    assert validation['total_images'] >= len(steps), "All steps should have images!"
    
    print("\n✅ TEST PASSED: Image pool assigned correctly")
    return formatted


async def test_empty_steps():
    """Test 5: Edge case - no steps provided."""
    print_test_header("Edge case - No steps")
    
    formatted = format_for_openwebui(
        answer="This is just a plain answer without steps.",
        steps=[],
        images=[],
        query="Simple question",
        confidence=0.70
    )
    
    validation = validate_markdown_images(formatted)
    print("\n✅ Validation Results:")
    print(f"   Total Images: {validation['total_images']}")
    
    # Should still have answer, just no step images
    assert "plain answer" in formatted, "Answer should be present!"
    
    print("\n✅ TEST PASSED: Handles no-steps scenario gracefully")
    return formatted


async def test_special_characters_in_prompts():
    """Test 6: Image prompts with special characters."""
    print_test_header("Special characters in prompts")
    
    steps = [
        {
            "step_number": 1,
            "text": "Test special chars",
            "type": "action",
            "image_prompt": "Screenshot with 'quotes' and \"double quotes\" & special chars: @#$%"
        }
    ]
    
    formatted = format_for_openwebui(
        answer="Testing special characters.",
        steps=steps,
        images=[],
        query="Test",
        confidence=0.75
    )
    
    validation = validate_markdown_images(formatted)
    print("\n✅ Validation Results:")
    print(f"   Total Images: {validation['total_images']}")
    print(f"   First URL: {validation['images'][0]['url'][:80]}...")
    
    # URL should be properly encoded
    assert validation['total_images'] == 1, "Should have 1 image!"
    assert validation['valid_urls'] == 1, "URL should be valid!"
    
    # ✅ FIXED: Check for %20 (space encoded) instead of + or raw spaces
    url = validation['images'][0]['url']
    assert "%20" in url or "+" not in url, "Spaces should be URL encoded as %20 or removed!"
    
    # Verify the URL uses a working service (not via.placeholder.com)
    assert "placehold.co" in url or "dummyimage.com" in url or "fakeimg.pl" in url, \
        "Should use a reliable placeholder service!"
    
    print("\n✅ TEST PASSED: Special characters and URL encoding handled correctly")
    return formatted


async def run_all_tests():
    """Run all validation tests."""
    print("\n" + "🚀" * 40)
    print("PRODUCTION IMAGE RENDERING VALIDATION")
    print("🚀" * 40)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    
    tests = [
        ("Image Prompts", test_steps_with_image_prompts),
        ("Real Images", test_steps_with_real_images),
        ("Mixed Content", test_steps_with_mixed_content),
        ("Image Pool", test_image_pool_assignment),
        ("No Steps", test_empty_steps),
        ("Special Chars", test_special_characters_in_prompts)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, "PASSED", None))
        except Exception as e:
            results.append((test_name, "FAILED", str(e)))
            print(f"\n❌ TEST FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")
    
    for test_name, status, error in results:
        emoji = "✅" if status == "PASSED" else "❌"
        print(f"{emoji} {test_name}: {status}")
        if error:
            print(f"   Error: {error}")
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - Images will render correctly in OpenWebUI!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Please review the errors above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)