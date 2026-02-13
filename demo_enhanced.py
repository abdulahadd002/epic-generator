"""
Demo script for Enhanced Epic Generator
Shows side-by-side comparison of T5 model vs Claude API
"""

import os
from src.inference import EpicStoryGenerator
from src.enhanced_generator import EnhancedEpicGenerator

def demo():
    """Run demonstration"""

    project_description = "Build a mobile app for tracking daily water intake with reminders and analytics"

    print("="*80)
    print("EPIC/STORY GENERATOR COMPARISON")
    print("="*80)
    print(f"\nProject: {project_description}\n")

    # ========== T5 MODEL ==========
    print("\n" + "="*80)
    print("METHOD 1: T5 MODEL (Fast, Free, Basic)")
    print("="*80)
    print("\nLoading T5 model...")

    t5_generator = EpicStoryGenerator()

    print("Generating (takes 1-2 seconds)...")
    t5_result = t5_generator.generate_and_parse(project_description)

    print("\n--- T5 Model Output ---")
    print(f"Epic: {t5_result['epic']}")
    print(f"User Story: {t5_result['user_story']}")
    print(f"Story Points: {t5_result['story_points']}")
    if t5_result['tasks']:
        print(f"Tasks: {t5_result['tasks']}")

    # ========== CLAUDE API ==========
    print("\n\n" + "="*80)
    print("METHOD 2: CLAUDE API (Comprehensive, Professional)")
    print("="*80)

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print("\n⚠️  ANTHROPIC_API_KEY not set!")
        print("\nTo see Claude API in action:")
        print("1. Get API key from: https://console.anthropic.com/")
        print("2. Set environment variable:")
        print('   set ANTHROPIC_API_KEY=your-key-here')
        print("3. Run this script again")
        print("\n" + "="*80)
        return

    print("\nInitializing Claude API...")
    claude_generator = EnhancedEpicGenerator(api_key=api_key)

    print("Generating (takes 10-20 seconds)...")
    print("This will generate: 2 Epics, 2 User Stories each, with Test Cases\n")

    claude_result = claude_generator.generate_comprehensive_documentation(
        project_description,
        num_epics=2,
        num_stories_per_epic=2,
        include_test_cases=True
    )

    if claude_result["success"]:
        print("\n--- Claude API Output ---")
        print(claude_result["documentation"]["formatted_text"])

        # Save to file
        output_file = "d:/epic model/output/demo_output.md"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        claude_generator.export_to_markdown(claude_result, output_file)

        print("\n" + "="*80)
        print(f"✓ Full output saved to: {output_file}")
        print("="*80)
    else:
        print(f"\n✗ Error: {claude_result['error']}")

    # ========== SUMMARY ==========
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
T5 MODEL:
  ✓ Fast (1-2 seconds)
  ✓ Free (no API costs)
  ✓ Works offline
  ✓ Basic format
  ✗ Limited detail

CLAUDE API:
  ✓ Comprehensive (like your PDF)
  ✓ Professional format
  ✓ Includes test cases
  ✓ Production-ready
  ✗ Requires API key
  ✗ Costs ~$0.05 per generation
  ✗ Takes 10-20 seconds

RECOMMENDATION:
- Use T5 for quick iterations and learning
- Use Claude API for real project documentation
""")

    print("="*80)


if __name__ == "__main__":
    demo()
