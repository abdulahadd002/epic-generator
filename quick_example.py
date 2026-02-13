"""
Quick Example - How to Use Your Trained AI Model
Just run this file to see the model in action!
"""

from src.inference import EpicStoryGenerator

print("="*80)
print("EPIC/STORY GENERATOR - QUICK EXAMPLE")
print("="*80)

# Step 1: Load the trained model
print("\n[1/3] Loading your trained AI model...")
generator = EpicStoryGenerator()

# Step 2: Define your project description
print("\n[2/3] Generating epic and user story from description...")
project_description = "Build a mobile app for tracking daily water intake with reminders"

print(f"\nYour input: '{project_description}'")

# Step 3: Generate the output
result = generator.generate_and_parse(project_description)

# Step 4: Display the results nicely
print("\n[3/3] Here's what the AI generated:")
generator.print_formatted_output(result)

# Also show the raw data
print("\nRaw output (as dictionary):")
print(f"  Epic: {result['epic']}")
print(f"  User Story: {result['user_story']}")
print(f"  Story Points: {result['story_points']}")
print(f"  Tasks: {result['tasks']}")
print(f"  Acceptance Criteria: {result['acceptance_criteria']}")

print("\n" + "="*80)
print("TRY IT YOURSELF!")
print("="*80)
print("\nTo generate your own:")
print("1. Edit line 17 with your project description")
print("2. Run: py -3.12 quick_example.py")
print("\nOr use interactive mode: py -3.12 src/inference.py")
print("="*80)
