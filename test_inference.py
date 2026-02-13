"""Quick test of inference pipeline"""

from src.inference import EpicStoryGenerator

# Initialize generator
print("Loading model...")
generator = EpicStoryGenerator()

# Test with one example
description = "Build a real-time chat application with user authentication and message history"

print(f"\nInput: {description}")
print("\nGenerating...")

# Generate and parse
result = generator.generate_and_parse(description)

# Print formatted output
generator.print_formatted_output(result)

# Also print raw dictionary
print("\nRaw parsed output:")
import json
print(json.dumps(result, indent=2))
