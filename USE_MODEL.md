# How to Use Your Trained AI Model

## Three Ways to Use the Model

### **Option 1: Interactive Mode (Easiest)**

Simply run the inference script and type your project descriptions:

```bash
py -3.12 src/inference.py
```

You'll see:
```
Enter your project description (or 'quit' to exit):
> Build a mobile app for tracking fitness goals
```

The model will generate the epic, user story, story points, and tasks!

---

### **Option 2: Quick Test Script**

Use the test script to try a single example:

```bash
py -3.12 test_inference.py
```

This will load the model and generate output for the built-in example.

---

### **Option 3: Use in Your Own Python Code (Most Flexible)**

Create your own Python script:

```python
from src.inference import EpicStoryGenerator

# 1. Load the trained model (only do this once)
print("Loading AI model...")
generator = EpicStoryGenerator()

# 2. Generate from your project description
description = "Build a mobile app for tracking fitness goals"

result = generator.generate_and_parse(description)

# 3. Use the results
print(f"Epic: {result['epic']}")
print(f"User Story: {result['user_story']}")
print(f"Story Points: {result['story_points']}")
print(f"Tasks: {result['tasks']}")
print(f"Acceptance Criteria: {result['acceptance_criteria']}")
```

---

## Simple Example Script

Here's a complete working example you can copy and run:

**File: `my_model_test.py`**
```python
from src.inference import EpicStoryGenerator

# Load model
generator = EpicStoryGenerator()

# Your project descriptions
projects = [
    "Build a mobile app for tracking fitness goals",
    "Create a dashboard for sales analytics",
    "Implement user authentication with OAuth",
    "Add real-time notifications to the app"
]

# Generate outputs for each
for description in projects:
    print("\n" + "="*80)
    print(f"INPUT: {description}")
    print("="*80)

    # Generate
    result = generator.generate_and_parse(description)

    # Display
    generator.print_formatted_output(result)
```

Run it:
```bash
py -3.12 my_model_test.py
```

---

## What You Get

The model returns a dictionary with:

```python
{
  "epic": "Category/type of the work",
  "user_story": "As a user, I want to...",
  "story_points": "2",
  "tasks": ["Task 1", "Task 2", ...],
  "acceptance_criteria": ["Criteria 1", "Criteria 2", ...]
}
```

---

## Advanced Usage

### Save Results to JSON

```python
import json

result = generator.generate_and_parse("Your description here")

# Save to file
with open('output.json', 'w') as f:
    json.dump(result, f, indent=2)
```

### Batch Processing

```python
# Process multiple descriptions
descriptions = [
    "Build feature A",
    "Build feature B",
    "Build feature C"
]

results = []
for desc in descriptions:
    result = generator.generate_and_parse(desc)
    results.append(result)

# Save all results
with open('all_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Adjust Generation Quality

```python
# Higher num_beams = better quality but slower
result = generator.generate(
    "Your description",
    num_beams=8,        # Default is 4
    max_length=512      # Default is 256
)
```

---

## Tips

1. **Be specific**: More detailed descriptions → better outputs
   - ❌ "Build app"
   - ✅ "Build a mobile app for iOS with user login and profile management"

2. **Use natural language**: Write like you're talking to a PM
   - ✅ "Create a dashboard showing sales metrics by region"
   - ✅ "Add a feature to export reports as PDF"

3. **First load is slow**: Model loads in ~2 seconds, but only needs to load once

4. **Generation is fast**: 1-2 seconds per description after loading

---

## Troubleshooting

### "Model not found"
- Make sure you're in the `d:\epic model` directory
- Check that `models/epic-story-model/final` exists

### "CUDA out of memory"
- Close other GPU-using applications
- Reduce batch size if processing multiple at once

### "Import error"
- Make sure you're using Python 3.12: `py -3.12`
- Check that libraries are installed: `pip list`

---

## Next Steps

- Try the interactive mode to experiment
- Create your own script for your specific use case
- Integrate with your project management tools
- Fine-tune the model further with your own data

**Have fun with your AI model!** 🚀
