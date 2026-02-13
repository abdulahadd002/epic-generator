# 🚀 Enhanced Epic/Story/Test Case Generator

## Two Generation Options

You now have **two ways** to generate project documentation:

### 1. ⚡ **T5 Model** (Fast, Free, Offline)
- Your trained 60M parameter model
- Runs on your GPU
- Generates basic epics and user stories
- **FREE** - No API costs
- **FAST** - 1-2 seconds
- **LIMITED** - Basic format only

### 2. 🎯 **Claude API** (Comprehensive, Professional)
- Uses Anthropic's Claude Sonnet 4
- Generates detailed documentation like your PDF
- **Comprehensive** - Epics, Stories, Acceptance Criteria, Test Cases
- **Professional** - Production-ready format
- **Requires API key** - Costs ~$0.01-0.05 per generation
- Takes 10-20 seconds

---

## 📊 **Format Comparison**

### T5 Model Output:
```
EPIC: Mobile
USER_STORY: As a user, I want to build a mobile app
STORY_POINTS: 2
```

### Claude API Output (matches your PDF):
```
**Epic ID:** E1
**Epic Title:** Autonomous Navigation and Path Planning
**Description:** As a system administrator, I want the vehicle to navigate
autonomously between specified locations within controlled environments so that
it can transport passengers or goods without human intervention.

**User Story ID:** E1-US1
**User Story Title:** Destination Input and Route Planning
**Description:** As a passenger, I want to input my destination through a
touchscreen interface so that the vehicle can automatically calculate and
follow the optimal route.

**Acceptance Criteria:**
- Given the vehicle is in standby mode
- When I select a destination from the available locations
- Then the system should display the route, estimate time of arrival, and
  begin navigation upon confirmation

**Test Case ID:** E1-US1-TC1
**Test Case Description:** Verify that the touchscreen interface correctly
accepts destination input and calculates optimal routes

**Input:**
- Preconditions: Vehicle in standby mode, touchscreen active
- Input: Select "Engineering Building Block A" from destination list
- User Action: Press "Confirm" button

**Expected Result:**
1. Route displayed on map screen
2. Estimated time of arrival shown (e.g., "5 minutes")
3. Total distance displayed (e.g., "1.2 km")
4. "Start Navigation" button appears
5. Navigation begins after confirmation
```

---

## 🔧 **Setup: Claude API Enhanced Generator**

### Step 1: Get Anthropic API Key

1. Go to https://console.anthropic.com/
2. Sign up / Log in
3. Navigate to **API Keys**
4. Create a new key
5. Copy your key (starts with `sk-ant-...`)

### Step 2: Set Environment Variable

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**Permanent (Windows):**
```
System Properties → Environment Variables → New →
  Variable: ANTHROPIC_API_KEY
  Value: sk-ant-your-key-here
```

### Step 3: Test It

```bash
py -3.12 src/enhanced_generator.py
```

---

## 💻 **Usage**

### **Option A: Command Line**

```python
from src.enhanced_generator import EnhancedEpicGenerator

# Initialize (reads ANTHROPIC_API_KEY from environment)
generator = EnhancedEpicGenerator()

# Generate comprehensive documentation
result = generator.generate_comprehensive_documentation(
    project_description="Build a mobile app for tracking fitness goals",
    num_epics=3,
    num_stories_per_epic=2,
    include_test_cases=True
)

# Print the result
print(result["documentation"]["formatted_text"])

# Export to markdown file
generator.export_to_markdown(result, "my_project.md")
```

### **Option B: Quick Summary**

```python
# Generate 1 epic with 3 stories and test cases
result = generator.generate_quick_summary(
    "Create a dashboard for sales analytics"
)
```

---

## 📁 **Complete Example**

Create a file `generate_my_project.py`:

```python
from src.enhanced_generator import EnhancedEpicGenerator

# Your project description
description = """
Build an autonomous solar vehicle that can navigate campus environments
without human intervention, powered entirely by renewable solar energy.
The system should include obstacle detection, GPS navigation, and
remote monitoring capabilities.
"""

# Initialize generator
generator = EnhancedEpicGenerator()

# Generate complete documentation
print("Generating comprehensive documentation...")
print("This will take 15-20 seconds...\n")

result = generator.generate_comprehensive_documentation(
    project_description=description,
    num_epics=4,  # 4 major system capabilities
    num_stories_per_epic=3,  # 3 user stories per epic
    include_test_cases=True  # Include detailed test cases
)

if result["success"]:
    # Save to file
    generator.export_to_markdown(result, "solar_vehicle_requirements.md")
    print("✓ Documentation generated successfully!")
    print("✓ Saved to: solar_vehicle_requirements.md")
else:
    print(f"✗ Error: {result['error']}")
```

Run it:
```bash
py -3.12 generate_my_project.py
```

---

## 💰 **Cost Estimation**

Claude API pricing (as of 2025):
- **Input:** $3 per million tokens
- **Output:** $15 per million tokens

Typical generation:
- Input: ~1,000 tokens (your prompt)
- Output: ~3,000 tokens (comprehensive docs)
- **Cost: ~$0.048 per generation** (less than 5 cents!)

For 100 project descriptions: **~$4.80**

---

## 🎯 **When to Use Each**

### Use **T5 Model** when:
- ✅ You need quick, basic epics and story points
- ✅ You're generating hundreds of descriptions
- ✅ You're offline or don't want API costs
- ✅ Simple format is sufficient

### Use **Claude API** when:
- ✅ You need professional, production-ready documentation
- ✅ You want detailed test cases like your PDF
- ✅ You're creating actual project specifications
- ✅ Quality > Speed
- ✅ Cost isn't a concern (< 5 cents per generation)

---

## 🔄 **Integration with Web App**

Want to add Claude API to the web interface? Create `web_app_enhanced.py`:

```python
from flask import Flask, render_template, request, jsonify
from src.inference import EpicStoryGenerator  # T5 model
from src.enhanced_generator import EnhancedEpicGenerator  # Claude API
import os

app = Flask(__name__)

# Load both generators
t5_generator = EpicStoryGenerator()
claude_generator = EnhancedEpicGenerator() if os.getenv("ANTHROPIC_API_KEY") else None

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json()
    description = data.get('description')
    mode = data.get('mode', 'basic')  # 'basic' or 'comprehensive'

    if mode == 'comprehensive' and claude_generator:
        # Use Claude API for detailed docs
        result = claude_generator.generate_comprehensive_documentation(
            description,
            num_epics=2,
            num_stories_per_epic=2,
            include_test_cases=True
        )
        return jsonify(result)
    else:
        # Use T5 model for basic generation
        result = t5_generator.generate_and_parse(description)
        return jsonify({"success": True, "result": result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## 📖 **Example Output Files**

After running the generator, you'll get markdown files like:

**`solar_vehicle_requirements.md`**
```markdown
# Project Documentation

## Project Description
Build an autonomous solar vehicle...

---

## Epic E1: Autonomous Navigation and Path Planning

**Description:** As a system administrator, I want the vehicle...

### User Story E1-US1: Destination Input

**Description:** As a passenger, I want to input my destination...

**Acceptance Criteria:**
- Given the vehicle is in standby mode
- When I select a destination
- Then the system should display the route

### Test Case E1-US1-TC1

**Test Case Description:** Verify destination input functionality

**Input:**
- Preconditions: Vehicle in standby, touchscreen active
- Test Scenario: Select destination from list

**Expected Result:**
1. Route displayed on screen
2. ETA calculated and shown
3. Distance displayed
4. Navigation begins on confirmation
...
```

---

## 🛠️ **Troubleshooting**

### "ANTHROPIC_API_KEY environment variable not set"
**Solution:** Set your API key in environment variables (see Step 2 above)

### "API key is invalid"
**Solution:** Check your key at https://console.anthropic.com/settings/keys

### "Rate limit exceeded"
**Solution:** You're making too many requests. Wait 60 seconds.

### "Insufficient credits"
**Solution:** Add credits at https://console.anthropic.com/settings/billing

---

## 📚 **Resources**

- **Anthropic API Docs:** https://docs.anthropic.com/
- **Claude Models:** https://www.anthropic.com/api
- **Pricing:** https://www.anthropic.com/pricing
- **API Console:** https://console.anthropic.com/

---

## ⚙️ **Advanced Configuration**

### Custom Prompts

Edit `src/enhanced_generator.py` to customize the generation:

```python
# Change the model
model="claude-opus-4-20250514"  # More capable, slower, more expensive

# Adjust output length
max_tokens=16000  # For very detailed docs

# Control creativity
temperature=0.3  # More deterministic (less creative)
temperature=1.0  # More creative (more variation)
```

### Batch Processing

```python
descriptions = [
    "Project 1 description...",
    "Project 2 description...",
    "Project 3 description...",
]

for i, desc in enumerate(descriptions):
    result = generator.generate_comprehensive_documentation(desc)
    generator.export_to_markdown(result, f"project_{i+1}.md")
    print(f"Generated {i+1}/{len(descriptions)}")
```

---

## 🎉 **Summary**

You now have **TWO powerful tools**:

1. **T5 Model** - Your trained model for fast, free, basic generation
2. **Claude API** - Professional, comprehensive documentation matching your PDF format

Use the T5 model for quick iterations, and Claude API when you need production-ready, detailed documentation with epics, user stories, acceptance criteria, and test cases!

**Happy generating!** 🚀
