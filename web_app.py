"""
Flask Web Application for Epic/Story Generator
Run this to start a web interface for your AI model
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from src.inference import EpicStoryGenerator
from src.gemini_generator import GeminiEpicGenerator
from src.data_collector import TrainingDataCollector
import os
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for API calls

# Configuration - Google Gemini API Key
# Set your API key as environment variable: GEMINI_API_KEY
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Debug: Check if API key is loaded
if GEMINI_API_KEY:
    print(f"[DEBUG] API Key loaded: {GEMINI_API_KEY[:25]}...")
    sys.stdout.flush()
else:
    print("[WARNING] No GEMINI_API_KEY found in environment!")
    sys.stdout.flush()

# Global model instances (load once when server starts)
print("="*80)
print("INITIALIZING AI GENERATORS")
print("="*80)

# Initialize Gemini API generator (primary)
print("\n[1/2] Initializing Google Gemini API Generator...")
try:
    gemini_generator = GeminiEpicGenerator(api_key=GEMINI_API_KEY)
    print("[SUCCESS] Gemini API Generator ready!")
except Exception as e:
    print(f"[ERROR] Gemini API initialization failed: {e}")
    gemini_generator = None

# Initialize T5 model (DISABLED for testing Gemini API only)
print("\n[2/2] T5 Model (DISABLED - Testing Gemini API only)...")
t5_generator = None
print("[INFO] T5 Model disabled - Gemini API exclusive mode")

# Initialize Training Data Collector
print("\n[3/3] Initializing Training Data Collector...")
try:
    data_collector = TrainingDataCollector()
    stats = data_collector.get_stats()
    print(f"[SUCCESS] Data Collector ready! ({stats['total_examples']} examples collected)")
except Exception as e:
    print(f"[ERROR] Data Collector initialization failed: {e}")
    data_collector = None

print("\n" + "="*80)
print("GENERATOR STATUS:")
print(f"  Gemini API (Primary):  {'[READY]' if gemini_generator else '[UNAVAILABLE]'}")
print(f"  T5 Model (Fallback):   [DISABLED for testing]")
print(f"  Data Collector:        {'[READY]' if data_collector else '[UNAVAILABLE]'}")
print("="*80)
print("\nWeb server starting...")


def reformat_model_output(raw_text: str, description: str) -> str:
    """
    Reformat raw model output into proper comprehensive format
    matching the Autonomous Solar Vehicle PDF structure
    """
    import re

    # Generate structured IDs
    epic_id = "E1"
    story_id = f"{epic_id}-US1"
    test_id = f"{story_id}-TC1"

    # Extract key phrases from description
    desc_lower = description.lower()

    # Determine epic title from description
    epic_title = description[:50] + "..." if len(description) > 50 else description
    epic_title = epic_title.title()

    # Format output matching PDF structure
    formatted = f"""Epic {epic_id}: {epic_title}
Description: As a system administrator, I want to {description.lower()} so that ensure reliable and scalable backend operations

User Story {story_id}: {epic_title}
Description: As a user, I want to {description.lower()} so that I can accomplish my goals efficiently
Story Points: 5
Acceptance Criteria: Given the user has necessary permissions, When the user performs the required input, Then the system should process the request successfully

Test Case ID: {test_id}
Test Case Description: Verify that {description.lower()} functions correctly
Input:
  - Preconditions: System initialized, all services running
  - Test Data: Valid input matching requirements
  - User Action: Execute primary function
Expected Result:
1. System accepts input and validates format within 500ms
2. Processing completes successfully with no errors
3. Expected output displayed with correct data
4. Success notification shown to user
5. System state updated correctly in database
6. Operation logged with timestamp and user details"""

    return formatted


def parse_multiple_epics(text: str) -> dict:
    """
    Parse comprehensive output with MULTIPLE Epics, User Stories, and Test Cases
    Returns a structured format for displaying all epics
    """
    import re

    result = {
        "epics": [],
        "raw_text": text
    }

    # Split text by Epic markers to find all epics
    epic_sections = re.split(r'(?=Epic E\d+:)', text)

    for section in epic_sections:
        if not section.strip() or 'Epic E' not in section:
            continue

        epic_data = {
            "epic_id": "",
            "epic_title": "",
            "epic_description": "",
            "user_stories": []
        }

        # Extract Epic ID and Title
        epic_match = re.search(r'Epic\s+(E\d+):\s*([^\n]+)', section, re.IGNORECASE)
        if epic_match:
            epic_data["epic_id"] = epic_match.group(1)
            epic_data["epic_title"] = epic_match.group(2).strip()

        # Extract Epic Description
        epic_desc_match = re.search(r'Description:\s*([^\n]+(?:\n(?!User Story|Epic E\d+)[^\n]+)*)', section, re.IGNORECASE)
        if epic_desc_match:
            epic_data["epic_description"] = epic_desc_match.group(1).strip()

        # Find all user stories in this epic
        story_sections = re.split(r'(?=User Story\s+E\d+-US\d+)', section)

        for story_section in story_sections:
            if not story_section.strip() or 'User Story' not in story_section:
                continue

            story_data = {
                "story_id": "",
                "story_title": "",
                "story_description": "",
                "story_points": "",
                "acceptance_criteria": "",
                "test_cases": []
            }

            # Extract User Story ID and Title
            story_match = re.search(r'User Story\s+(E\d+-US\d+):\s*([^\n]+)', story_section, re.IGNORECASE)
            if story_match:
                story_data["story_id"] = story_match.group(1)
                story_data["story_title"] = story_match.group(2).strip()

            # Extract User Story Description
            story_desc_match = re.search(r'User Story.*?Description:\s*([^\n]+(?:\n(?!Story Points|Acceptance|Test Case)[^\n]+)*)', story_section, re.DOTALL | re.IGNORECASE)
            if story_desc_match:
                story_data["story_description"] = story_desc_match.group(1).strip()

            # Extract Story Points
            points_match = re.search(r'Story Points:\s*(\d+)', story_section, re.IGNORECASE)
            if points_match:
                story_data["story_points"] = points_match.group(1)

            # Extract Acceptance Criteria
            ac_match = re.search(r'Acceptance Criteria:\s*([^\n]+(?:\n(?!Test Case|User Story|Epic)[^\n]+)*)', story_section, re.DOTALL | re.IGNORECASE)
            if ac_match:
                story_data["acceptance_criteria"] = ac_match.group(1).strip()

            # Find all test cases for this user story
            test_sections = re.findall(r'Test Case ID:\s*(E\d+-US\d+-TC\d+)\s*Test Case Description:\s*([^\n]+).*?Expected Result:\s*(.+?)(?=Test Case ID:|User Story|Epic E\d+|$)', story_section, re.DOTALL | re.IGNORECASE)

            for tc_match in test_sections:
                test_data = {
                    "test_case_id": tc_match[0],
                    "test_case_description": tc_match[1].strip(),
                    "expected_results": []
                }

                # Parse expected results
                expected_text = tc_match[2].strip()
                numbered_items = re.findall(r'(\d+)\.\s*([^\n]+(?:\n(?!\d+\.)[^\n]+)*)', expected_text)
                test_data["expected_results"] = [item[1].strip() for item in numbered_items]

                story_data["test_cases"].append(test_data)

            epic_data["user_stories"].append(story_data)

        result["epics"].append(epic_data)

    return result


def parse_comprehensive_output(text: str) -> dict:
    """
    Parse comprehensive model output with Epic IDs, User Stories, Test Cases
    Handles both T5 model output and Claude API output formats

    Expected format:
    Epic E1: Title
    Description: As a [role], I want [capability] so that [benefit]

    User Story E1-US1: Title
    Description: As a [role], I want [feature] so that [benefit]
    Story Points: X
    Acceptance Criteria: Given... When... Then...

    Test Case ID: E1-US1-TC1
    Test Case Description: ...
    Expected Result:
    1. ...
    2. ...
    """
    import re

    result = {
        "epic_id": "",
        "epic_title": "",
        "epic_description": "",
        "story_id": "",
        "story_title": "",
        "story_description": "",
        "story_points": "",
        "acceptance_criteria": "",
        "test_case_id": "",
        "test_case_description": "",
        "expected_results": [],
        "raw_text": text
    }

    # Extract Epic ID and Title (handle both plain and markdown bold format)
    epic_match = re.search(r'(?:\*\*)?Epic\s+(?:ID:\s*)?(E\d+)(?:\*\*)?:\s*(?:\*\*)?([^\n*]+?)(?:\*\*)?(?:\n|$)', text, re.IGNORECASE)
    if epic_match:
        result["epic_id"] = epic_match.group(1)
        result["epic_title"] = epic_match.group(2).strip()

    # Extract Epic Description (more flexible pattern)
    epic_desc_match = re.search(r'(?:\*\*)?Description(?:\*\*)?:\s*([^\n]+(?:\n(?!User Story|Epic)[^\n]+)*)', text, re.IGNORECASE)
    if epic_desc_match:
        result["epic_description"] = epic_desc_match.group(1).strip()

    # Extract User Story ID and Title
    story_match = re.search(r'(?:\*\*)?User Story\s+(?:ID:\s*)?(E\d+-US\d+)(?:\*\*)?:\s*(?:\*\*)?([^\n*]+?)(?:\*\*)?(?:\n|$)', text, re.IGNORECASE)
    if story_match:
        result["story_id"] = story_match.group(1)
        result["story_title"] = story_match.group(2).strip()

    # Extract User Story Description (after User Story section)
    story_desc_match = re.search(r'User Story.*?(?:\*\*)?Description(?:\*\*)?:\s*([^\n]+(?:\n(?!Story Points|Acceptance|Test Case|Epic)[^\n]+)*)', text, re.DOTALL | re.IGNORECASE)
    if story_desc_match:
        result["story_description"] = story_desc_match.group(1).strip()

    # Extract Story Points
    points_match = re.search(r'(?:\*\*)?Story Points(?:\*\*)?:\s*(\d+)', text, re.IGNORECASE)
    if points_match:
        result["story_points"] = points_match.group(1)

    # Extract Acceptance Criteria (more flexible to capture full content)
    ac_match = re.search(r'(?:\*\*)?Acceptance Criteria(?:\*\*)?:\s*([^\n]+(?:\n(?!Test Case|User Story|Epic)[^\n]+)*)', text, re.DOTALL | re.IGNORECASE)
    if ac_match:
        result["acceptance_criteria"] = ac_match.group(1).strip()

    # Extract Test Case ID
    tc_id_match = re.search(r'(?:\*\*)?Test Case (?:ID)?(?:\*\*)?:\s*(E\d+-US\d+-TC\d+)', text, re.IGNORECASE)
    if tc_id_match:
        result["test_case_id"] = tc_id_match.group(1)

    # Extract Test Case Description
    tc_desc_match = re.search(r'(?:\*\*)?Test Case Description(?:\*\*)?:\s*([^\n]+)', text, re.IGNORECASE)
    if tc_desc_match:
        result["test_case_description"] = tc_desc_match.group(1).strip()

    # Extract Expected Results (numbered list, handle various formats)
    expected_section = re.search(r'(?:\*\*)?Expected Result(?:s)?(?:\*\*)?:\s*(.+?)(?=\n(?:Epic|User Story|Test Case ID:|$)|\Z)', text, re.DOTALL | re.IGNORECASE)
    if expected_section:
        expected_text = expected_section.group(1).strip()
        # Find numbered items (1. 2. 3. etc), capturing full lines including those that wrap
        numbered_items = re.findall(r'(\d+)\.\s*([^\n]+(?:\n(?!\d+\.)[^\n]+)*)', expected_text)
        result["expected_results"] = [item[1].strip() for item in numbered_items]

    return result


@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    API endpoint to generate epic/story from description using Claude API or T5 model

    Request JSON:
    {
        "description": "Your project description here",
        "use_t5": false  // Optional: force T5 model usage
    }

    Response JSON:
    {
        "success": true,
        "result": {
            "epic": "...",
            "user_story": "...",
            "story_points": "...",
            "tasks": [...],
            "acceptance_criteria": [...]
        },
        "raw_output": "...",
        "generator_used": "Claude API" or "T5 Model"
    }
    """
    try:
        # Get description from request
        data = request.get_json()
        description = data.get('description', '').strip()
        force_t5 = data.get('use_t5', False)

        if not description:
            return jsonify({
                'success': False,
                'error': 'Please provide a project description'
            }), 400

        generator_used = None
        formatted_output = None

        # Try Gemini API first (unless T5 is explicitly requested)
        if not force_t5 and gemini_generator:
            try:
                print(f"\n[API] Using Gemini API for generation...")
                sys.stdout.flush()
                # Generate using Gemini API with quick summary (5 epics, 2 stories each)
                result = gemini_generator.generate_quick_summary(description)

                if result.get("success"):
                    formatted_output = result["raw_output"]
                    generator_used = "Gemini API"
                    print(f"[API] [SUCCESS] Gemini API generation successful")
                    sys.stdout.flush()

                    # Save training data for T5 model learning
                    if data_collector:
                        data_collector.save_training_example(
                            project_description=description,
                            generated_output=formatted_output,
                            generator_used="Gemini API"
                        )
                        sys.stdout.flush()
                else:
                    print(f"[API] [ERROR] Gemini API generation failed: {result.get('error')}")
                    sys.stdout.flush()
                    # Fall back to T5
                    raise Exception("Gemini API generation failed")

            except Exception as e:
                print(f"[API] Gemini API error: {e}, falling back to T5 model...")
                sys.stdout.flush()
                formatted_output = None

        # T5 model disabled for testing - Gemini API only
        # Fall back disabled to test Gemini API exclusively

        # If Gemini API failed
        if formatted_output is None:
            error_msg = 'Gemini API generation failed. T5 model is disabled for testing. Check server logs.'
            print(f"[ERROR] {error_msg}")
            sys.stdout.flush()
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500

        # Parse comprehensive output
        # Use multi-epic parser for Gemini API (better format), single-epic for T5
        if generator_used == "Gemini API":
            result = parse_multiple_epics(formatted_output)
        else:
            # T5 model - use old single-epic parser for compatibility
            result = parse_comprehensive_output(formatted_output)

        # Return results with formatted output
        return jsonify({
            'success': True,
            'result': result,
            'raw_output': formatted_output,
            'generator_used': generator_used
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example project descriptions"""
    examples = [
        "Build a real-time chat application with user authentication and message history",
        "Create a mobile app for restaurant reservations with real-time availability",
        "Develop an e-commerce platform with product catalog, shopping cart, and checkout",
        "Build a task management system with teams, projects, and deadlines",
        "Create a dashboard for sales analytics with charts and reports",
        "Implement user authentication with OAuth and social login",
        "Add real-time notifications to the mobile app",
        "Build a REST API for customer data management",
        "Create a payment processing system with Stripe integration",
        "Develop a content management system with role-based permissions"
    ]

    return jsonify({
        'success': True,
        'examples': examples
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    collector_stats = data_collector.get_stats() if data_collector else {}

    return jsonify({
        'success': True,
        'status': 'running',
        'generators': {
            'gemini_api': {
                'available': gemini_generator is not None,
                'model': 'Gemini 2.5 Flash',
                'status': 'Primary Generator' if gemini_generator else 'Unavailable'
            },
            't5_model': {
                'available': t5_generator is not None,
                'model': 'T5-Small Comprehensive',
                'parameters': '60.5M',
                'status': 'Fallback Generator' if t5_generator else 'Unavailable',
                'model_path': 'd:/epic model/models/comprehensive-model/final'
            },
            'data_collector': {
                'available': data_collector is not None,
                'status': 'Active - Collecting Training Data' if data_collector else 'Unavailable',
                'total_examples': collector_stats.get('total_examples', 0),
                'last_updated': collector_stats.get('last_updated')
            }
        },
        'primary_generator': 'Gemini API' if gemini_generator else ('T5 Model' if t5_generator else 'None')
    })


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    print("\n" + "="*80)
    print("EPIC/STORY GENERATOR WEB APP")
    print("="*80)
    print("\nServer running at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("\n" + "="*80 + "\n")

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
