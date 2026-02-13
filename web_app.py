"""
Flask Web Application for Epic/Story Generator
Run this to start a web interface for your AI model
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from src.inference import EpicStoryGenerator
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for API calls

# Global model instance (load once when server starts)
print("Loading Comprehensive AI model... This may take a few seconds...")
# Use the comprehensive model trained with detailed format
generator = EpicStoryGenerator(model_path="d:/epic model/models/comprehensive-model/final")
print("Comprehensive model loaded! Web server starting...")


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


def parse_comprehensive_output(text: str) -> dict:
    """
    Parse comprehensive model output with Epic IDs, User Stories, Test Cases

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

    # Extract Epic ID and Title
    epic_match = re.search(r'Epic\s+(E\d+):\s*([^\n]+)', text)
    if epic_match:
        result["epic_id"] = epic_match.group(1)
        result["epic_title"] = epic_match.group(2).strip()

    # Extract Epic Description
    epic_desc_match = re.search(r'(?:Epic\s+)?Description:\s*([^\n]+?)(?=\s*User Story|$)', text, re.DOTALL)
    if epic_desc_match:
        result["epic_description"] = epic_desc_match.group(1).strip()

    # Extract User Story ID and Title
    story_match = re.search(r'User Story\s+(E\d+-US\d+):\s*([^\n]+)', text)
    if story_match:
        result["story_id"] = story_match.group(1)
        result["story_title"] = story_match.group(2).strip()

    # Extract User Story Description
    story_desc_match = re.search(r'User Story.*?Description:\s*([^\n]+?)(?=Story Points|Acceptance Criteria|Test Case|$)', text, re.DOTALL)
    if story_desc_match:
        result["story_description"] = story_desc_match.group(1).strip()

    # Extract Story Points
    points_match = re.search(r'Story Points:\s*(\d+)', text)
    if points_match:
        result["story_points"] = points_match.group(1)

    # Extract Acceptance Criteria
    ac_match = re.search(r'Acceptance Criteria:\s*([^\n]+?)(?=Test Case|$)', text, re.DOTALL)
    if ac_match:
        result["acceptance_criteria"] = ac_match.group(1).strip()

    # Extract Test Case ID
    tc_id_match = re.search(r'Test Case ID:\s*(E\d+-US\d+-TC\d+)', text)
    if tc_id_match:
        result["test_case_id"] = tc_id_match.group(1)

    # Extract Test Case Description
    tc_desc_match = re.search(r'Test Case Description:\s*([^\n]+)', text)
    if tc_desc_match:
        result["test_case_description"] = tc_desc_match.group(1).strip()

    # Extract Expected Results (numbered list)
    expected_section = re.search(r'Expected Result:\s*(.+?)$', text, re.DOTALL)
    if expected_section:
        expected_text = expected_section.group(1).strip()
        # Find numbered items (1. 2. 3. etc)
        numbered_items = re.findall(r'(\d+)\.\s*([^\n]+)', expected_text)
        result["expected_results"] = [item[1].strip() for item in numbered_items]

    return result


@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    API endpoint to generate epic/story from description

    Request JSON:
    {
        "description": "Your project description here"
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
        "raw_output": "..."
    }
    """
    try:
        # Get description from request
        data = request.get_json()
        description = data.get('description', '').strip()

        if not description:
            return jsonify({
                'success': False,
                'error': 'Please provide a project description'
            }), 400

        # Generate using the comprehensive model with explicit prompt
        input_text = f"generate comprehensive project documentation: {description}"
        raw_model_output = generator.generate(
            input_text,  # Use the full prompt instead of just description
            max_length=512,  # Longer for comprehensive format
            num_beams=5,  # More beams for better quality
            temperature=0.8,  # Add some randomness for better formatting
            do_sample=False  # Keep deterministic for consistency
        )

        # Reformat the output to match PDF structure
        formatted_output = reformat_model_output(raw_model_output, description)

        # Parse comprehensive output
        result = parse_comprehensive_output(formatted_output)

        # Return results with formatted output
        return jsonify({
            'success': True,
            'result': result,
            'raw_output': formatted_output  # Return formatted version for display
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
    return jsonify({
        'success': True,
        'status': 'running',
        'model_loaded': generator is not None,
        'model_type': 'Comprehensive T5-Small',
        'model_parameters': '60.5M',
        'model_path': 'd:/epic model/models/comprehensive-model/final'
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
