"""
Gemini-based Epic/Story/Test Case Generator using Google Generative AI
Generates detailed outputs matching the format from the Autonomous Solar Vehicle proposal
"""

import google.generativeai as genai
import json
import os
from typing import Dict, List


class GeminiEpicGenerator:
    """
    Uses Google's Gemini API to generate comprehensive project documentation
    including epics, user stories, acceptance criteria, and test cases
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the Gemini generator

        Args:
            api_key: Google API key (or set GEMINI_API_KEY environment variable)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter"
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_comprehensive_documentation(
        self,
        project_description: str,
        num_epics: int = 3,
        num_stories_per_epic: int = 2,
        include_test_cases: bool = True
    ) -> Dict:
        """
        Generate complete project documentation from a description

        Args:
            project_description: High-level project description
            num_epics: Number of epics to generate (default: 3)
            num_stories_per_epic: User stories per epic (default: 2)
            include_test_cases: Whether to include detailed test cases

        Returns:
            Dictionary containing structured documentation
        """
        prompt = self._build_comprehensive_prompt(
            project_description,
            num_epics,
            num_stories_per_epic,
            include_test_cases
        )

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            return {
                "success": True,
                "project_description": project_description,
                "documentation": {"formatted_text": response_text, "format": "markdown"},
                "raw_output": response_text
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "project_description": project_description
            }

    def _build_comprehensive_prompt(
        self,
        description: str,
        num_epics: int,
        num_stories_per_epic: int,
        include_test_cases: bool
    ) -> str:
        """Build the prompt for Gemini API - generates specific, detailed documentation"""

        prompt = f"""You are a senior software architect creating comprehensive, SPECIFIC project documentation.

CRITICAL INSTRUCTIONS:
- Read the ENTIRE project description carefully and ANALYZE all features mentioned
- BREAK DOWN the project into {num_epics} DISTINCT major feature areas (Epics)
- Each Epic must represent a DIFFERENT major system capability
- Generate SPECIFIC, DETAILED content based on the ACTUAL requirements provided
- DO NOT use generic placeholders or vague descriptions
- Every Epic, User Story, and Test Case must be directly relevant to the project description
- Use actual feature names, specific metrics, and concrete acceptance criteria from the description

ANALYSIS INSTRUCTIONS:
1. Identify {num_epics} major feature categories from the project description
2. Each Epic should cover a distinct functional area (e.g., Authentication, Dashboard, Data Entry, Analytics, etc.)
3. Number Epics sequentially: E1, E2, E3, E4, E5...
4. User Stories should be numbered per Epic: E1-US1, E1-US2, E2-US1, E2-US2, etc.
5. Test Cases should follow User Story IDs: E1-US1-TC1, E1-US2-TC1, E2-US1-TC1, etc.

PROJECT DESCRIPTION:
{description}

Generate {num_epics} comprehensive EPICs (E1 through E{num_epics}), each representing a DIFFERENT major feature area.
Each Epic must have {num_stories_per_epic} detailed USER STORIES.
Output must follow this EXACT format (plain text, not markdown):

Epic E[number]: [Specific Title Based on Project Description]
Description: As a [specific stakeholder role from project], I want [specific high-level capability mentioned in project] so that [actual business value from requirements]

User Story E[epic#]-US[story#]: [Specific Feature Title from Project]
Description: As a [specific user role from project], I want [exact feature from project description] so that [actual user benefit from requirements]
Story Points: [Appropriate number based on complexity: 1-13]
Acceptance Criteria: Given [specific initial context from project], When [specific action from requirements], Then [specific expected behavior with metrics from project]

Test Case ID: E[epic#]-US[story#]-TC1
Test Case Description: Verify that [specific functionality from project description] works as specified
Input:
  - Preconditions: [Specific system requirements from project description]
  - Test Data: [Actual data examples relevant to the feature]
  - User Action: [Specific action from project requirements]
Expected Result:
1. [Specific outcome with actual metrics from project - e.g., "User registration completes within 2 seconds"]
2. [Specific validation based on project - e.g., "Dashboard displays user's daily calorie intake of 2000 calories"]
3. [Specific UI behavior from requirements - e.g., "Success notification appears with message: 'Workout logged successfully'"]
4. [Specific data persistence requirement - e.g., "Workout entry saved to database with timestamp, duration, and calories burned"]
5. [Specific error handling from project - e.g., "If network fails, display: 'Unable to sync. Data saved locally'"]
6. [Specific completion state - e.g., "User redirected to dashboard showing updated statistics"]

EXAMPLE OF GOOD (SPECIFIC) vs BAD (GENERIC):
BAD: "System accepts input and validates format within 500ms"
GOOD: "Fitness app accepts workout log entry (exercise type, duration, calories) and validates all fields are filled within 300ms"

BAD: "As a user, I want to log data"
GOOD: "As a fitness app user, I want to log my cardio workout (running, duration: 30 minutes, calories: 350) so that I can track my daily calorie burn progress"

Now generate the documentation using ONLY specific details from the project description above:"""

        return prompt

    def _count_features(self, description: str) -> int:
        """
        Analyze project description and count distinct features

        Args:
            description: Project description text

        Returns:
            Number of features detected (minimum 3, maximum 15)
        """
        import re

        # Count numbered features (1. 2. 3. etc.)
        numbered_pattern = r'(?:^|\n)\s*\d+\.\s+[A-Z]'
        numbered_features = len(re.findall(numbered_pattern, description, re.MULTILINE))

        # Count bullet points/dashes (- Feature or * Feature)
        bullet_pattern = r'(?:^|\n)\s*[-*]\s+[A-Z]'
        bullet_features = len(re.findall(bullet_pattern, description, re.MULTILINE))

        # Count "Core Features Required:" section items
        core_features_match = re.search(r'Core Features Required:(.+?)(?=\n\n|\Z)', description, re.DOTALL | re.IGNORECASE)
        core_features_count = 0
        if core_features_match:
            core_section = core_features_match.group(1)
            core_features_count = len(re.findall(r'\d+\.', core_section))

        # Use the highest count found
        feature_count = max(numbered_features, bullet_features, core_features_count)

        # Default to 5 if no clear structure found, otherwise clamp between 3-15
        if feature_count == 0:
            return 5
        return max(3, min(15, feature_count))

    def generate_quick_summary(self, project_description: str) -> Dict:
        """
        Generate comprehensive documentation with multiple epics
        Analyzes the description and breaks it into multiple major feature areas

        Args:
            project_description: Project description

        Returns:
            Comprehensive documentation with multiple epics
        """
        # Dynamically determine number of epics based on features in description
        num_epics = self._count_features(project_description)

        return self.generate_comprehensive_documentation(
            project_description,
            num_epics=num_epics,  # Dynamic epic count based on feature analysis
            num_stories_per_epic=2,  # 2 user stories per epic for detailed coverage
            include_test_cases=True
        )

    def export_to_markdown(self, result: Dict, filename: str):
        """
        Export the generated documentation to a markdown file

        Args:
            result: Result from generate_comprehensive_documentation
            filename: Output file path
        """
        if not result.get("success"):
            raise ValueError(f"Cannot export failed generation: {result.get('error')}")

        content = f"""# Project Documentation

## Project Description
{result['project_description']}

---

{result['documentation']['formatted_text']}

---
*Generated using Gemini Epic Generator with Google Generative AI*
"""

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Documentation exported to: {filename}")


def main():
    """Example usage"""
    import sys

    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("\nTo use this Gemini generator:")
        print('1. Get your API key from: https://makersuite.google.com/app/apikey')
        print('2. Set environment variable: set GEMINI_API_KEY=your-key-here')
        print('3. Run this script again')
        sys.exit(1)

    # Initialize generator
    generator = GeminiEpicGenerator(api_key=api_key)

    # Example project description
    description = "Build a mobile app for tracking fitness goals with workout plans and progress charts"

    print("="*80)
    print("GEMINI EPIC/STORY/TEST CASE GENERATOR")
    print("Using Google Gemini API for comprehensive documentation")
    print("="*80)

    print(f"\nGenerating documentation for:")
    print(f'  "{description}"')
    print("\nThis may take 10-20 seconds...\n")

    result = generator.generate_comprehensive_documentation(
        description,
        num_epics=2,
        num_stories_per_epic=2,
        include_test_cases=True
    )

    if result["success"]:
        print("="*80)
        print("GENERATED DOCUMENTATION")
        print("="*80)
        print(result["documentation"]["formatted_text"])
        print("\n" + "="*80)

        # Export to file
        output_file = "d:/epic model/output/gemini_documentation.md"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        generator.export_to_markdown(result, output_file)

        print(f"\n[SUCCESS] Full documentation saved to: {output_file}")
    else:
        print(f"ERROR: {result['error']}")


if __name__ == "__main__":
    main()
