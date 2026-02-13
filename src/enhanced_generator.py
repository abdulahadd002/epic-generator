"""
Enhanced Epic/Story/Test Case Generator using Anthropic Claude API
Generates detailed outputs matching the format from the Autonomous Solar Vehicle proposal
"""

import anthropic
import json
import os
from typing import Dict, List


class EnhancedEpicGenerator:
    """
    Uses Anthropic's Claude API to generate comprehensive project documentation
    including epics, user stories, acceptance criteria, and test cases
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the enhanced generator with Claude API

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY environment variable)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter"
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)

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
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",  # Latest Claude Sonnet
                max_tokens=8000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Parse the response
            response_text = message.content[0].text
            result = self._parse_response(response_text)

            return {
                "success": True,
                "project_description": project_description,
                "documentation": result,
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
        """Build the prompt for Claude API"""

        test_case_instruction = ""
        if include_test_cases:
            test_case_instruction = """
### TEST CASES (for each user story):
For each user story, create 1-2 detailed test cases in this format:

**Test Case ID:** [Epic-US-TC format, e.g., E1-US1-TC1]
**Test Case Description:** [What functionality is being tested]
**Input:**
- Preconditions: [System state requirements]
- Vehicle/System State: [Current state]
- Test Scenario: [Specific test setup]

**Expected Result:**
1. [First expected outcome - be specific]
2. [Second expected outcome - include metrics]
3. [Third expected outcome - include timings if relevant]
4. [Fourth expected outcome - include UI/UX details]
5. [Fifth expected outcome - include error handling]
6. [Sixth expected outcome - include completion state]
"""

        prompt = f"""You are a senior software architect creating comprehensive project documentation.

Given this project description:
"{description}"

Generate {num_epics} EPICS, each with {num_stories_per_epic} USER STORIES, following this EXACT format:

### EPIC FORMAT:
**Epic ID:** E[number]
**Epic Title:** [Descriptive title]
**Description:** As a [stakeholder role], I want [high-level capability] so that [business value/benefit].

### USER STORY FORMAT (for each epic):
**User Story ID:** E[epic#]-US[story#]
**User Story Title:** [Descriptive title]
**Description:** As a [user role], I want [specific feature] so that [user benefit].
**Acceptance Criteria:**
- Given [initial context/precondition]
- When [action or event occurs]
- Then [expected system behavior/outcome]

{test_case_instruction}

IMPORTANT GUIDELINES:
1. Make epics represent major system capabilities
2. User stories should be specific, measurable, and testable
3. Acceptance criteria must use Given/When/Then format
4. Test cases should include specific metrics, timings, and measurable outcomes
5. Cover different stakeholder perspectives (end users, operators, admins, safety officers)
6. Include both positive and negative test scenarios
7. Be detailed and comprehensive like professional software requirements

OUTPUT FORMAT:
Organize your response with clear markdown headings for each epic, user story, and test case.
Use bold text for labels (Epic ID, Description, etc.).
Number all lists clearly.

Begin your response now:"""

        return prompt

    def _parse_response(self, response_text: str) -> Dict:
        """
        Parse Claude's response into structured format

        Args:
            response_text: Raw text response from Claude

        Returns:
            Structured dictionary with epics, stories, and test cases
        """
        # For now, return the formatted text
        # You could add more sophisticated parsing here
        return {
            "formatted_text": response_text,
            "format": "markdown"
        }

    def generate_quick_summary(self, project_description: str) -> Dict:
        """
        Generate a quick 1-epic, 3-story summary

        Args:
            project_description: Project description

        Returns:
            Simplified documentation
        """
        return self.generate_comprehensive_documentation(
            project_description,
            num_epics=1,
            num_stories_per_epic=3,
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
*Generated using Enhanced Epic Generator with Claude API*
"""

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Documentation exported to: {filename}")


def main():
    """Example usage"""
    import sys

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("\nTo use this enhanced generator:")
        print('1. Get your API key from: https://console.anthropic.com/')
        print('2. Set environment variable: set ANTHROPIC_API_KEY=your-key-here')
        print('3. Run this script again')
        sys.exit(1)

    # Initialize generator
    generator = EnhancedEpicGenerator(api_key=api_key)

    # Example project descriptions
    examples = [
        "Build a mobile app for tracking fitness goals with workout plans and progress charts",
        "Create a real-time chat application with end-to-end encryption and file sharing",
        "Develop an e-commerce platform with product recommendations and payment processing"
    ]

    print("="*80)
    print("ENHANCED EPIC/STORY/TEST CASE GENERATOR")
    print("Using Anthropic Claude API for comprehensive documentation")
    print("="*80)

    # Generate for first example
    description = examples[0]
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
        output_file = "d:/epic model/output/enhanced_documentation.md"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        generator.export_to_markdown(result, output_file)

        print(f"\n✓ Full documentation saved to: {output_file}")
    else:
        print(f"ERROR: {result['error']}")


if __name__ == "__main__":
    main()
