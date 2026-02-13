"""
Test the newly trained comprehensive T5 model
Tests if it generates epics, user stories, acceptance criteria, and test cases
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def test_comprehensive_model():
    """Test the comprehensive model with sample inputs"""

    print("="*80)
    print("TESTING COMPREHENSIVE T5 MODEL")
    print("="*80)

    # Load model and tokenizer
    model_path = "d:/epic model/models/comprehensive-model/final"

    print(f"\nLoading model from: {model_path}")
    print("This may take a moment...")

    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model loaded successfully on {device}!")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test with sample project descriptions
    test_cases = [
        "Build a mobile app for tracking daily water intake with reminders and analytics",
        "Create a task management system with user authentication and real-time collaboration",
        "Develop an e-commerce platform with product catalog, shopping cart, and payment processing"
    ]

    for i, description in enumerate(test_cases, 1):
        print("\n" + "="*80)
        print(f"TEST CASE {i}")
        print("="*80)
        print(f"Input: {description}\n")

        # Prepare input with T5 prefix
        input_text = f"generate comprehensive project documentation: {description}"

        # Tokenize
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        # Generate
        print("Generating (this may take 10-20 seconds)...")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=0.7
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("\n" + "-"*80)
        print("GENERATED OUTPUT:")
        print("-"*80)
        print(generated_text)
        print("-"*80)

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print("\nModel Location:", model_path)
    print("\nThe model has been successfully loaded and tested!")
    print("You can now use it in your applications or web interface.")


if __name__ == "__main__":
    test_comprehensive_model()
