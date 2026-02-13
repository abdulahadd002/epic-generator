"""
Retrain T5 Model using collected Gemini API outputs
This script fine-tunes the T5 model on data collected from Gemini generations
"""
import os
import sys
from datetime import datetime
from src.data_collector import TrainingDataCollector
from src.train_model import train_model


def retrain_t5_model(min_examples: int = 10):
    """
    Retrain T5 model using collected training data

    Args:
        min_examples: Minimum number of examples required for retraining
    """
    print("="*80)
    print("T5 MODEL RETRAINING")
    print("Using Gemini API outputs as training data")
    print("="*80)

    # Initialize data collector
    print("\n[1/4] Loading collected training data...")
    collector = TrainingDataCollector()
    stats = collector.get_stats()

    print(f"  Total examples collected: {stats['total_examples']}")
    print(f"  Last updated: {stats['last_updated']}")

    if stats['total_examples'] < min_examples:
        print(f"\n[ERROR] Not enough training examples!")
        print(f"  Required: {min_examples}")
        print(f"  Collected: {stats['total_examples']}")
        print(f"  Please generate more examples before retraining.")
        return False

    # Prepare data for training
    print(f"\n[2/4] Preparing {stats['total_examples']} examples for training...")
    source_file, target_file = collector.prepare_for_training()

    # Train model
    print("\n[3/4] Fine-tuning T5 model on collected data...")
    print("  This may take 30-60 minutes depending on the number of examples...")

    model_name = f"t5-retrained-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = f"d:/epic model/models/{model_name}"

    try:
        # Train the model
        train_model(
            source_file=source_file,
            target_file=target_file,
            model_name="t5-small",
            output_dir=output_dir,
            epochs=3,
            batch_size=4,
            learning_rate=1e-4,
            max_length=512
        )

        print(f"\n[4/4] Retraining complete!")
        print(f"  Model saved to: {output_dir}")
        print(f"  Examples used: {stats['total_examples']}")
        print(f"\nTo use the retrained model:")
        print(f"  1. Update web_app.py to point to: {output_dir}")
        print(f"  2. Restart the web server")

        return True

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        return False


def main():
    """Main function"""
    print("\n" + "="*80)
    print("T5 MODEL RETRAINING UTILITY")
    print("="*80)

    # Check if user wants to proceed
    print("\nThis will retrain the T5 model using Gemini API outputs.")
    print("The process may take 30-60 minutes.")

    response = input("\nProceed with retraining? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("Retraining cancelled.")
        return

    # Start retraining
    success = retrain_t5_model(min_examples=10)

    if success:
        print("\n" + "="*80)
        print("SUCCESS: T5 model has been retrained!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("FAILED: Retraining did not complete successfully")
        print("="*80)


if __name__ == "__main__":
    main()
