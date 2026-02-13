# T5 Model Training Guide

## Overview

This system now includes **automatic training data collection** from Gemini API outputs. Every time the Gemini API successfully generates documentation, that input-output pair is saved as training data for the T5 model.

## How It Works

```
User Input → Gemini API → Generated Output
                ↓
        Data Collector saves for T5 learning
                ↓
        Periodic retraining of T5 model
```

## Data Collection

### Automatic Collection
- Every successful Gemini generation is automatically saved
- No user action required
- Data stored in: `training_data/gemini_training_examples.jsonl`

### Collection Statistics
Check collection stats via the health endpoint:
```bash
curl http://localhost:5000/api/health
```

Response includes:
```json
{
  "data_collector": {
    "available": true,
    "status": "Active - Collecting Training Data",
    "total_examples": 25,
    "last_updated": "2026-02-13T17:00:00"
  }
}
```

## Retraining the T5 Model

### When to Retrain
- After collecting **10+ new examples** (minimum)
- Recommended: **50-100 examples** for significant improvement
- More data = better model performance

### How to Retrain

1. **Check collected examples:**
```bash
python -c "from src.data_collector import TrainingDataCollector; print(TrainingDataCollector().get_stats())"
```

2. **Run retraining script:**
```bash
python retrain_t5.py
```

3. **Follow prompts:**
   - Confirm you want to proceed
   - Wait 30-60 minutes for training to complete

4. **Update web_app.py** to use the new model:
```python
# Update the model path in web_app.py
model_path = "d:/epic model/models/t5-retrained-YYYYMMDD-HHMMSS/final"
```

5. **Restart the server:**
```bash
python web_app.py
```

## Training Data Format

Each training example is stored as JSON:
```json
{
  "timestamp": "2026-02-13T16:58:30",
  "input": "Build a fitness tracking app with workout logging",
  "output": "Epic E1: User Authentication\nDescription: ...",
  "generator": "Gemini API",
  "metadata": {
    "input_length": 54,
    "output_length": 2843
  }
}
```

## Benefits

1. **Continuous Improvement**: T5 model learns from high-quality Gemini outputs
2. **Offline Capability**: Trained T5 can work without API dependency
3. **Cost Reduction**: Once T5 is well-trained, can reduce Gemini API usage
4. **Customization**: Model learns your specific documentation style

## Training Parameters

Default settings in `retrain_t5.py`:
- **Epochs**: 3
- **Batch Size**: 4
- **Learning Rate**: 1e-4
- **Max Length**: 512 tokens
- **Model**: t5-small (60.5M parameters)

Adjust these based on:
- Available GPU memory
- Number of training examples
- Desired training time

## File Structure

```
epic-generator/
├── training_data/
│   ├── gemini_training_examples.jsonl  # Raw collected data
│   ├── collection_stats.json           # Statistics
│   └── processed/
│       ├── source.txt                  # Prepared inputs
│       └── target.txt                  # Prepared outputs
├── models/
│   ├── t5-retrained-20260213-170000/  # Retrained models
│   └── comprehensive-model/            # Original T5 model
└── src/
    ├── data_collector.py               # Collection system
    └── train_model.py                  # Training utilities
```

## Best Practices

1. **Diverse Training Data**: Generate documentation for various project types
2. **Quality Over Quantity**: Gemini outputs are high-quality, don't rush collection
3. **Regular Retraining**: Retrain every 50-100 new examples
4. **Backup Models**: Keep previous versions before retraining
5. **Test After Training**: Verify new model quality before production use

## Troubleshooting

### "Not enough training examples"
- Need at least 10 examples (configurable in `retrain_t5.py`)
- Generate more documentation to collect more data

### "Training failed: CUDA out of memory"
- Reduce batch size in `retrain_t5.py`
- Try: `batch_size=2` or `batch_size=1`

### "Model performance degraded"
- May need more training examples
- Try training for more epochs
- Revert to previous model version

## Monitoring

Track your collection progress:
```python
from src.data_collector import TrainingDataCollector

collector = TrainingDataCollector()
stats = collector.get_stats()
print(f"Examples: {stats['total_examples']}")
print(f"Last: {stats['last_updated']}")
```

## Future Enhancements

- Automatic retraining when threshold reached
- A/B testing between Gemini and T5 outputs
- User feedback collection for quality improvement
- Multi-model ensemble approach
