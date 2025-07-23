# CLI Tools for UI Component Annotation Evaluation

This directory contains command-line tools for generating test datasets and evaluating LLM predictions against ground truth annotations.

## 📊 Tools Overview

### 1. `generate_test_data.py` - Test Dataset Generator
Generates realistic test datasets with 100 ground truth annotation files and 100 corresponding LLM prediction files.

### 2. `evaluate_annotations.py` - Evaluation Tool
Evaluates LLM predictions against ground truth annotations and provides detailed metrics for each UI component tag.

## 🚀 Quick Start

### Step 1: Generate Test Data

```bash
# Generate 100 test annotation pairs
python src/cli/generate_test_data.py --output-dir ./test_datasets --count 100

# Or with custom parameters
python src/cli/generate_test_data.py --output-dir ./custom_test --count 50 --seed 123
```

**Output Structure:**
```
test_datasets/
├── ground_truth/          # 100 ground truth annotation files
│   ├── uuid1.json
│   ├── uuid2.json
│   └── ...
├── predictions/           # 100 LLM prediction files
│   ├── uuid1.json
│   ├── uuid2.json
│   └── ...
└── dataset_stats.json     # Dataset statistics
```

### Step 2: Run Evaluation

```bash
# Basic evaluation
python src/cli/evaluate_annotations.py \
    --ground-truth ./test_datasets/ground_truth \
    --predictions ./test_datasets/predictions

# With custom IoU threshold
python src/cli/evaluate_annotations.py \
    --ground-truth ./test_datasets/ground_truth \
    --predictions ./test_datasets/predictions \
    --iou-threshold 0.3

# Save detailed results to JSON
python src/cli/evaluate_annotations.py \
    --ground-truth ./test_datasets/ground_truth \
    --predictions ./test_datasets/predictions \
    --output-json ./evaluation_results.json
```

## 📊 Evaluation Output

The evaluation tool provides comprehensive metrics for each UI component tag:

```
================================================================================
📊 UI COMPONENT ANNOTATION EVALUATION REPORT
================================================================================
📁 Files processed: 100/100
🎯 IoU threshold: 0.5

📈 OVERALL METRICS:
   Total Ground Truth Boxes: 456
   Total Predicted Boxes: 423
   Correctly Predicted Boxes: 367
   Overall Precision: 0.868
   Overall Recall: 0.805
   Overall F1-Score: 0.835

🏷️  METRICS BY TAG:
--------------------------------------------------------------------------------
Tag          GT Boxes   Pred Boxes   Correct  Precision  Recall   F1-Score
--------------------------------------------------------------------------------
button       124        118          102      0.864      0.823    0.843
input        89         87           76       0.874      0.854    0.864
radio        134        125          108      0.864      0.806    0.834
dropdown     109        93           81       0.871      0.743    0.802
--------------------------------------------------------------------------------

💡 INSIGHTS:
   🎯 Best performing tag: input (F1: 0.864)
   ⚠️  Worst performing tag: dropdown (F1: 0.802)
```

## 🔧 Tool Details

### Test Data Generator (`generate_test_data.py`)

**Features:**
- Generates realistic UI component bounding boxes
- Supports multiple screen sizes (desktop, mobile, tablet)
- Adds realistic noise to simulate LLM prediction errors
- Creates false positives and false negatives
- Ensures minimal component overlap
- Reproducible results with seed parameter

**Parameters:**
- `--output-dir`: Output directory for test data (default: `./test_datasets`)
- `--count`: Number of annotation pairs to generate (default: 100)
- `--seed`: Random seed for reproducible results (default: 42)

**Generated Data Characteristics:**
- **Components per image**: 2-8 UI elements
- **Component sizes**: Realistic dimensions for each tag type
- **Screen sizes**: Desktop (1920×1080), Mobile (375×812), Tablet (768×1024)
- **Prediction accuracy**: ~80% true positive rate with coordinate noise
- **False positives**: 0-2 per image
- **Tag distribution**: Balanced across button, input, radio, dropdown

### Evaluation Tool (`evaluate_annotations.py`)

**Features:**
- Processes folders of JSON annotation files
- IoU-based matching with configurable threshold
- Per-tag metrics calculation
- Comprehensive reporting with insights
- JSON output for further analysis

**Parameters:**
- `--ground-truth`: Path to ground truth directory (required)
- `--predictions`: Path to predictions directory (required)
- `--iou-threshold`: IoU threshold for matches (default: 0.5)
- `--output-json`: Save detailed results to JSON file
- `--quiet`: Suppress progress messages

**Metrics Calculated:**
- **Precision**: `correct_predictions / total_predictions`
- **Recall**: `correct_predictions / total_ground_truth`
- **F1-Score**: `2 * (precision * recall) / (precision + recall)`
- **True Positives**: Correctly identified components
- **False Positives**: Predicted components with no ground truth match
- **False Negatives**: Ground truth components that were missed

## 📁 File Formats

### Ground Truth Format (`*.json`)
```json
[
  {
    "id": "uuid",
    "image_id": "image-uuid",
    "bounding_box": {
      "x": 100.0,
      "y": 50.0,
      "width": 200.0,
      "height": 30.0
    },
    "tag": "button",
    "confidence": null,
    "annotator": "human_annotator",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
    "status": "active",
    "reviewed_by": null,
    "reviewed_at": null,
    "conflicts_with": [],
    "reasoning": null
  }
]
```

### Predictions Format (`*.json`)
```json
{
  "image_id": "image-uuid",
  "predictions": [
    {
      "id": "pred-uuid",
      "image_id": "image-uuid",
      "bounding_box": {
        "x": 105.2,
        "y": 48.7,
        "width": 195.3,
        "height": 32.1
      },
      "tag": "button",
      "confidence": 0.95,
      "annotator": "gpt-4v",
      "created_at": "2024-01-01T00:00:00",
      "updated_at": "2024-01-01T00:00:00",
      "status": "draft",
      "reviewed_by": null,
      "reviewed_at": null,
      "conflicts_with": [],
      "reasoning": null
    }
  ],
  "llm_model": "gpt-4o",
  "processing_time": 5.2,
  "created_at": "2024-01-01T00:00:00"
}
```

## 🎯 Use Cases

### 1. **Homework Compliance**
These tools fulfill the homework requirement for a command-line evaluation tool that processes folders of annotation files.

### 2. **Model Development**
- Generate test datasets for model training and evaluation
- Compare different LLM models or prompt strategies
- Test different IoU thresholds and evaluation criteria

### 3. **Quality Assurance**
- Validate annotation quality before production use
- Identify systematic biases in predictions
- Monitor model performance over time

### 4. **Research and Benchmarking**
- Create standardized evaluation datasets
- Compare annotation tools and approaches
- Publish reproducible evaluation results

## 🔍 Example Workflow

```bash
# 1. Generate test dataset
python src/cli/generate_test_data.py --count 100 --output-dir ./test_data

# 2. Run evaluation with standard threshold
python src/cli/evaluate_annotations.py \
    --ground-truth ./test_data/ground_truth \
    --predictions ./test_data/predictions \
    --output-json ./results_0.5.json

# 3. Run evaluation with stricter threshold
python src/cli/evaluate_annotations.py \
    --ground-truth ./test_data/ground_truth \
    --predictions ./test_data/predictions \
    --iou-threshold 0.7 \
    --output-json ./results_0.7.json

# 4. Compare results
diff ./results_0.5.json ./results_0.7.json
```

## 🛠️ Development Notes

### Data Engineering Principles
- **Reproducibility**: Fixed seeds ensure consistent test data
- **Scalability**: Tools handle large datasets efficiently
- **Modularity**: Separate generation and evaluation concerns
- **Validation**: Comprehensive error handling and input validation
- **Documentation**: Clear interfaces and comprehensive help text

### Performance Considerations
- **Memory Efficient**: Processes files individually rather than loading all data
- **Progress Tracking**: Visual feedback for long-running operations
- **Error Recovery**: Continues processing even if individual files fail
- **Batch Processing**: Optimized for processing 100+ files

### Future Enhancements
- Support for additional annotation formats (COCO, YOLO)
- Multi-threaded processing for large datasets
- Advanced visualization of evaluation results
- Integration with ML experiment tracking tools

## 📝 Requirements

- Python 3.7+
- Standard library only (no external dependencies)
- Compatible with existing project data formats

---

**Happy Evaluating! 🎯** 