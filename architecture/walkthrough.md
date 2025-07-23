# UI Component Labeling System: Requirement Walkthrough

## I. Objective

> **Create a tool that helps users label UI components on design screenshots, and evaluate the performance of an LLM model that automatically generates these tags.**

---

## II. Task Breakdown

### 1. Ground Truth Labeling Tool

**Requirement:**  
- Web-based UI for:
  - Uploading design images/screenshots
  - Drawing rectangular bounding boxes
  - Assigning a tag to each box (button, input, radio, dropdown)
  - Saving labeled results as JSON

**How We Achieve This:**

| Sub-requirement | Implementation                                       | Key Files/Services                                                                                                            |
| --------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Upload image    | Streamlit frontend UI, FastAPI backend, file storage | `frontend/streamlit_app.py`, `frontend/components/image_viewer.py`, `services/file_storage.py`, `adapters/fastapi_adapter.py` |
| Draw boxes      | Interactive canvas in Streamlit                      | `frontend/components/annotation_canvas.py`, `frontend/pages/enhanced_annotation_viewer.py`                                    |
| Assign tags     | Tag selection in annotation UI                       | `frontend/components/annotation_canvas.py`, `annotation_renderer.py`                                                          |
| Save as JSON    | Backend API, file storage                            | `services/annotation_validation_service.py`, `models/annotation_models.py`, `data/annotations/`                               |

---

### 2. "Predict" Button – Call LLM for Auto-Tagging

**Requirement:**  
- UI button to trigger LLM prediction
- LLM detects/tags UI elements (Button, Input, Radio, Dropdown)
- Save LLM predictions as JSON (same format as ground truth)

**How We Achieve This:**

| Sub-requirement  | Implementation                                | Key Files/Services                                                                  |
| ---------------- | --------------------------------------------- | ----------------------------------------------------------------------------------- |
| "Predict" button | Streamlit UI triggers backend                 | `frontend/pages/enhanced_annotation_viewer.py`, `frontend/utils/api_client.py`      |
| LLM prediction   | FastAPI endpoint calls LLM service            | `services/llm_service.py`, `services/mcp_service.py`, `adapters/fastapi_adapter.py` |
| Tag restriction  | LLM prompt and validation restricts to 4 tags | `services/llm_service.py`, `models/annotation_models.py`                            |
| Save predictions | Backend saves to predictions folder           | `services/llm_service.py`, `data/predictions/`                                      |

---

### 3. Evaluation Tool for Tagging Accuracy

**Requirement:**  
- Command-line tool/script
- Inputs: folder of ground truth JSON, folder of prediction JSON
- Calculates: total ground truth boxes, correct predictions, precision, recall, F1-score (per tag)
- Outputs a report

**How We Achieve This:**

| Sub-requirement | Implementation                              | Key Files/Services                                                  |
| --------------- | ------------------------------------------- | ------------------------------------------------------------------- |
| CLI tool        | Python script for evaluation                | `src/cli/evaluate_annotations.py`                                   |
| Input folders   | Accepts ground truth and prediction folders | `test_data/annotations/`, `test_data/predictions/`                  |
| Metrics         | Calculates per-tag and overall metrics      | `src/cli/evaluate_annotations.py`                                   |
| Output report   | Prints and saves results as JSON            | `evaluation_report.json`, `evaluation_results.json`, `results.json` |

---

### 4. Documentation

**Requirement:**  
- Clean, well-documented code
- README with instructions

**How We Achieve This:**

| Sub-requirement | Implementation                             | Key Files/Services                                                                                                                                                  |
| --------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Code comments   | All major modules and functions documented | All `.py` files, especially in `services/`, `models/`, `frontend/`                                                                                                  |
| README          | Usage, setup, and architecture explained   | `README.md`                                                                                                                                                         |
| Architecture    | Detailed design and data flow docs         | `architecture/ARCHITECTURE.md`, `architecture/DATAFLOW.md`, `architecture/IMPLEMENTATION_PLAN.md`, `architecture/MCP_ARCHITECTURE.md`, `architecture/DATA_TYPES.md` |

---

### 5. Automation Design (Open Question)

**Requirement:**  
- If you had to process hundreds of images, how would you automate the prediction step?

**How We Achieve This:**

| Sub-requirement  | Implementation                                           | Key Files/Services                                                                                                                   |
| ---------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Batch processing | Async batch endpoints, scalable worker design            | `services/llm_service.py`, `services/mcp_service.py`, `architecture/ARCHITECTURE.md` (see "Batch Processing", "ServiceArchitecture") |
| Folder structure | Organized batch and prediction folders                   | `data/predictions/`, `data/batches/`                                                                                                 |
| CLI/script       | Script can loop over images and call prediction endpoint | `src/cli/evaluate_annotations.py`, (see README for automation note)                                                                  |

---

## III. Deliverables Mapping

| Deliverable                                | Where to Find It                                                                               |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Web-based UI for annotation and prediction | `frontend/`, `services/`, `adapters/`                                                          |
| Command-line evaluation tool               | `src/cli/evaluate_annotations.py`                                                              |
| Clean code and documentation               | All modules, `README.md`, `architecture/`                                                      |
| Automation design discussion               | `README.md`, `architecture/ARCHITECTURE.md` (see "Batch Processing" and "ServiceArchitecture") |

---

## IV. Data Model Reference

- **Annotation JSON**: See `models/annotation_models.py`, `architecture/DATA_TYPES.md`
- **Prediction JSON**: Same structure as annotation JSON, stored in `data/predictions/`
- **Evaluation Output**: See `results.json`, `evaluation_report.json`

---

## V. How to Run

- **Web UI**:  
  - `docker-compose up` or `./start.sh`
  - Access at [http://localhost:8501](http://localhost:8501)
- **Evaluation Tool**:  
  - `python3 src/cli/evaluate_annotations.py --ground-truth <folder> --predictions <folder> --output-json <file>`
- **Batch Prediction**:  
  - Use API endpoints or script to process multiple images (see README and architecture docs)

---

## VI. Further Reading

- **System Architecture**: `architecture/ARCHITECTURE.md`
- **Implementation Plan**: `architecture/IMPLEMENTATION_PLAN.md`
- **Data Flow**: `architecture/DATAFLOW.md`
- **MCP Integration**: `architecture/MCP_ARCHITECTURE.md`
- **Data Types**: `architecture/DATA_TYPES.md`

---

**This document provides a clear mapping from requirements to implementation, making it easy for reviewers to verify each deliverable.** 