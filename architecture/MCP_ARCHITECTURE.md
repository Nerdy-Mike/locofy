# MCP (Model Context Protocol) Architecture

## Document Purpose

This document defines the **Model Context Protocol integration strategy** for the UI Component Labeling System. Following the **single responsibility principle**, this document focuses exclusively on MCP architecture, implementation decisions, and trade-offs.

**Related Documents:**
- [Main Architecture](./ARCHITECTURE.md) - Overall system design
- [Implementation Plan](./IMPLEMENTATION_PLAN.md) - Development roadmap  
- [Data Flow](./DATAFLOW.md) - System data flows
- [Data Types](./DATA_TYPES.md) - Data model definitions

---

## Table of Contents

1. [MVP Decision & Trade-offs](#mvp-decision--trade-offs)
2. [MCP Component Architecture](#mcp-component-architecture)
3. [MVP Implementation Strategy](#mvp-implementation-strategy)
4. [Data Flow Integration](#data-flow-integration)
5. [Future Enhancement Path](#future-enhancement-path)
6. [Technical Specifications](#technical-specifications)

---

## MVP Decision & Trade-offs

### Decision: Simplified MCP for MVP

**Chosen Approach:** Lightweight MCP integration with basic context awareness

**Business Rationale:**
- Deliver immediate prediction quality improvements
- Minimize development risk and complexity
- Establish foundation for future advanced features
- Maintain development velocity for core MVP features

### Trade-off Analysis

| Aspect                     | Direct OpenAI API    | MVP MCP          | Advanced MCP        | Decision  |
| -------------------------- | -------------------- | ---------------- | ------------------- | --------- |
| **Development Time**       | ✅ 0 hours (existing) | ⚠️ 3-4 hours      | ❌ 20-30 hours       | ✅ MVP MCP |
| **Prediction Quality**     | ⭐ Basic              | ⭐⭐ Context-aware | ⭐⭐⭐ Learning system | ✅ MVP MCP |
| **Maintenance Complexity** | ⭐ Simple             | ⭐⭐ Moderate      | ⭐⭐⭐ Complex         | ✅ MVP MCP |
| **Risk Level**             | ✅ Known              | ⚠️ Low            | ❌ High              | ✅ MVP MCP |
| **Future Extensibility**   | ❌ Limited            | ✅ Good           | ✅ Excellent         | ✅ MVP MCP |

### What We Gain with MVP MCP

| Improvement           | Impact                                    | Effort | ROI  |
| --------------------- | ----------------------------------------- | ------ | ---- |
| **Context Awareness** | Predictions consider existing annotations | Low    | High |
| **Structured Output** | More reliable than prompt engineering     | Low    | High |
| **Future Foundation** | Easy path to advanced features            | Low    | High |
| **Better Accuracy**   | Improved UI element detection             | Low    | High |

### What We Defer to Phase 2

| Advanced Feature                 | Business Impact        | Technical Complexity | MVP Decision |
| -------------------------------- | ---------------------- | -------------------- | ------------ |
| **Session Learning**             | High long-term value   | High complexity      | ⏭️ Defer      |
| **User Feedback Loop**           | Medium long-term value | Medium complexity    | ⏭️ Defer      |
| **Batch Context Sharing**        | Low immediate value    | Medium complexity    | ⏭️ Defer      |
| **Advanced Quality Integration** | Medium immediate value | High complexity      | ⏭️ Defer      |

---

## MCP Component Architecture

### MVP Architecture Overview

```mermaid
graph TD
    A[FastAPI Backend] --> B[SimpleMCPClient]
    B --> C[BasicContextBuilder]
    B --> D[UIDetectionTool]
    B --> E[MCPServerConnection]
    
    E --> F[@anthropic-ai/mcp-server-openai]
    F --> G[GPT-4 Vision API]
    
    C --> H[Load Existing Annotations]
    C --> I[Load Image Metadata]
    
    D --> J[Structured UI Detection]
    J --> K[Parse Results]
    K --> L[Existing Quality System]
    L --> M[Save Predictions]
    
    style B fill:#e1f5fe
    style D fill:#f3e5f5
    style F fill:#e8f5e8
```

### Component Responsibilities

| Component               | Single Responsibility                     | MVP Scope                            |
| ----------------------- | ----------------------------------------- | ------------------------------------ |
| **SimpleMCPClient**     | Orchestrate MCP communication             | Basic request/response handling      |
| **BasicContextBuilder** | Build context from existing data          | Include annotations + image metadata |
| **UIDetectionTool**     | Define structured UI detection capability | Single tool for element detection    |
| **MCPServerConnection** | Manage external MCP server process        | Simple stdio communication           |
| **ResultParser**        | Convert MCP output to data models         | Parse structured JSON responses      |

### MVP Component Constraints

**What Each Component Does NOT Do (for MVP):**

| Component               | MVP Limitations              | Future Enhancement             |
| ----------------------- | ---------------------------- | ------------------------------ |
| **SimpleMCPClient**     | ❌ No session persistence     | ✅ Add session management       |
| **BasicContextBuilder** | ❌ No learning patterns       | ✅ Add user correction patterns |
| **UIDetectionTool**     | ❌ No dynamic tool adaptation | ✅ Add tool learning            |
| **MCPServerConnection** | ❌ No advanced error recovery | ✅ Add circuit breakers         |
| **ResultParser**        | ❌ No confidence calibration  | ✅ Add adaptive confidence      |

---

## MVP Implementation Strategy

### Phase 1: Core MCP Integration (MVP)

#### 1.1 Server Setup
```bash
# External dependency - managed separately
npm install -g @anthropic-ai/mcp-server-openai
```

#### 1.2 Client Integration Points

| Integration Point       | Existing Component          | MCP Enhancement                  |
| ----------------------- | --------------------------- | -------------------------------- |
| **Prediction Endpoint** | `/api/predict/{image_id}`   | Add MCP-powered alternative      |
| **Context Building**    | Image + metadata loading    | Include existing annotations     |
| **Result Processing**   | Direct API response parsing | Structured tool response parsing |
| **Quality Integration** | Existing conflict detection | Enhanced with MCP metadata       |

#### 1.3 MVP Data Flow

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant API as 🔧 FastAPI
    participant MCP as 🤖 SimpleMCPClient
    participant Server as 📡 MCP Server
    participant GPT as 🧠 GPT-4V
    participant Storage as 💾 Storage
    
    User->>API: POST /api/predict-mcp/{image_id}
    API->>Storage: Load image + existing annotations
    Storage-->>API: Image data + annotation context
    
    API->>MCP: build_context(image, annotations)
    MCP->>MCP: Create basic context structure
    
    MCP->>Server: Execute UI detection tool
    Server->>GPT: Analyze image with context
    GPT-->>Server: Structured UI element predictions
    Server-->>MCP: Tool execution results
    
    MCP->>API: Parsed prediction results
    API->>Storage: Save MCP predictions
    API-->>User: Enhanced prediction response
```

### Phase 1 Success Criteria

| Criterion                 | Measurement                                  | Target            |
| ------------------------- | -------------------------------------------- | ----------------- |
| **Integration Time**      | Development hours                            | < 4 hours         |
| **Prediction Quality**    | Accuracy vs direct API                       | > 10% improvement |
| **System Stability**      | Error rate                                   | < 5%              |
| **Response Time**         | API latency                                  | < 10 seconds      |
| **Context Effectiveness** | Predictions considering existing annotations | > 80%             |

### Phase 1 Risk Mitigation

| Risk                        | Probability | Impact | Mitigation                      |
| --------------------------- | ----------- | ------ | ------------------------------- |
| **MCP Server Failure**      | Medium      | High   | Fallback to direct API          |
| **Integration Complexity**  | Low         | Medium | Keep implementation simple      |
| **Performance Degradation** | Low         | Medium | Timeout and caching strategies  |
| **OpenAI API Issues**       | Medium      | High   | Existing error handling applies |

---

## Data Flow Integration

### MVP Context Building

```json
{
  "image_id": "uuid",
  "task": "ui_element_detection",
  "image_metadata": {
    "dimensions": {"width": 1920, "height": 1080},
    "format": "PNG"
  },
  "existing_annotations": [
    {
      "tag": "button",
      "bounding_box": {"x": 50, "y": 50, "width": 100, "height": 30},
      "created_by": "manual"
    }
  ],
  "detection_instructions": "Detect UI elements avoiding overlap with existing annotations"
}
```

### MVP Tool Definition

```json
{
  "name": "detect_ui_elements",
  "description": "Detect UI elements with awareness of existing annotations",
  "parameters": {
    "type": "object",
    "properties": {
      "elements": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "tag": {"type": "string", "enum": ["button", "input", "radio", "dropdown"]},
            "bounding_box": {
              "type": "object",
              "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "width": {"type": "number"},
                "height": {"type": "number"}
              },
              "required": ["x", "y", "width", "height"]
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string"}
          },
          "required": ["tag", "bounding_box", "confidence"]
        }
      }
    },
    "required": ["elements"]
  }
}
```

### MVP Result Format

```json
{
  "prediction_id": "pred_uuid_timestamp",
  "image_id": "uuid",
  "model_version": "gpt-4-vision-mcp-mvp",
  "processing_time": 2.3,
  "total_elements": 3,
  "status": "completed",
  "mcp_enabled": true,
  "context_used": true,
  
  "elements": [
    {
      "id": "elem_0",
      "tag": "input",
      "bounding_box": {"x": 100, "y": 80, "width": 200, "height": 25},
      "confidence": 0.85,
      "reasoning": "Text input field with visible border, distinct from existing button annotation"
    }
  ]
}
```

---

## Future Enhancement Path

### Phase 2: Session Management & Learning

#### Enhanced Components (Future)

| Component               | Phase 1 (MVP)              | Phase 2 Enhancement                 |
| ----------------------- | -------------------------- | ----------------------------------- |
| **MCP Client**          | Stateless requests         | Session-aware with persistence      |
| **Context Builder**     | Basic existing annotations | User correction patterns + learning |
| **Quality Integration** | Basic conflict detection   | Predictive quality assessment       |
| **Tool Registry**       | Single UI detection tool   | Multiple specialized tools          |
| **Feedback System**     | Manual quality review      | Automated learning from corrections |

#### Future Architecture Evolution

```mermaid
graph TD
    A[MVP MCP Client] --> B[Enhanced MCP Client]
    B --> C[Session Manager]
    B --> D[Learning Engine]
    B --> E[Advanced Context Builder]
    
    C --> F[Persistent Context Storage]
    D --> G[User Pattern Recognition]
    E --> H[Multi-source Context Synthesis]
    
    I[Single UI Tool] --> J[Tool Registry]
    J --> K[Specialized Detection Tools]
    J --> L[Validation Tools]
    J --> M[Quality Assessment Tools]
```

### Migration Strategy (MVP → Advanced)

| Migration Step                       | Effort | Risk   | Business Value |
| ------------------------------------ | ------ | ------ | -------------- |
| **Add Session Storage**              | Medium | Low    | High           |
| **Implement Learning Patterns**      | High   | Medium | High           |
| **Add Advanced Quality Integration** | Medium | Low    | Medium         |
| **Create Specialized Tools**         | High   | Medium | Medium         |
| **Add Real-time Feedback**           | High   | High   | High           |

---

## Technical Specifications

### MVP Environment Requirements

```bash
# External Dependencies
Node.js >= 16.0.0 (for MCP server)
@anthropic-ai/mcp-server-openai >= 1.0.0

# Python Dependencies  
mcp >= 0.1.0
asyncio (built-in)
```

### MVP Configuration

```python
# Environment Variables
OPENAI_API_KEY=your_api_key
MCP_ENABLED=true
MCP_TIMEOUT=30
MCP_FALLBACK_TO_DIRECT_API=true

# MCP Client Configuration
MCP_SERVER_COMMAND="npx"
MCP_SERVER_ARGS=["-y", "@anthropic-ai/mcp-server-openai"]
MCP_TRANSPORT="stdio"
```

### MVP API Changes

```python
# New Endpoint (alongside existing)
@router.post("/predict-mcp/{image_id}")
async def predict_with_mcp(image_id: str) -> PredictionResponse:
    """MCP-powered prediction with context awareness"""
    pass

# Existing endpoint remains unchanged
@router.post("/predict/{image_id}") 
async def predict_direct(image_id: str) -> PredictionResponse:
    """Direct OpenAI API prediction (fallback)"""
    pass
```

### MVP Error Handling

| Error Type                   | MCP Behavior              | Fallback Strategy         |
| ---------------------------- | ------------------------- | ------------------------- |
| **MCP Server Unavailable**   | Detect connection failure | Use direct OpenAI API     |
| **Tool Execution Failure**   | Parse error response      | Retry with simpler prompt |
| **Context Building Error**   | Log and continue          | Predict without context   |
| **Response Parsing Failure** | Structured error handling | Return error response     |

### MVP Monitoring

| Metric                 | Purpose                         | Target        |
| ---------------------- | ------------------------------- | ------------- |
| **MCP Success Rate**   | Monitor MCP reliability         | > 95%         |
| **Context Usage Rate** | Track context effectiveness     | > 80%         |
| **Fallback Rate**      | Monitor MCP vs direct API usage | < 10%         |
| **Response Time**      | Performance monitoring          | < 10 seconds  |
| **Prediction Quality** | Accuracy improvement tracking   | Monitor trend |

---

## Implementation Checklist

### MVP Phase 1 Tasks

- [ ] Install MCP server dependency
- [ ] Create SimpleMCPClient service
- [ ] Implement BasicContextBuilder
- [ ] Define single UIDetectionTool
- [ ] Add MCP prediction endpoint
- [ ] Implement fallback to direct API
- [ ] Add basic error handling
- [ ] Update frontend to use MCP endpoint
- [ ] Add monitoring and logging
- [ ] Test with existing images
- [ ] Document MVP usage

### Success Validation

- [ ] MCP predictions include existing annotation context
- [ ] Prediction quality improves vs direct API
- [ ] System remains stable with MCP integration
- [ ] Fallback works when MCP fails
- [ ] Response times acceptable for UX
- [ ] Integration completed in < 4 hours

---

This MCP architecture document provides a focused, single-responsibility view of the Model Context Protocol integration while maintaining clear boundaries with other system components and a practical path from MVP to advanced features. 