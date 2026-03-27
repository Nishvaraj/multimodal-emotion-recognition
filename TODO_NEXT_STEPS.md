# TODO - Next Steps (Post-Implementation Alignment Sprint)

Date: March 27, 2026
Focus: Documentation accuracy, evidence quality, model improvement, and submission readiness.

## Week 1 (Right Now): Documentation and Evidence First

### 1. Documentation Sync (High examiner impact, same-day)
- Update project status document to match implemented code
- Update README to reflect full capabilities:
  - Explainability (Grad-CAM and audio saliency)
  - Authentication
  - Supabase history storage and export
  - Combined analysis with separate and video modes
- Remove stale statements that mark completed features as pending

### 2. Dissertation Sync (High examiner impact, 1-2 days)
- Update implementation chapter to include:
  - FastAPI endpoint coverage
  - React dashboard and protected route flow
  - Supabase auth + per-user history persistence
  - Explainability pipeline (facial and speech)
- Update limitations chapter to clearly state:
  - Facial model accuracy gap
  - Explainability sensitivity to input quality
- Update architecture diagrams to include:
  - Supabase
  - Explainability outputs
  - Combined/video analysis data paths

### 3. Evidence Pack for Examiners (High impact, 1 day)
Capture and archive evidence for:
- Login/signup flow
- Facial prediction with and without Grad-CAM
- Speech prediction with and without saliency
- Combined analysis (separate mode)
- Video analysis mode
- History operations (note, pin/unpin, delete)
- Export outputs (CSV and summary report)
- Sample API responses for facial, speech, combined, and video routes

## Week 2: Validation and Quality

### 1. End-to-End Validation Checklist
- Validate all tabs and input modes
- Validate explainability toggles and fallback behavior
- Validate Supabase history CRUD and exports
- Record pass/fail evidence with timestamps

### 2. Facial Model Improvement Sprint
- Prioritize experiments aimed at improving facial model accuracy
- Track experiment settings, metrics, and reproducibility details
- Update dissertation results tables with final verified metrics

## Week 3: Final Hardening

### 1. Production Readiness Checks
- Strengthen error handling and log visibility
- Review environment variable documentation
- Re-run clean setup from scratch using README instructions

### 2. Submission Packaging
- Freeze final screenshots and response samples
- Ensure all figures/tables reference the current implementation
- Final language cleanup for examiner-facing clarity

## Definition of Done for Submission

- Documentation matches implementation exactly
- Dissertation implementation chapter matches running system
- Evidence pack covers all major features and modes
- Known limitations are clearly and honestly stated
- Reproducible setup steps are complete and tested
