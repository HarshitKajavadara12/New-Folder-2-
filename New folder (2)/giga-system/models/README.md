This directory should contain versioned model artifacts (e.g., .pkl, .onnx).
Phase 2 mandates that strategies must load FROZEN artifacts from here,
rather than calculating logic from source code at runtime.

Structure:
- models/prod/ (Live approved)
- models/research/ (Experimental)
