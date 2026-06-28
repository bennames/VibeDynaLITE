# Gemini Agent Customization Rules

This document outlines the workflow and verification constraints for agent interactions within the **VibeDynaLITE** repository. All agents operating in this workspace must adhere to these rules.

## Workspace Rules

1. **Commit and Push After Major Implementation**
   - After completing any major implementation, whether it is defined as a sprint or guided by a formal `implementation_plan.md`, you must make a code commit containing only the relevant changes and push it to the GitHub repository.

2. **Wait for CI/CD Pipeline Verification**
   - In addition to running the local test suite, you must always wait for and verify that the committed and pushed changes successfully pass all standard CI/CD pipeline tests before considering the task complete.

3. **Mandatory Test and Benchmark Execution**
   - You must run the entire unit test suite and execute all physical benchmarks as part of the standard verification process after modifying the code in any way. Never skip verification.

4. **Wait for Explicit Approval Before Executing Plans**
   - When in planning mode, you must NOT proceed to execute the implementation plan (i.e., do not begin making code changes, writing scripts, or running tests) when asked to review comments, address feedback, or update the plan document. You must update/revise the implementation plan as requested, request feedback again, and wait for the user's explicit approval to proceed to the execution phase.
