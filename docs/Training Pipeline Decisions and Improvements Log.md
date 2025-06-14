Log: Key Decisions and Fixes in the Training Pipeline
This document outlines the major issues encountered, the decisions made, and the final fixes implemented to create a stable and effective training framework for the Graph-to-Text task.

1. Initial Challenge: Environment & Dependency Hell
Problem: The initial setup was plagued by a series of low-level environment errors, including bitsandbytes CUDA errors, use_2to3 is invalid from setuptools, and torch not found during the Unsloth installation.

Decision: Instead of patching individual libraries, the most robust solution was to adopt a specific, sequential installation process.

Final Fix:

Install a compatible version of torch first.

Pre-install a modern version of the decorator package to satisfy legacy dependencies.

Finally, install unsloth, allowing its installer to correctly resolve the remaining dependencies against the pre-existing torch installation.

2. Core Issue: Unstable Configuration System
Problem: The system was plagued by a series of AttributeError and KeyError exceptions (e.g., Key 'model' is not in struct, Key 'base_config' not in struct, additional config directory ... not found). This indicated a fundamental flaw in the Hydra configuration. The initial setup used an ambiguous composition strategy that was not robust to overrides.

Decision: The entire configuration system was refactored away from implicit defaults to an explicit, self-contained structure. This is the standard best practice for complex Hydra projects.

Final Fix:

Self-Contained Experiment Files: Each experiment/*.yaml file was made completely self-contained, defining all necessary blocks (model, training, peft, etc.).

No More Base Config: The base_config.yaml was removed to eliminate any ambiguity about where a parameter was coming from.

Explicit Script Execution: The run_experiment.sh script was rewritten to be robust to the execution path and to use the explicit --config-dir and --config-name flags, ensuring Hydra loads the correct file every time.

3. Critical Bug: Model Hallucination & Zero-Loss Training
Problem: After fixing the configuration, the training process started, but the model's loss plummeted to near-zero almost instantly. This led to a poorly trained model that would either repeat the input prompt or generate garbage (like a sequence of semicolons) during inference.

Decision: This is a classic symptom of the model being shown the "answer" as part of its input without a proper loss mask. The most robust way to fix this is to stop manually formatting prompts and instead use the tokenizer's official apply_chat_template method.

Final Fix:

Data Loader Refactor: The data_loader.py script was completely rewritten. It now "flattens" the dataset (creating one example per text description) and applies the chat template to each example, structuring the data with clear user and assistant roles.

SFTTrainer Update: The train.py script was updated to use this pre-formatted dataset by passing it to the dataset_text_field argument. The previous, more complex formatting_func was removed as it was no longer necessary.

Tokenizer Padding Token: A critical fix was added to model.py to explicitly set tokenizer.pad_token = tokenizer.eos_token, as Llama 3 models do not have a default padding token. This is essential for correct training.

4. Final Polish: Stabilizing the Learning Process
Problem: Even with the data format fixed, the loss was observed to decrease too quickly, a sign of rapid overfitting on the initial batches.

Decision: The learning process needed to be slowed down and stabilized.

Final Fix (in train.py):

Lowered Learning Rate: The learning_rate was reduced from 1e-4 to 2e-5 to encourage more gradual and robust learning instead of memorization.

Increased Warmup: The warmup_steps were increased from 5 to 50, allowing the model's optimizer to stabilize before the main learning phase.

Enabled Packing: packing=True was re-enabled in the SFTTrainer to make training on short sequences more efficient, which can also contribute to stability.

5. Added Feature: Debugging and Observability
Problem: It was difficult to see how the model was learning over time without seeing concrete examples.

Decision: Add a custom callback to the trainer to generate sample outputs.

Final Fix (in train.py): A SampleGenerationCallback class was implemented. At the end of each epoch, it takes a few examples from the validation set, generates a description using the current state of the model, and saves the input, output, and reference text to a .txt file in the experiment's output directory. This provides clear insight into the model's progress.
