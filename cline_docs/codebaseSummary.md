## Key Components and Their Interactions

- **Models:** Contains PyTorch and TensorFlow implementations of various NILM models (e.g., Seq2Point, Seq2Seq, UNetNILM).
- **Data Loaders:** Provides utilities for loading and preprocessing data for both PyTorch and TensorFlow models.
- **Trainers:** Includes `torch_trainer.py` and `keras_trainer.py` for training the respective models.
- **Utils:** Contains helper functions for various tasks like logging, experiment setup, and compatibility checks.
- **Disaggregator:** Houses the `nilm_experiment.py` script for running disaggregation experiments.

## Data Flow

1. Data is loaded using the appropriate data loader (PyTorch or TensorFlow).
2. Data is preprocessed (normalized, thresholded, etc.).
3. The model is trained using the corresponding trainer.
4. Post-processing is applied to the model's output (denormalization, aggregation).
5. Results are evaluated and logged.

## External Dependencies

- The project depends on several external libraries, managed through `requirements.txt`.
- Key dependencies include PyTorch, TensorFlow, NumPy, and Pandas.
- NILMTK is used for energy disaggregation tasks.

## Recent Significant Changes

- The `requirements.txt` file has been updated, potentially affecting dependencies.

## User Feedback Integration and Its Impact on Development

- Currently, no user feedback has been integrated.
- Future development may incorporate user feedback to improve the project's usability and features.

## Additional Documents in cline_docs

- `projectRoadmap.md`: Outlines project goals, features, and progress.
- `currentTask.md`: Describes the current objectives and next steps.
- `techStack.md`: Lists the key technologies used in the project.