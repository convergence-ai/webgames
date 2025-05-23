## Data Summary

The primary data sources are evaluation logs, likely from the `inspect_ai` model testing framework, and a supplementary CSV file for game categorization.

**1. Evaluation Run Data:**
This data provides information about each evaluation session executed. Key fields include:

- **Identifiers**: `eval_id` (unique ID for the evaluation), `run_id`, `task_id` (ID for the overall evaluation configuration).
- **Model Information**: `model` (name of the language model used, e.g., `openai/gpt-4o-mini`, `google/gemini-2.5-flash-preview-04-17`).
- **Execution Details**: `created` (timestamp), `status` (e.g., "success"), `log` (path to the raw log file).
- **Environment**: `git_origin`, `git_commit` (for reproducibility), `packages` (versions of key libraries like `inspect_ai`).
- **Overall Performance Metrics**: `total_samples`, `completed_samples`, `score_headline_name` (primary metric, e.g., "webgames_scorer"), `score_headline_metric` (type of metric, e.g., "accuracy"), `score_headline_value` (the actual score for the headline metric).
  The `analyse.py` script filters these evaluation runs based on criteria like `status == 'success'`, `total_samples > 150`, and `score_headline_value > 0.02`.

**2. Individual Sample Data:**
This data contains details for each individual task or "sample" processed within an evaluation run. Key fields include:

- **Identifiers**: `sample_id` (unique ID for the sample), `id` (specific task identifier, e.g., "buttons-easy", "datehard"), `eval_id` (to link back to the evaluation run).
- **Task Content**: `input` (the prompt or instructions given to the model), `target` (the expected correct outcome or password for the task).
- **Task Metadata**:
  - `metadata_base_task`: The fundamental name of the game or task (e.g., "buttons", "date"). This is often referred to as `game` after processing.
  - `metadata_variant`: The specific version or difficulty of the task (e.g., "easy", "base", "hard"). This is often referred to as `difficulty` after processing.
  - `metadata_description`: A textual description of the task.
  - `metadata_title`: A user-friendly title for the task (e.g., "Button megastar (Easy)").
  - `metadata_path`: An identifier or path for the task.
  - `metadata_difficulty` (original field, distinct from `metadata_variant` sometimes): Stated difficulty.
- **Scoring & Outcome**:
  - `score_webgames_scorer`: The result of the webgames-specific scorer, typically 'C' for Correct/Completed or 'I' for Incorrect. This is the primary field used to determine if a sample `passed`.
- **Performance**: `total_time` (total time taken for the sample), `working_time`.
- **Operational Data**: `epoch`, `retries`, `error` (if any).

**3. Game Category Data:**
This data comes from an external CSV file (`webgames_categories.csv`) and is used to group games into broader categories for analysis.

- `game`: The name of the game (matches `metadata_base_task` from sample data).
- `category`: The assigned category for the game (e.g., "Navigation", "Comprehension", "Data Entry").
- `weight`: A numerical weight assigned to the game within its category, used for weighted success rate calculations.

**Processing & Merging:**

- The `analyse.py` script and the `Untitled.ipynb` notebook both load the raw evaluation and sample data using `inspect_ai` library functions (`evals_df`, `samples_df`).
- They filter the evaluation runs to select relevant ones (e.g., based on status, number of samples, and headline score).
- The sample data is merged with the filtered evaluation data to associate each sample with a specific model name.
- Model names are standardized (e.g., `openai/gpt-4o-mini` is processed to `gpt-4o-mini`).
- Relevant metadata fields like `difficulty` (from `metadata_variant` or `metadata.variant`) and `game` (from `metadata_base_task` or `metadata.base_task`) are extracted and standardized.
- A boolean `passed` column is derived for each sample, primarily based on the `score_webgames_scorer == 'C'` condition, with fallbacks to other score fields if necessary.
- The game category data from `webgames_categories.csv` is loaded, transformed (melted), and merged with the sample data based on the game name.
- The `analyse.py` script caches this processed, comprehensive dataframe (in `outputs/processed_data.pkl`) to avoid redundant processing in subsequent runs.

This combined and processed dataset allows for various analyses, such as comparing model performance (e.g., success rates) across different games, varying difficulties, and game categories, as demonstrated by the plotting functions within `analyse.py`.
