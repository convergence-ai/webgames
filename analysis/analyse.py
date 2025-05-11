import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from inspect_ai.analysis.beta import (
    EvalInfo,
    EvalModel,
    EvalResults,
    SampleSummary,
    evals_df,
    samples_df,
)

# Define the desired model order
MODEL_ORDER = [
    "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-20250219-computeruse",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-preview-05-06-textonly",
    "gemini-2.5-flash-preview-04-17",
    "gpt-4o",
    "gpt-4o-mini",
    "qwen2.5-vl-72b-instruct",
    "qwen2.5-vl-32b-instruct",
    "qwen2.5-vl-7b-instruct",
]

# Define the model color palette for consistent plotting
# Keys must match model names in DataFrame (lowercase, as processed)

CLAUDE_COLOR = sns.color_palette("YlOrBr", 3)
GEMINI_COLOR = sns.color_palette("Greens", 4)
GPT_COLOR = sns.color_palette("Blues", 3)
QWEN_COLOR = sns.color_palette("Purples", 4)


MODEL_PALETTE = {
    "claude-3-7-sonnet-20250219": CLAUDE_COLOR[1],
    "claude-3-7-sonnet-20250219-computeruse": CLAUDE_COLOR[2],
    "gemini-2.5-pro-preview-05-06": GEMINI_COLOR[3],
    "gemini-2.5-pro-preview-05-06-textonly": GEMINI_COLOR[2],
    "gemini-2.5-flash-preview-04-17": GEMINI_COLOR[1],
    "gpt-4o": GPT_COLOR[2],
    "gpt-4o-mini": GPT_COLOR[1],
    "qwen2.5-vl-72b-instruct": QWEN_COLOR[3],
    "qwen2.5-vl-32b-instruct": QWEN_COLOR[2],
    "qwen2.5-vl-7b-instruct": QWEN_COLOR[1],
}

# Path to the categories CSV
CATEGORIES_CSV_PATH = "webgames_categories.csv"
CACHE_FILE_PATH = "outputs/processed_data.pkl"


def load_and_preprocess_data():
    """Loads data from eval logs and merges with category information.
    Uses a cache to avoid reprocessing if possible.
    """
    # Check if cache file exists and load it
    if os.path.exists(CACHE_FILE_PATH) and False:
        try:
            with open(CACHE_FILE_PATH, "rb") as f:
                print(f"Loading cached data from {CACHE_FILE_PATH}")
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache file: {e}. Re-processing data.")

    print("No cache found or cache invalid. Processing data from logs...")
    # Load all evals from the "logs" directory
    raw_evals_df = evals_df(
        logs="logs", columns=EvalInfo + EvalModel + EvalResults, strict=True
    )

    if raw_evals_df.empty:
        print("No evaluation data loaded from 'logs' directory. Exiting.")
        return pd.DataFrame()

    # Ensure numeric types for filtering columns and handle missing columns
    if "score_headline_value" in raw_evals_df.columns:
        raw_evals_df["score_headline_value"] = pd.to_numeric(
            raw_evals_df["score_headline_value"], errors="coerce"
        )
    else:
        print(
            "Warning: 'score_headline_value' column not found in evals_df. Filtering on this criterion will be skipped."
        )
        raw_evals_df["score_headline_value"] = (
            pd.NA
        )  # Use NA to avoid issues with > 0.02 comparison if all are NA

    if "total_samples" in raw_evals_df.columns:
        raw_evals_df["total_samples"] = pd.to_numeric(
            raw_evals_df["total_samples"], errors="coerce"
        )
    else:
        print(
            "Warning: 'total_samples' column not found in evals_df. Filtering on this criterion will be skipped."
        )
        raw_evals_df["total_samples"] = pd.NA

    # Filter evals based on notebook criteria
    # Apply filters sequentially to handle potential NAs from missing columns
    filtered_evals_df = raw_evals_df[raw_evals_df["status"] == "success"]
    if "total_samples" in filtered_evals_df.columns:
        filtered_evals_df = filtered_evals_df[filtered_evals_df["total_samples"] > 100]
    if "score_headline_value" in filtered_evals_df.columns:
        filtered_evals_df = filtered_evals_df[
            filtered_evals_df["score_headline_value"] > 0.02
        ]

    if filtered_evals_df.empty:
        print("No evals passed the filtering criteria. Exiting.")
        return pd.DataFrame()

    # Load all samples from the "logs" directory
    all_samples_df = samples_df(logs="logs", columns=SampleSummary, strict=True)

    if all_samples_df.empty:
        print("No sample data loaded from 'logs' directory. Exiting.")
        return pd.DataFrame()

    # Select relevant columns from filtered_evals_df for merging
    evals_to_merge = filtered_evals_df[
        ["eval_id", "model"]
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Merge samples with filtered evals to get model information
    df_combined = pd.merge(
        all_samples_df,
        evals_to_merge,
        on="eval_id",
        how="inner",  # Keeps only samples that have a corresponding filtered eval
    )

    if df_combined.empty:
        print("No samples found for the filtered evaluations. Exiting.")
        return pd.DataFrame()

    # Rename 'id' (sample_id) to 'game' for individual game identification
    if "id" in df_combined.columns:
        df_combined.rename(columns={"id": "game"}, inplace=True)
    else:
        print(
            "Warning: 'id' column not found in samples_df output. Cannot rename to 'game'."
        )
        return pd.DataFrame()

    # Standardize model names (e.g., 'openai/gpt-4o-mini' -> 'gpt-4o-mini')
    if "model" in df_combined.columns:
        df_combined["model"] = df_combined["model"].str.split("/").str[-1]
        df_combined["model"] = df_combined["model"].str.lower()  # Convert to lowercase
        # Convert 'model' column to categorical with specified order
        df_combined["model"] = pd.Categorical(
            df_combined["model"], categories=MODEL_ORDER, ordered=True
        )
        # Drop rows where model is not in MODEL_ORDER (became NaT due to categorical conversion)
        df_combined.dropna(subset=["model"], inplace=True)

    # Extract metadata and scores.
    # The `SampleSummary` columns include `metadata_*` and `score_*`.
    # We expect `metadata_variant` for difficulty and `metadata_base_task` for game name.
    # We need to determine how 'passed' is represented.
    # Let's assume `score_passed` is a boolean column from `score_*`
    # or derive it from a numeric score column like `score_accuracy`.
    # For this example, we'll try to find a column like 'score_passed'.
    # If not found, we will try to use 'score_accuracy' == 1 as passed.

    # Check for 'metadata.variant' and 'metadata.base_task' and rename if they have prefixes
    # Handle 'difficulty' from 'metadata_variant'
    if "metadata.variant" in df_combined.columns:
        df_combined.rename(columns={"metadata.variant": "difficulty"}, inplace=True)
    elif "metadata_variant" in df_combined.columns:  # Common pattern from inspect_ai
        df_combined.rename(columns={"metadata_variant": "difficulty"}, inplace=True)
    else:
        print(
            "Warning: 'difficulty' column (metadata.variant or metadata_variant) not found. Assigning 'base' as default."
        )
        df_combined["difficulty"] = "base"

    # Handle 'parent_task' from 'metadata_base_task'
    if "metadata.base_task" in df_combined.columns:
        df_combined.rename(columns={"metadata.base_task": "parent_task"}, inplace=True)
    elif "metadata_base_task" in df_combined.columns:
        df_combined.rename(columns={"metadata_base_task": "parent_task"}, inplace=True)
    else:
        print(
            "Warning: 'parent_task' column (metadata.base_task or metadata_base_task) not found."
        )
        # This could be critical if categories CSV relies on it.
        # Assigning NA; subsequent operations need to handle this.
        df_combined["parent_task"] = pd.NA

    # Determine 'passed' status
    # Prioritize 'score_webgames_scorer' == 'C' based on notebook insights
    if "score_webgames_scorer" in df_combined.columns:
        df_combined["passed"] = df_combined["score_webgames_scorer"] == "C"
    # Option 1: A direct 'score_passed' column
    elif "score_passed" in df_combined.columns:
        df_combined["passed"] = df_combined["score_passed"].astype(bool)
    # Option 2: Derive from a numeric score, e.g., 'score_accuracy' or 'score_value'
    elif "score_accuracy" in df_combined.columns:
        df_combined["passed"] = df_combined["score_accuracy"] == 1.0
    elif "score_value" in df_combined.columns:  # Generic numeric score
        # Assuming a score of 1.0 means pass for tasks where this is the max score.
        # This might need adjustment based on actual score range and meaning.
        df_combined["passed"] = df_combined["score_value"] == 1.0
    else:
        print(
            "Warning: Could not determine 'passed' status from 'score_passed', 'score_accuracy', or 'score_value'. Assuming all failed."
        )
        df_combined["passed"] = False

    # Load categories
    try:
        df_categories = pd.read_csv(CATEGORIES_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Categories file not found at {CATEGORIES_CSV_PATH}")
        # Continue without category merge if file not found, plots needing categories will be skipped.
        df_categories = pd.DataFrame(
            columns=["game"]
        )  # Empty df with 'game' column for consistent merge

    # Melt categories for easier merging and processing
    df_categories_melted = df_categories.melt(
        id_vars=["game"], var_name="category", value_name="weight"
    )
    # Rename 'game' in df_categories_melted to map to 'parent_task'
    df_categories_melted.rename(columns={"game": "parent_task_key"}, inplace=True)
    df_categories_melted = df_categories_melted[df_categories_melted["weight"] > 0]

    # Merge with main data
    df_merged = pd.merge(
        df_combined,
        df_categories_melted,
        left_on="parent_task",
        right_on="parent_task_key",
        how="left",
    )
    if "parent_task_key" in df_merged.columns:  # Clean up the merge key
        df_merged.drop(columns=["parent_task_key"], inplace=True)
    df_merged["category"] = df_merged["category"].fillna("Uncategorized")
    df_merged["weight"] = df_merged["weight"].fillna(0)

    # Create parent_task_difficulty_variant for filtering specific parent tasks
    if "parent_task" in df_merged.columns and "difficulty" in df_merged.columns:
        # Ensure 'parent_task' and 'difficulty' are strings before concatenation, handling potential NAs in parent_task
        df_merged["parent_task_str"] = (
            df_merged["parent_task"].astype(str).fillna("UnknownParent")
        )
        df_merged["difficulty_str"] = (
            df_merged["difficulty"].astype(str).fillna("UnknownDifficulty")
        )
        df_merged["_temp_parent_task_variant_for_filter"] = (
            df_merged["parent_task_str"] + "_" + df_merged["difficulty_str"]
        )

        # Define variants to exclude (based on parent_task and difficulty)
        excluded_variants = ["tab-sync-easy", "tab-sync-hard"]

        # Filter out the excluded variants
        initial_row_count = len(df_merged)
        df_merged = df_merged[~df_merged["game"].isin(excluded_variants)]
        rows_filtered = initial_row_count - len(df_merged)
        if rows_filtered > 0:
            print(
                f"Filtered out {rows_filtered} rows for excluded game IDs: {', '.join(excluded_variants)}"
            )
        # Clean up temporary columns used for the previous filtering method if they are no longer needed
        # For now, we are keeping parent_task_str, difficulty_str and _temp_parent_task_variant_for_filter if they were created
        # but the filter itself now uses the "game" column.
        # If these temp columns were ONLY for the old filter, they could be removed more cleanly.
        # Let's remove the one specifically for the old filter logic to avoid confusion.
        if "_temp_parent_task_variant_for_filter" in df_merged.columns:
            df_merged = df_merged.drop(columns=["_temp_parent_task_variant_for_filter"])
        # Keep parent_task_str and difficulty_str for now if they exist, as they might be used elsewhere or were intended for other purposes.

    else:
        print(
            "Warning: 'parent_task' or 'difficulty' columns not found in df_merged. Cannot create variant for exclusion (old method)."
        )
        # Attempt to filter based on game if excluded_variants are defined, even if parent_task/difficulty are missing for the old method
        excluded_variants = [
            "tab-sync-easy",
            "tab-sync-hard",
        ]  # Ensure defined for fallback
        if "game" in df_merged.columns and excluded_variants:
            initial_row_count = len(df_merged)
            df_merged = df_merged[~df_merged["game"].isin(excluded_variants)]
            rows_filtered = initial_row_count - len(df_merged)
            if rows_filtered > 0:
                print(
                    f"Filtered out {rows_filtered} rows for excluded game IDs (fallback): {', '.join(excluded_variants)}"
                )
        else:
            print(
                "Warning: 'game' column not found or no excluded_variants defined. Skipping exclusion."
            )

    # Create 'is_base_task' indicator
    # True if parent_task is the same as the game itself, and parent_task is not NA
    if "parent_task" in df_merged.columns and "game" in df_merged.columns:
        df_merged["is_base_task"] = (
            (df_merged["parent_task"] == df_merged["game"])
            & df_merged["parent_task"].notna()
            & df_merged["game"].notna()
        )
    else:
        df_merged["is_base_task"] = False  # Default if columns are missing

    # Save the processed data to cache
    try:
        with open(CACHE_FILE_PATH, "wb") as f:
            pickle.dump(df_merged, f)
            print(f"Saved processed data to cache: {CACHE_FILE_PATH}")
    except Exception as e:
        print(f"Error saving cache file: {e}")

    return df_merged


def save_plot(plt, output_dir: str, filename_without_extension: str, dpi: int = 300):
    """
    Save a plot in both PDF and PNG formats.

    Args:
        plt: The matplotlib pyplot object
        output_dir: Directory to save the files in
        filename_without_extension: Filename without extension (may include prefixes)
        dpi: DPI for the PNG output (default: 300)
    """
    # Use the filename as is, without splitting, since the input should not have an extension
    # The os.path.splitext was causing truncation with filenames containing dots (like model version numbers)
    base_filename = filename_without_extension

    # Save PDF
    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    plt.savefig(pdf_path)

    # Save PNG with high DPI
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=dpi)

    print(f"Generated {base_filename}.pdf and {base_filename}.png in {output_dir}")


def list_unsolved_tasks(df: pd.DataFrame, output_dir: str):
    """
    Identifies tasks that were not solved by any model and saves their IDs to a CSV file.
    A task is identified by its 'game' column (formerly 'id').
    """
    if "passed" not in df.columns or "game" not in df.columns:  # 'id' is now 'game'
        print("Skipping list_unsolved_tasks: 'passed' or 'game' column missing.")
        return

    if df.empty:
        print("Input DataFrame is empty. No unsolved tasks to list.")
        return

    # For each task 'game', check if it was solved by at least one model
    # Group by 'game' and check if 'any' of the 'passed' values are True for that group
    solved_by_any_model = df.groupby("game")["passed"].any()

    # Filter to get task IDs that were NOT solved by any model
    # These are the tasks where 'passed'.any() is False
    unsolved_task_ids_series = solved_by_any_model[~solved_by_any_model]

    if unsolved_task_ids_series.empty:
        print("All tasks were solved by at least one model. No unsolved tasks to list.")
        return

    unsolved_task_ids_list = unsolved_task_ids_series.index.tolist()

    # Create a DataFrame with a single column for the unsolved task IDs
    unsolved_df_output = pd.DataFrame(unsolved_task_ids_list, columns=["task_id"])

    # Use a more descriptive filename
    output_file_path = os.path.join(output_dir, "tasks_unsolved_by_any_model.csv")
    try:
        unsolved_df_output.to_csv(output_file_path, index=False)
        print(f"List of task IDs unsolved by any model saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving list of unsolved task IDs to CSV: {e}")


def list_tasks_solved_by_one_model(df: pd.DataFrame, output_dir: str):
    """
    Identifies tasks solved by exactly one model and saves their IDs and the solving model to a CSV file.
    """
    if not all(
        col in df.columns for col in ["game", "model", "passed"]
    ):  # 'id' is now 'game'
        print(
            "Skipping list_tasks_solved_by_one_model: Required columns ('game', 'model', 'passed') missing."
        )
        return

    if df.empty:
        print("Skipping list_tasks_solved_by_one_model: Input DataFrame is empty.")
        return

    # Filter for passed tasks
    passed_tasks_df = df[df["passed"] == True]

    if passed_tasks_df.empty:
        print(
            "No tasks were passed by any model. Cannot find tasks solved by one model."
        )
        return

    # Count how many models solved each task
    solved_counts = passed_tasks_df.groupby("game")[
        "model"
    ].nunique()  # 'id' is now 'game'

    # Get tasks solved by exactly one model
    tasks_solved_by_one_model_ids = solved_counts[solved_counts == 1].index

    if tasks_solved_by_one_model_ids.empty:
        print("No tasks were solved by exactly one model.")
        return

    # Get the details (model) for these tasks
    # We need to go back to passed_tasks_df and filter by these task ids
    result_df = (
        passed_tasks_df[
            passed_tasks_df["game"].isin(tasks_solved_by_one_model_ids)
        ][  # 'id' is now 'game'
            ["game", "model"]  # 'id' is now 'game'
        ]
        .drop_duplicates()  # Should be unique by id already due to the filter, but good practice
        .rename(
            columns={"game": "task_id", "model": "solving_model"}
        )  # task_id now refers to the individual 'game'
    )

    output_file_path = os.path.join(output_dir, "tasks_solved_by_exactly_one_model.csv")
    try:
        result_df.to_csv(output_file_path, index=False)
        print(f"List of tasks solved by exactly one model saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving tasks solved by one model to CSV: {e}")


def list_unsolved_task_families(df: pd.DataFrame, output_dir: str):
    """
    Identifies task families (e.g., 'brick-buster' based on the 'parent_task' column)
    that were not solved at any difficulty level and saves their names to a CSV.
    A task family is considered unsolved if no task instance belonging to that
    family (e.g., 'game-easy', 'game-hard', 'game-base') was passed.
    """
    required_cols = ["parent_task", "passed"]  # Changed from 'game' to 'parent_task'
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(
            f"Skipping list_unsolved_task_families: Required columns missing: {missing}."
        )
        return

    if df.empty:
        print("Skipping list_unsolved_task_families: Input DataFrame is empty.")
        return

    # Determine if any task within each 'parent_task' (family) was solved.
    family_solved_status = df.groupby("parent_task")[
        "passed"
    ].any()  # Changed from 'game'

    # Identify families where no task was solved (all entries for that family are False)
    unsolved_families_series = family_solved_status[~family_solved_status]

    if unsolved_families_series.empty:
        print(
            "All task families have at least one solved instance at some difficulty level. No unsolved parent_tasks to list."
        )
        return

    unsolved_families_list = unsolved_families_series.index.tolist()
    unsolved_families_df = pd.DataFrame(
        unsolved_families_list,
        columns=["unsolved_parent_task"],  # Changed column name
    )

    output_file_path = os.path.join(
        output_dir,
        "parent_tasks_unsolved_at_any_difficulty.csv",  # Changed filename
    )
    try:
        unsolved_families_df.to_csv(output_file_path, index=False)
        print(
            f"List of parent_task families unsolved at any difficulty saved to: {output_file_path}"
        )
    except Exception as e:
        print(f"Error saving list of unsolved parent_task families to CSV: {e}")


def save_02_average_success_per_difficulty_globally_csv(
    df: pd.DataFrame, output_dir: str
):
    """Calculates and saves the OVERALL average success rate per difficulty to a CSV file."""
    if not all(col in df.columns for col in ["difficulty", "passed"]):
        print(
            "Skipping 02_average_success_per_difficulty_globally_csv: Required columns ('difficulty', 'passed') missing."
        )
        return

    if df.empty:
        print(
            "Skipping 02_average_success_per_difficulty_globally_csv: Input DataFrame is empty."
        )
        return

    difficulty_order = ["easy", "base", "hard"]
    filtered_df = df[df["difficulty"] != "unknown"].copy()
    if filtered_df.empty:
        print(
            "No data for 02_average_success_per_difficulty_globally_csv after filtering 'unknown'."
        )
        return

    filtered_df["passed"] = pd.to_numeric(filtered_df["passed"], errors="coerce")
    filtered_df.dropna(subset=["passed"], inplace=True)

    average_success_df = (
        filtered_df.groupby("difficulty")["passed"].mean().reset_index()
    )
    average_success_df["difficulty"] = pd.Categorical(
        average_success_df["difficulty"], categories=difficulty_order, ordered=True
    )
    average_success_df = average_success_df.sort_values("difficulty").reset_index(
        drop=True
    )
    average_success_df.rename(columns={"passed": "average_success_rate"}, inplace=True)

    if average_success_df.empty:
        print(
            "No data to save for 02_average_success_rate_per_difficulty_globally.csv."
        )
        return

    output_file_path = os.path.join(
        output_dir, "02_average_success_rate_per_difficulty_globally.csv"
    )
    try:
        average_success_df.to_csv(output_file_path, index=False)
        print(
            f"Overall average success rate per difficulty saved to: {output_file_path}"
        )
    except Exception as e:
        print(f"Error saving 02_average_success_rate_per_difficulty_globally.csv: {e}")


def save_03_success_vs_difficulty_per_model_csv(
    df: pd.DataFrame, output_dir: str, model_order_global: list[str]
):
    """Calculates and saves success rate vs. difficulty per model to a CSV file."""
    if (
        "difficulty" not in df.columns
        or "passed" not in df.columns
        or "model" not in df.columns
    ):
        print(
            "Skipping 03_success_vs_difficulty_per_model_csv: Required columns missing."
        )
        return
    if df.empty:
        print("Skipping 03_success_vs_difficulty_per_model_csv: DataFrame is empty.")
        return

    difficulty_order = ["easy", "base", "hard"]
    summary_df = (
        df.groupby(["model", "difficulty"], observed=False)["passed"]
        .mean()
        .reset_index()
    )
    summary_df = summary_df[summary_df["difficulty"] != "unknown"]
    summary_df["difficulty"] = pd.Categorical(
        summary_df["difficulty"], categories=difficulty_order, ordered=True
    )
    # Ensure model is categorical according to global order
    summary_df["model"] = pd.Categorical(
        summary_df["model"], categories=model_order_global, ordered=True
    )
    summary_df = summary_df.sort_values(["model", "difficulty"])
    summary_df.rename(columns={"passed": "success_rate"}, inplace=True)

    if summary_df.empty:
        print("No data to save for 03_success_vs_difficulty_per_model.csv")
        return

    output_file_path = os.path.join(
        output_dir, "03_success_vs_difficulty_per_model.csv"
    )
    try:
        summary_df.to_csv(output_file_path, index=False)
        print(f"Success rate vs. difficulty per model saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving 03_success_vs_difficulty_per_model.csv: {e}")


def plot_03_success_vs_difficulty_per_model_chart(
    output_dir: str, model_order_global: list[str]
):
    """Plots success rate vs. difficulty per model from a CSV file."""
    csv_path = os.path.join(output_dir, "03_success_vs_difficulty_per_model.csv")
    try:
        summary_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(
            f"Skipping plot_03_success_vs_difficulty_per_model_chart: File not found: {csv_path}"
        )
        return

    if summary_df.empty:
        print(
            "Skipping plot_03_success_vs_difficulty_per_model_chart: CSV data is empty."
        )
        return

    difficulty_order = ["easy", "base", "hard"]
    summary_df["difficulty"] = pd.Categorical(
        summary_df["difficulty"], categories=difficulty_order, ordered=True
    )
    # Ensure model is categorical and ordered for consistent plotting
    summary_df["model"] = pd.Categorical(
        summary_df["model"], categories=model_order_global, ordered=True
    )
    summary_df = summary_df.sort_values(["model", "difficulty"])

    # Get a sorted list of models present in the data for hue_order
    # This ensures that only models actually in the CSV are used for hue_order
    # and they respect the global model order.
    models_in_data = summary_df["model"].unique().tolist()
    # Filter model_order_global to only those present in models_in_data to maintain order
    hue_order = [model for model in model_order_global if model in models_in_data]

    plt.figure(figsize=(12, 7))
    sns.barplot(
        x="difficulty",
        y="success_rate",  # Changed from "passed"
        hue="model",
        data=summary_df,
        palette=MODEL_PALETTE,
        hue_order=hue_order,  # Use the filtered and ordered list
    )
    plt.ylabel("Success Rate (0.0 - 1.0)", fontsize=18, weight="bold")
    plt.xlabel("Difficulty", fontsize=18, weight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(axis="y", visible=False)
    plt.grid(axis="x", visible=False)
    plt.tight_layout()
    save_plot(plt, output_dir, "03_success_vs_difficulty_per_model_chart")
    plt.close()


def save_average_success_per_difficulty_csv(df: pd.DataFrame, output_dir: str):
    """Calculates and saves the average success rate per difficulty to a CSV file."""
    if not all(col in df.columns for col in ["difficulty", "passed"]):
        print(
            "Skipping save_average_success_per_difficulty_csv: Required columns ('difficulty', 'passed') missing."
        )
        return

    if df.empty:
        print(
            "Skipping save_average_success_per_difficulty_csv: Input DataFrame is empty."
        )
        return

    # Define a custom order for difficulty, similar to plots
    difficulty_order = ["easy", "base", "hard"]

    # Calculate average success rate per difficulty
    # Filter out 'unknown' difficulty
    filtered_df = df[df["difficulty"] != "unknown"].copy()
    if filtered_df.empty:
        print(
            "No data available for average success rate per difficulty after filtering 'unknown'."
        )
        return

    # Ensure 'passed' is numeric for mean calculation
    filtered_df["passed"] = pd.to_numeric(filtered_df["passed"], errors="coerce")
    # Drop rows where 'passed' could not be coerced to numeric, if any
    filtered_df.dropna(subset=["passed"], inplace=True)

    average_success_df = (
        filtered_df.groupby("difficulty")["passed"].mean().reset_index()
    )

    # Ensure the 'difficulty' column is categorical and ordered
    average_success_df["difficulty"] = pd.Categorical(
        average_success_df["difficulty"], categories=difficulty_order, ordered=True
    )
    average_success_df = average_success_df.sort_values("difficulty").reset_index(
        drop=True
    )

    average_success_df.rename(columns={"passed": "average_success_rate"}, inplace=True)

    if average_success_df.empty:
        print("No data to save for average success rate per difficulty.")
        return

    output_file_path = os.path.join(
        output_dir, "average_success_rate_per_difficulty.csv"
    )
    try:
        average_success_df.to_csv(output_file_path, index=False)
        print(f"Average success rate per difficulty saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving average success rate per difficulty to CSV: {e}")


def save_11_model_category_performance_csv_and_tex(df: pd.DataFrame, output_dir: str):
    """
    Generates a CSV table showing model performance per category with averages.
    Rows are models, columns are categories, and cells contain success rates.
    Includes row and column averages.
    """
    required_cols = ["model", "category", "passed", "weight"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(
            f"Skipping model-category performance table: Missing required columns: {missing}"
        )
        return
    if df.empty:
        print("Skipping model-category performance table: DataFrame is empty.")
        return

    # Create a copy and ensure passed is numeric
    df_copy = df.copy()
    df_copy["passed_numeric"] = df_copy["passed"].astype(int)
    df_copy["weighted_passed_score"] = df_copy["passed_numeric"] * df_copy["weight"]

    # Calculate weighted success rate per model and category
    model_category_performance = (
        df_copy.groupby(["model", "category"], observed=False)
        .agg(
            total_weighted_passed=("weighted_passed_score", "sum"),
            total_weight=("weight", "sum"),
        )
        .reset_index()
    )

    # Calculate success rate
    model_category_performance["success_rate"] = (
        model_category_performance["total_weighted_passed"]
        / model_category_performance["total_weight"]
    ).fillna(0)

    # Pivot the table to get models as rows and categories as columns
    pivot_table = model_category_performance.pivot(
        index="model", columns="category", values="success_rate"
    ).fillna(0)

    # Sort pivot_table by model name (index) before adding 'Average' row
    # This will respect the categorical order set in load_and_preprocess_data
    pivot_table = pivot_table.sort_index(ascending=True)

    # Remove 'Uncategorized' column if it exists
    if "Uncategorized" in pivot_table.columns:
        pivot_table = pivot_table.drop(columns=["Uncategorized"])

    # Calculate row averages (model averages across categories)
    pivot_table["Average"] = pivot_table.mean(axis=1)

    # Calculate column averages (category averages across models)
    category_means = pivot_table.mean()
    pivot_table.loc["Average"] = category_means
    # Ensure 'Average' row is at the bottom
    if "Average" in pivot_table.index:
        avg_row = pivot_table.loc[["Average"]]
        pivot_table = pivot_table.drop(index="Average")
        pivot_table = pd.concat([pivot_table, avg_row])

    # Format all values as percentages for CSV
    formatted_table_csv = pivot_table.map(
        lambda x: f"{x:.1%}" if pd.notnull(x) else "-"
    )

    # Save to CSV
    output_file_path_csv = os.path.join(
        output_dir, "11_model_category_performance.csv"
    )  # Will be renamed later
    try:
        formatted_table_csv.to_csv(output_file_path_csv)
        print(f"Model-category performance table saved to: {output_file_path_csv}")
    except Exception as e:
        print(f"Error saving model-category performance table to CSV: {e}")

    # Prepare table for LaTeX
    latex_table_for_tex = pivot_table.copy()
    latex_table_for_tex = (latex_table_for_tex * 100).round(1)

    # Map values to strings with percent sign for LaTeX
    latex_table_percent_str_df = latex_table_for_tex.map(
        lambda x: f"{x:.1f}\\%" if pd.notnull(x) else "-"  # Escaped % for LaTeX
    )

    output_file_path_latex = os.path.join(
        output_dir, "11_model_category_performance.tex"
    )  # Will be renamed later
    try:
        latex_string = latex_table_percent_str_df.to_latex(  # type: ignore
            na_rep="-",
            escape=False,
        )
        latex_table_code = latex_string

        with open(output_file_path_latex, "w") as f:
            f.write(latex_table_code)
        print(
            f"Model-category performance LaTeX table saved to: {output_file_path_latex}"
        )
    except Exception as e:
        print(f"Error saving model-category performance table to LaTeX: {e}")
        raise e


def plot_00_game_id_model_pass_status_heatmap(
    output_dir: str, model_order_global: list[str]
):
    """Plots a heatmap of game ID vs model pass status from a CSV file."""
    csv_path = os.path.join(output_dir, "00_game_id_model_pass_status.csv")
    try:
        pass_matrix_df = pd.read_csv(csv_path, index_col="game")  # Set 'game' as index
    except FileNotFoundError:
        print(
            f"Skipping plot_00_game_id_model_pass_status_heatmap: File not found: {csv_path}"
        )
        return
    except Exception as e:
        print(f"Error reading {csv_path} for heatmap: {e}")
        return

    if pass_matrix_df.empty:
        print("Skipping plot_00_game_id_model_pass_status_heatmap: CSV data is empty.")
        return

    # Ensure columns (model names) are in the desired order
    # Filter model_order_global to only those present in the CSV columns
    ordered_models_in_data = [
        m for m in model_order_global if m in pass_matrix_df.columns
    ]
    pass_matrix_df = pass_matrix_df.reindex(columns=ordered_models_in_data)

    # Dynamic figure size based on matrix dimensions
    num_games = len(pass_matrix_df.index)
    num_models = len(pass_matrix_df.columns)

    fig_width = max(10, num_models * 0.6)  # Adjusted multiplier
    fig_height = max(8, num_games * 0.2)  # Adjusted multiplier

    fig_width = min(fig_width, 40)  # Max width
    fig_height = min(
        fig_height, 100
    )  # Max height, increased for potentially many games

    plt.figure(figsize=(fig_width, fig_height))
    try:
        sns.heatmap(
            pass_matrix_df.astype(bool),  # Convert 0/1 to False/True for RdYlGn
            annot=False,
            cmap="RdYlGn",  # Red for False (0), Green for True (1)
            cbar=False,
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title("Game Pass Status by Model (Green=Passed, Red=Failed)", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Game ID", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(
            rotation=0, fontsize=max(6, 10 - num_games // 20)
        )  # Dynamic ytick font size
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for title

        save_plot(plt, output_dir, "00_game_id_model_pass_status_heatmap")
    except Exception as e:
        print(f"Error generating heatmap for 00_game_id_model_pass_status_heatmap: {e}")
    finally:
        plt.close()


def save_00_game_id_model_pass_status_csv(
    df: pd.DataFrame, output_dir: str, model_order_global: list[str]
):
    """
    Generates a CSV file showing which game ID was passed by which model.
    Rows: Game IDs, Columns: Models, Values: 1 if passed, 0 otherwise.
    This is the '00_' prefixed artifact.
    """
    required_cols = ["game", "model", "passed"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(
            f"Skipping 00_game_id_model_pass_status_csv: Missing required columns: {missing}"
        )
        return
    if df.empty:
        print("Skipping 00_game_id_model_pass_status_csv: DataFrame is empty.")
        return

    df_copy = df.copy()
    df_copy["passed_int"] = df_copy["passed"].astype(int)

    try:
        pass_matrix_df = df_copy.pivot_table(
            index="game",
            columns="model",
            values="passed_int",
            fill_value=0,
            observed=False,
        )
        ordered_models_in_data = [
            m for m in model_order_global if m in pass_matrix_df.columns
        ]
        pass_matrix_df = pass_matrix_df.reindex(
            columns=ordered_models_in_data, fill_value=0
        )
    except Exception as e:
        print(f"Error creating pivot table for 00_game_id_model_pass_status_csv: {e}")
        return

    if pass_matrix_df.empty:
        print(
            "Pivoted data for 00_game_id_model_pass_status_csv is empty. Skipping CSV generation."
        )
        return

    pass_matrix_df_int = pass_matrix_df.astype(int)

    csv_filename = "00_game_id_model_pass_status.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    try:
        pass_matrix_df_int.to_csv(csv_filepath)
        print(f"Generated {csv_filename} in {output_dir}")
    except Exception as e:
        print(f"Error saving {csv_filename} to CSV: {e}")


def save_01_overall_success_per_model_csv(
    df: pd.DataFrame, output_dir: str, model_order_global: list[str]
):
    """
    Calculates and saves the overall success rate per model to a CSV file.
    This version reads from the '00_game_id_model_pass_status.csv' file.
    """
    # Path to the 00_game_id_model_pass_status.csv file
    source_csv_path = os.path.join(output_dir, "00_game_id_model_pass_status.csv")

    try:
        pass_matrix_df = pd.read_csv(source_csv_path, index_col="game")
    except FileNotFoundError:
        print(
            f"Skipping 01_overall_success_per_model_csv: Source file not found: {source_csv_path}"
        )
        return
    except Exception as e:
        print(
            f"Skipping 01_overall_success_per_model_csv: Error reading {source_csv_path}: {e}"
        )
        return

    if pass_matrix_df.empty:
        print("Skipping 01_overall_success_per_model_csv: Source CSV data is empty.")
        return

    # Calculate success rate for each model
    # Success rate = (sum of passed games for a model) / (total number of games)
    # Total number of games is the number of rows in pass_matrix_df
    total_games = len(pass_matrix_df)
    if total_games == 0:
        print(
            "Skipping 01_overall_success_per_model_csv: No games found in source CSV."
        )
        return

    # Calculate the sum of passes for each model (each column)
    # .sum() on a DataFrame with 0/1 will give the count of 1s (passes) per column
    model_success_counts = pass_matrix_df.sum()

    # Calculate success rate
    model_success_rates = model_success_counts / total_games

    # Convert to DataFrame
    model_summary = model_success_rates.reset_index()
    model_summary.columns = ["model", "overall_success_rate"]

    # Ensure model column is categorical and ordered according to model_order_global
    # Filter out models not in model_order_global if any, then apply categorical
    model_summary = model_summary[model_summary["model"].isin(model_order_global)]
    model_summary["model"] = pd.Categorical(
        model_summary["model"], categories=model_order_global, ordered=True
    )
    model_summary = model_summary.sort_values("model").reset_index(drop=True)

    if model_summary.empty:
        print("No data to save for 01_overall_success_per_model_csv after processing.")
        return

    output_file_path = os.path.join(output_dir, "01_overall_success_per_model.csv")
    try:
        model_summary.to_csv(output_file_path, index=False)
        print(f"Overall success rate per model saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving 01_overall_success_per_model.csv: {e}")


def plot_01_overall_success_per_model_chart(
    output_dir: str, model_order_global: list[str]
):
    """Plots overall success rate per model from a CSV file."""
    csv_path = os.path.join(output_dir, "01_overall_success_per_model.csv")
    try:
        model_summary = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(
            f"Skipping plot_01_overall_success_per_model_chart: File not found: {csv_path}"
        )
        return

    if model_summary.empty:
        print("Skipping plot_01_overall_success_per_model_chart: CSV data is empty.")
        return

    model_summary["model"] = pd.Categorical(
        model_summary["model"], categories=model_order_global, ordered=True
    )
    model_summary = model_summary.sort_values("model")

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="overall_success_rate",
        y="model",
        data=model_summary,
        hue="model",
        palette=MODEL_PALETTE,
        legend=False,
        order=model_summary["model"].tolist(),
    )
    plt.xlabel("Success Rate (0.0 - 1.0)", fontsize=18, weight="bold")
    plt.ylabel("Model", fontsize=18, weight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis="y", visible=False)
    plt.grid(axis="x", visible=False)
    plt.tight_layout()
    save_plot(plt, output_dir, "01_overall_success_per_model_chart")
    plt.close()


def save_04_success_rate_per_category_csv(df: pd.DataFrame, output_dir: str):
    """Calculates and saves weighted success rate per category to a CSV file."""
    if (
        "category" not in df.columns
        or "passed" not in df.columns
        or "weight" not in df.columns
    ):
        print("Skipping 04_success_rate_per_category_csv: Required columns missing.")
        return
    if df.empty:
        print("Skipping 04_success_rate_per_category_csv: DataFrame is empty.")
        return

    df_copy = df.copy()  # Work on a copy
    df_copy["passed_numeric"] = df_copy["passed"].astype(int)
    df_copy["weighted_passed_score"] = df_copy["passed_numeric"] * df_copy["weight"]

    category_summary = (
        df_copy.groupby("category")
        .agg(
            total_weighted_passed_score=("weighted_passed_score", "sum"),
            total_weight=("weight", "sum"),
        )
        .reset_index()
    )

    category_summary["success_rate"] = (
        category_summary["total_weighted_passed_score"]
        / category_summary["total_weight"]
    ).fillna(0)  # fillna(0) in case total_weight is 0
    category_summary = category_summary[category_summary["total_weight"] > 0]
    category_summary = category_summary.sort_values("success_rate", ascending=False)

    if category_summary.empty:
        print("No data to save for 04_success_rate_per_category.csv")
        return

    output_file_path = os.path.join(output_dir, "04_success_rate_per_category.csv")
    try:
        category_summary[
            ["category", "success_rate", "total_weighted_passed_score", "total_weight"]
        ].to_csv(output_file_path, index=False)
        print(f"Weighted success rate per category saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving 04_success_rate_per_category.csv: {e}")


def plot_04_success_rate_per_category_chart(output_dir: str):
    """Plots overall weighted success rate per category from a CSV file."""
    csv_path = os.path.join(output_dir, "04_success_rate_per_category.csv")
    try:
        category_summary = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(
            f"Skipping plot_04_success_rate_per_category_chart: File not found: {csv_path}"
        )
        return

    if category_summary.empty:
        print("Skipping plot_04_success_rate_per_category_chart: CSV data is empty.")
        return

    # Sort by success_rate for plotting if not already sorted in CSV (it is, but good practice)
    category_summary = category_summary.sort_values("success_rate", ascending=False)

    plt.figure(figsize=(14, 8))
    sns.barplot(
        x="success_rate",
        y="category",
        data=category_summary,
        hue="category",
        palette="mako",
        legend=False,
    )
    # plt.title("Overall Weighted Success Rate per Game Category") # Title removed
    plt.xlabel("Weighted Success Rate (0.0 - 1.0)", fontsize=18, weight="bold")
    plt.ylabel("Game Category", fontsize=18, weight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.grid(axis="y", visible=False)
    plt.tight_layout()
    save_plot(plt, output_dir, "04_success_rate_per_category_chart")
    plt.close()


def save_05_success_rate_per_difficulty_per_category_csv(
    df: pd.DataFrame, output_dir: str
):
    """Prepares and saves data for success rate per difficulty per category plots to a CSV file."""
    if (
        "category" not in df.columns
        or "difficulty" not in df.columns
        or "passed" not in df.columns
    ):
        print(
            "Skipping 05_success_rate_per_difficulty_per_category_csv: Required columns missing."
        )
        return
    if df.empty:
        print(
            "Skipping 05_success_rate_per_difficulty_per_category_csv: DataFrame is empty."
        )
        return

    difficulty_order = ["easy", "base", "hard"]
    # Filter out 'Uncategorized' category and 'unknown' difficulty
    csv_df = df[
        (df["category"] != "Uncategorized") & (df["difficulty"] != "unknown")
    ].copy()

    if csv_df.empty:
        print(
            "No data available for 05_success_rate_per_difficulty_per_category_csv after filtering."
        )
        return

    # Calculate mean of 'passed' for the plot (which will be done by catplot but good to have in CSV)
    # The catplot itself will do the aggregation, so we save the raw data needed for it.
    # Ensure 'passed' is float for mean calculation by catplot/pivot.
    csv_df["passed"] = csv_df["passed"].astype(float)
    csv_df["difficulty"] = pd.Categorical(
        csv_df["difficulty"], categories=difficulty_order, ordered=True
    )
    # Sort values to ensure consistent plot order if not inherently handled by catplot grouping
    csv_df = csv_df.sort_values(["category", "difficulty"])

    # Select relevant columns for the CSV
    # The catplot will use category, difficulty, and passed.
    # If there are other dimensions like 'model' that might be implicitly aggregated or filtered,
    # they should be considered or explicitly handled.
    # For this specific plot, it seems to average over all models for each cat/diff pair.
    # So, the raw data for these columns is what's needed.
    # Let's refine the CSV to be the direct input for a simple barplot if catplot wasn't used.
    # Success rate = mean of 'passed' grouped by category and difficulty

    # Actually, the catplot function does the groupby and mean internally.
    # So, saving the filtered df with relevant columns is appropriate.
    # The plot function will then read this and pass to catplot.
    output_columns = [
        "category",
        "difficulty",
        "passed",
    ]  # Add other relevant columns if needed by catplot
    # If model or other dimensions are implicitly part of the aggregation in the original plot,
    # ensure they are present in csv_df if catplot needs them.
    # For this specific plot, the original function did not seem to use model in its catplot call directly for hue etc.
    # It was df -> catplot(x=difficulty, y=passed, col=category)

    # We need to compute the mean success rate here for the CSV.
    # The plot was: sns.catplot(x="difficulty", y="passed", col="category", data=plot_df, kind="bar", ...)
    # This means 'passed' is averaged for each difficulty within each category.

    # Let's calculate the mean success rate for the CSV explicitly.
    agg_df = (
        csv_df.groupby(["category", "difficulty"], observed=False)["passed"]
        .mean()
        .reset_index()
    )
    agg_df.rename(columns={"passed": "success_rate"}, inplace=True)

    if agg_df.empty:
        print(
            "No aggregated data to save for 05_success_rate_per_difficulty_per_category.csv"
        )
        return

    output_file_path = os.path.join(
        output_dir, "05_success_rate_per_difficulty_per_category.csv"
    )
    try:
        agg_df.to_csv(output_file_path, index=False)
        print(
            f"Success rate per difficulty per category data saved to: {output_file_path}"
        )
    except Exception as e:
        print(f"Error saving 05_success_rate_per_difficulty_per_category.csv: {e}")


def plot_05_success_rate_per_difficulty_per_category_chart(output_dir: str):
    """Plots success rate per difficulty per category from a CSV file using catplot."""
    csv_path = os.path.join(
        output_dir, "05_success_rate_per_difficulty_per_category.csv"
    )
    try:
        plot_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(
            f"Skipping plot_05_success_rate_per_difficulty_per_category_chart: File not found: {csv_path}"
        )
        return

    if plot_df.empty:
        print(
            "Skipping plot_05_success_rate_per_difficulty_per_category_chart: CSV data is empty."
        )
        return

    difficulty_order = ["easy", "base", "hard"]
    plot_df["difficulty"] = pd.Categorical(
        plot_df["difficulty"], categories=difficulty_order, ordered=True
    )
    # Sort values for consistent plotting if needed (catplot should handle grouping)
    plot_df = plot_df.sort_values(["category", "difficulty"])

    g = sns.catplot(
        x="difficulty",
        y="success_rate",  # Changed from "passed" to "success_rate" from CSV
        hue="difficulty",
        col="category",
        data=plot_df,
        kind="bar",
        col_wrap=3,
        palette="viridis",
        height=4,
        aspect=1.2,
        sharey=True,
        legend=False,
    )
    g.set_axis_labels("Difficulty", "Success Rate (0.0 - 1.0)")
    g.set_titles("{col_name}")
    # g.fig.suptitle("Success Rate per Difficulty per Category", y=1.03, fontsize=16) # Title removed
    plt.tight_layout(rect=(0, 0, 1, 0.97))  # Adjusted for suptitle if it were present

    save_plot(plt, output_dir, "05_success_rate_per_difficulty_per_category_chart")
    plt.close(g.fig)


def save_06_weighted_unsolved_tasks_by_category_csv(df: pd.DataFrame, output_dir: str):
    """Calculates and saves the weighted count of completely unsolved tasks by category to a CSV file."""
    required_cols = ["game", "passed", "category", "weight"]
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(
            f"Skipping 06_weighted_unsolved_tasks_by_category_csv: Required columns missing: {missing_cols}."
        )
        return
    if df.empty:
        print(
            "Skipping 06_weighted_unsolved_tasks_by_category_csv: Input DataFrame is empty."
        )
        return

    solved_by_any_model = df.groupby("game")["passed"].any()
    unsolved_task_ids = solved_by_any_model[~solved_by_any_model].index

    if unsolved_task_ids.empty:
        print("No completely unsolved tasks found. Skipping CSV generation for 06_...")
        # Create an empty CSV with headers if desired, or just return
        # For now, just returning as the plot function will also skip.
        return

    unsolved_df_details = df[df["game"].isin(unsolved_task_ids)]
    if unsolved_df_details.empty:
        print(
            "No details found for unsolved tasks (this should not happen if unsolved_task_ids is not empty). Skipping 06_..."
        )
        return

    unique_unsolved_task_category_weights = unsolved_df_details[
        ["game", "category", "weight"]
    ].drop_duplicates()

    category_unsolved_summary = (
        unique_unsolved_task_category_weights.groupby("category")["weight"]
        .sum()
        .reset_index()
    )
    category_unsolved_summary = category_unsolved_summary.rename(
        columns={"weight": "total_weighted_unsolved_tasks"}
    )
    category_unsolved_summary = category_unsolved_summary[
        category_unsolved_summary["total_weighted_unsolved_tasks"] > 0
    ]
    category_unsolved_summary = category_unsolved_summary.sort_values(
        "total_weighted_unsolved_tasks", ascending=False
    )

    if category_unsolved_summary.empty:
        print(
            "No weighted unsolved tasks to save for 06_weighted_unsolved_tasks_by_category.csv after processing."
        )
        return

    output_file_path = os.path.join(
        output_dir, "06_weighted_unsolved_tasks_by_category.csv"
    )
    try:
        category_unsolved_summary.to_csv(output_file_path, index=False)
        print(f"Weighted unsolved tasks by category saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving 06_weighted_unsolved_tasks_by_category.csv: {e}")


def plot_06_weighted_unsolved_tasks_by_category_chart(output_dir: str):
    """Plots the weighted count of completely unsolved tasks by category from a CSV file."""
    csv_path = os.path.join(output_dir, "06_weighted_unsolved_tasks_by_category.csv")
    try:
        category_unsolved_summary = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(
            f"Skipping plot_06_weighted_unsolved_tasks_by_category_chart: File not found: {csv_path}"
        )
        return

    if category_unsolved_summary.empty:
        print(
            "Skipping plot_06_weighted_unsolved_tasks_by_category_chart: CSV data is empty."
        )
        return

    # Ensure sorting for consistent plot appearance (already sorted in CSV, but good practice)
    category_unsolved_summary = category_unsolved_summary.sort_values(
        "total_weighted_unsolved_tasks", ascending=False
    )

    plt.figure(figsize=(14, 8))
    sns.barplot(
        x="total_weighted_unsolved_tasks",
        y="category",
        data=category_unsolved_summary,
        hue="category",
        palette="viridis",
        legend=False,
    )
    # plt.title("Weighted Count of Completely Unsolved Tasks by Category") # Title removed
    plt.xlabel("Total Weighted Unsolved Tasks", fontsize=18, weight="bold")
    plt.ylabel("Game Category", fontsize=18, weight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.grid(axis="y", visible=False)
    plt.tight_layout()
    save_plot(plt, output_dir, "06_weighted_unsolved_tasks_by_category_chart")
    plt.close()


def save_07_model_performance_radar_data_csv(df: pd.DataFrame, output_dir: str):
    """Prepares and saves normalized model performance data per category for radar plot to CSV."""
    required_cols = ["game", "model", "category", "weight", "passed"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(
            f"Skipping 07_model_performance_radar_data_csv: Missing required columns: {missing}"
        )
        return
    if df.empty:
        print("Skipping 07_model_performance_radar_data_csv: DataFrame is empty.")
        return

    df_copy = df.copy()
    if "weighted_passed_score" not in df_copy.columns:
        df_copy["passed_numeric"] = df_copy["passed"].astype(int)
        df_copy["weighted_passed_score"] = df_copy["passed_numeric"] * df_copy["weight"]

    model_category_performance = (
        df_copy.groupby(["model", "category"], observed=False)["weighted_passed_score"]
        .sum()
        .reset_index()
    )

    unique_task_category_weights = df_copy[
        ["game", "category", "weight"]
    ].drop_duplicates()
    total_score_per_cat_df = (
        unique_task_category_weights.groupby("category")["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "total_possible_score"})
    )
    total_score_per_cat_df = total_score_per_cat_df[
        total_score_per_cat_df["total_possible_score"] > 0
    ]

    if total_score_per_cat_df.empty:
        print("Skipping 07_...: No categories with positive total possible scores.")
        return

    merged_data = pd.merge(
        model_category_performance,
        total_score_per_cat_df,
        on="category",
        how="inner",
    )

    if merged_data.empty:
        print(
            "Skipping 07_...: No data after merging model performance with category scores."
        )
        return

    merged_data["normalized_performance"] = (
        merged_data["weighted_passed_score"] / merged_data["total_possible_score"]
    ).clip(0, 1)

    # Pivot table for the radar plot data structure
    pivot_df = merged_data.pivot(
        index="model", columns="category", values="normalized_performance"
    ).fillna(0)
    pivot_df = pivot_df.sort_index(ascending=True)  # Sort by model name (index)

    if pivot_df.empty:
        print("Skipping 07_...: Pivoted data is empty.")
        return

    # Save the pivot_df which is the direct input for the radar plot structure
    output_file_path = os.path.join(output_dir, "07_model_performance_radar_data.csv")
    try:
        pivot_df.to_csv(output_file_path)  # index=True is default and correct here
        print(
            f"Normalized model performance per category (radar data) saved to: {output_file_path}"
        )
    except Exception as e:
        print(f"Error saving 07_model_performance_radar_data.csv: {e}")


def plot_07_model_performance_radar_chart(output_dir: str):
    """Plots a radar chart of normalized model performance per category from a CSV file."""
    csv_path = os.path.join(output_dir, "07_model_performance_radar_data.csv")
    try:
        # Read the pivot table, model is the index
        pivot_df = pd.read_csv(csv_path, index_col="model")
    except FileNotFoundError:
        print(
            f"Skipping plot_07_model_performance_radar_chart: File not found: {csv_path}"
        )
        return

    if pivot_df.empty:
        print("Skipping plot_07_model_performance_radar_chart: CSV data is empty.")
        return

    categories = pivot_df.columns.tolist()
    if not categories:
        print("Skipping plot_07_...: No categories to plot from CSV.")
        return

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    max_val = pivot_df.max().max()
    if max_val <= 0:
        max_val = 0.2
    elif max_val <= 0.5:
        max_val = np.ceil(max_val * 10) / 10
    elif max_val < 1.0:
        max_val = np.ceil(max_val / 0.2) * 0.2
    else:
        max_val = np.ceil(max_val)

    for model_name in pivot_df.index:
        values = pivot_df.loc[model_name].tolist()
        closed_values = values + values[:1]
        color = MODEL_PALETTE.get(str(model_name), "#808080")
        ax.plot(
            closed_angles,
            closed_values,
            linewidth=1.5,
            label=str(model_name),
            color=color,
        )
        ax.fill(closed_angles, closed_values, alpha=0.2, color=color)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=18, fontweight="bold")  # Increased fontsize
    ax.tick_params(axis="x", which="major", pad=30)  # Move category labels further out
    tick_step = max_val / 5
    if tick_step <= 0.02:
        tick_step = 0.02
    elif tick_step <= 0.05:
        tick_step = 0.05
    elif tick_step <= 0.1:
        tick_step = 0.1
    elif tick_step <= 0.2:
        tick_step = 0.2
    elif tick_step <= 0.25:
        tick_step = 0.25
    else:
        tick_step = 0.5
    ax.set_yticks(np.arange(0, max_val + tick_step, tick_step))
    ax.set_yticklabels(
        [f"{tick:.1f}" for tick in np.arange(0, max_val + tick_step, tick_step)],
        fontsize=16,
    )  # Added yticklabels with fontsize
    ax.set_ylim(0, max_val)

    num_models = len(pivot_df.index)
    legend_ncol = max(1, num_models // 4 if num_models > 2 else num_models)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.4),  # Moved legend further down
        ncol=legend_ncol,
        fontsize=18,  # Increased legend fontsize
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_plot(plt, output_dir, "07_model_performance_radar_chart")
    plt.close(fig)  # Added plt.close(fig) back


def save_08_tasks_unsolved_by_any_model_csv(df: pd.DataFrame, output_dir: str):
    """Identifies tasks unsolved by any model and saves their IDs to a CSV."""
    if "passed" not in df.columns or "game" not in df.columns:
        print(
            "Skipping 08_tasks_unsolved_by_any_model_csv: 'passed' or 'game' column missing."
        )
        return
    if df.empty:
        print("Skipping 08_tasks_unsolved_by_any_model_csv: DataFrame is empty.")
        return

    solved_by_any_model = df.groupby("game")["passed"].any()
    unsolved_task_ids_series = solved_by_any_model[~solved_by_any_model]

    if unsolved_task_ids_series.empty:
        print(
            "All tasks solved by at least one model. No CSV for 08_tasks_unsolved_by_any_model."
        )
        return

    unsolved_df_output = pd.DataFrame(
        unsolved_task_ids_series.index.tolist(), columns=["task_id"]
    )
    output_file_path = os.path.join(output_dir, "08_tasks_unsolved_by_any_model.csv")
    try:
        unsolved_df_output.to_csv(output_file_path, index=False)
        print(f"List of task IDs unsolved by any model saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving 08_tasks_unsolved_by_any_model.csv: {e}")


def save_09_tasks_solved_by_exactly_one_model_csv(df: pd.DataFrame, output_dir: str):
    """Identifies tasks solved by one model and saves their IDs and solving model to CSV."""
    if not all(col in df.columns for col in ["game", "model", "passed"]):
        print(
            "Skipping 09_tasks_solved_by_exactly_one_model_csv: Required columns missing."
        )
        return
    if df.empty:
        print("Skipping 09_tasks_solved_by_exactly_one_model_csv: DataFrame is empty.")
        return

    passed_tasks_df = df[df["passed"] == True]
    if passed_tasks_df.empty:
        print(
            "No tasks passed by any model. Skipping 09_tasks_solved_by_exactly_one_model_csv."
        )
        return

    solved_counts = passed_tasks_df.groupby("game")["model"].nunique()
    tasks_solved_by_one_model_ids = solved_counts[solved_counts == 1].index

    if tasks_solved_by_one_model_ids.empty:
        print(
            "No tasks solved by exactly one model. No CSV for 09_tasks_solved_by_exactly_one_model."
        )
        return

    result_df = (
        passed_tasks_df[passed_tasks_df["game"].isin(tasks_solved_by_one_model_ids)][
            ["game", "model"]
        ]
        .drop_duplicates()
        .rename(columns={"game": "task_id", "model": "solving_model"})
    )
    output_file_path = os.path.join(
        output_dir, "09_tasks_solved_by_exactly_one_model.csv"
    )
    try:
        result_df.to_csv(output_file_path, index=False)
        print(f"Tasks solved by exactly one model saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving 09_tasks_solved_by_exactly_one_model.csv: {e}")


def save_10_parent_tasks_unsolved_at_any_difficulty_csv(
    df: pd.DataFrame, output_dir: str
):
    """Identifies task families (parent_task) unsolved at any difficulty and saves names to CSV."""
    required_cols = ["parent_task", "passed"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(
            f"Skipping 10_parent_tasks_unsolved_csv: Required columns missing: {missing}."
        )
        return
    if df.empty:
        print("Skipping 10_parent_tasks_unsolved_csv: DataFrame is empty.")
        return

    family_solved_status = df.groupby("parent_task")["passed"].any()
    unsolved_families_series = family_solved_status[~family_solved_status]

    if unsolved_families_series.empty:
        print(
            "All task families have at least one solved instance. No CSV for 10_parent_tasks_unsolved."
        )
        return

    unsolved_families_df = pd.DataFrame(
        unsolved_families_series.index.tolist(), columns=["unsolved_parent_task"]
    )
    output_file_path = os.path.join(
        output_dir, "10_parent_tasks_unsolved_at_any_difficulty.csv"
    )
    try:
        unsolved_families_df.to_csv(output_file_path, index=False)
        print(f"Parent tasks unsolved at any difficulty saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving 10_parent_tasks_unsolved_at_any_difficulty.csv: {e}")


def save_per_model_category_contributions(
    df: pd.DataFrame,
    output_dir: str,
    categories_csv_path: str,
    model_order_list: list[str],
):
    """
    Generates CSV files per model, showing category contributions for passed tasks.
    For each task a model passed, it lists the task and the category weights
    of its parent task.
    """
    per_model_output_dir = os.path.join(output_dir, "per_model")
    if not os.path.exists(per_model_output_dir):
        os.makedirs(per_model_output_dir)
        print(f"Created directory: {per_model_output_dir}")

    try:
        categories_df_raw = pd.read_csv(categories_csv_path)
    except FileNotFoundError:
        print(
            f"Error: Categories file not found at {categories_csv_path}. Skipping per-model category contributions."
        )
        return

    if "game" not in categories_df_raw.columns:
        print(
            f"Error: 'game' column not found in {categories_csv_path}. Skipping per-model category contributions."
        )
        return

    category_columns = [col for col in categories_df_raw.columns if col != "game"]

    # Get models present in the data to avoid iterating over models with no data
    present_models_in_df = df["model"].unique()

    for model_name in model_order_list:
        if model_name not in present_models_in_df:
            # print(f"Model {model_name} not found in processed data. Skipping its category contribution CSV.")
            continue

        model_specific_df = df[(df["model"] == model_name) & (df["passed"] == True)]

        if model_specific_df.empty:
            # print(f"No passed tasks for model {model_name}. Skipping its category contribution CSV.")
            continue

        # Get unique passed games and their parent tasks for this model
        # Ensure parent_task is not NaN to avoid issues during lookup
        passed_games_info = model_specific_df[model_specific_df["parent_task"].notna()][
            ["game", "parent_task"]
        ].drop_duplicates()

        if passed_games_info.empty:
            # print(f"No passed tasks with valid parent_task for model {model_name}. Skipping.")
            continue

        result_for_model_list = []
        for _, row in passed_games_info.iterrows():
            actual_game_id = row["game"]
            parent_game_id = row["parent_task"]

            # Find the category contributions for the parent_game_id
            category_row_for_parent = categories_df_raw[
                categories_df_raw["game"] == parent_game_id
            ]

            if not category_row_for_parent.empty:
                # Take the first match if multiple (should be unique by parent_game_id in categories_df_raw)
                data_to_add = category_row_for_parent.iloc[0].copy().to_dict()
                data_to_add["game"] = (
                    actual_game_id  # Set the 'game' column to the actual passed game ID
                )
                result_for_model_list.append(data_to_add)
            # else:
            # print(f"Warning: Parent task '{parent_game_id}' for game '{actual_game_id}' not found in {categories_csv_path}.")

        if result_for_model_list:
            output_df_for_model = pd.DataFrame(result_for_model_list)
            # Ensure correct column order: 'game' first, then other category columns
            # Filter out any columns that might have been picked up accidentally and are not in the original categories file
            final_columns = ["game"] + [
                col for col in category_columns if col in output_df_for_model.columns
            ]
            output_df_for_model = output_df_for_model[final_columns]

            output_filename = os.path.join(
                per_model_output_dir, f"{model_name}_category_contributions.csv"
            )
            try:
                output_df_for_model.to_csv(output_filename, index=False, na_rep="")
                print(f"Generated {output_filename}")
            except Exception as e:
                print(f"Error saving {output_filename}: {e}")
        # else:
        # print(f"No category contribution data to save for model {model_name}.")


def save_claude_comparison_difficulty_csv_and_plot(
    df: pd.DataFrame,
    output_dir: str,
    model_base_name: str,
    model_compare_name: str,
):
    """
    Compares two specific models based on success rate per difficulty.
    Normalizes model_base_name's score to 1 and reports model_compare_name's score relative to it.
    Saves a CSV and a bar plot.
    """
    relevant_models = [model_base_name, model_compare_name]
    df_filtered = df[df["model"].isin(relevant_models)].copy()

    if df_filtered.empty:
        print(
            f"No data found for models {model_base_name} or {model_compare_name}. Skipping difficulty comparison."
        )
        return

    # Calculate success rate per model and difficulty
    difficulty_summary = (
        df_filtered.groupby(["model", "difficulty"], observed=False)["passed"]
        .mean()
        .reset_index()
    )
    difficulty_summary.rename(columns={"passed": "success_rate"}, inplace=True)

    # Pivot to get model scores side-by-side
    pivot_df = difficulty_summary.pivot(
        index="difficulty", columns="model", values="success_rate"
    )

    # Ensure both models are columns, fill with NaN if one is missing for a difficulty
    if model_base_name not in pivot_df.columns:
        pivot_df[model_base_name] = np.nan
    if model_compare_name not in pivot_df.columns:
        pivot_df[model_compare_name] = np.nan

    pivot_df = pivot_df.rename(
        columns={
            model_base_name: "model_base_score",
            model_compare_name: "model_compare_score",
        }
    )

    # Normalize: compare_score / base_score
    # Handle division by zero or NaN in base_score
    pivot_df["normalized_compare_score"] = (
        pivot_df["model_compare_score"] / pivot_df["model_base_score"]
    )
    pivot_df.replace(
        [np.inf, -np.inf], np.nan, inplace=True
    )  # Handle true division by zero

    # Order difficulties
    difficulty_order = ["easy", "base", "hard"]
    pivot_df = pivot_df.reindex(difficulty_order)
    pivot_df.index.name = "difficulty"  # Set index name for reset_index()
    csv_data = pivot_df.reset_index()

    # Save CSV
    csv_filename = (
        f"{model_base_name}_vs_{model_compare_name}_difficulty_comparison.csv"
    )
    csv_filepath = os.path.join(output_dir, csv_filename)
    try:
        csv_data.to_csv(csv_filepath, index=False)
        print(f"Saved Claude difficulty comparison to {csv_filepath}")
    except Exception as e:
        print(f"Error saving Claude difficulty comparison CSV: {e}")

    # Plotting
    plot_data = csv_data.dropna(subset=["normalized_compare_score"])
    if not plot_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            y="difficulty",
            x="normalized_compare_score",
            data=plot_data,
            color=MODEL_PALETTE.get(model_compare_name, "skyblue"),
            order=difficulty_order,  # Order for y-axis
            legend=False,  # No hue, so no hue legend needed
        )
        plt.axvline(
            x=1.0,
            color="r",
            linestyle="--",
            label=f"{model_base_name} (Normalized to 1.0)",
        )
        # plt.title removed
        plt.ylabel("Difficulty", fontsize=18, weight="bold")  # Swapped
        plt.xlabel(  # Swapped
            f"Computer-use relative success rate\n({model_base_name}=1.0)",
            fontsize=18,
            weight="bold",
        )
        plt.yticks(fontsize=16)  # Swapped
        plt.xticks(fontsize=16)  # Swapped
        plt.legend(
            fontsize=14,
            loc="lower right",  # Adjusted legend position
        )
        plt.grid(axis="x", visible=False)  # Swapped from y
        plt.grid(axis="y", visible=False)  # Swapped from x
        plt.tight_layout()
        plot_filename = (
            f"{model_base_name}_vs_{model_compare_name}_difficulty_comparison_plot"
        )
        save_plot(plt, output_dir, plot_filename)
        plt.close()
    else:
        print(
            f"No data to plot for {model_base_name} vs {model_compare_name} by difficulty."
        )


def save_claude_comparison_category_csv_and_plot(
    df: pd.DataFrame,
    output_dir: str,
    model_base_name: str,
    model_compare_name: str,
):
    """
    Compares two specific models based on weighted success rate per category.
    Normalizes model_base_name's score to 1 and reports model_compare_name's score relative to it.
    Saves a CSV and a bar plot.
    """
    relevant_models = [model_base_name, model_compare_name]
    df_filtered = df[df["model"].isin(relevant_models)].copy()

    if df_filtered.empty:
        print(
            f"No data found for models {model_base_name} or {model_compare_name}. Skipping category comparison."
        )
        return

    df_filtered["passed_numeric"] = df_filtered["passed"].astype(int)
    df_filtered["weighted_passed_score"] = (
        df_filtered["passed_numeric"] * df_filtered["weight"]
    )

    category_performance = (
        df_filtered.groupby(["model", "category"], observed=False)
        .agg(
            total_weighted_passed=("weighted_passed_score", "sum"),
            total_weight=("weight", "sum"),
        )
        .reset_index()
    )

    category_performance["weighted_success_rate"] = (
        category_performance["total_weighted_passed"]
        / category_performance["total_weight"]
    )
    # Handle cases where total_weight might be 0, leading to NaN or Inf
    category_performance["weighted_success_rate"] = category_performance[
        "weighted_success_rate"
    ].replace([np.inf, -np.inf], np.nan)
    category_performance["weighted_success_rate"] = category_performance[
        "weighted_success_rate"
    ].fillna(0)  # If NaN from 0/0, treat as 0 success

    # Pivot to get model scores side-by-side
    pivot_df = category_performance.pivot(
        index="category", columns="model", values="weighted_success_rate"
    )

    if model_base_name not in pivot_df.columns:
        pivot_df[model_base_name] = np.nan
    if model_compare_name not in pivot_df.columns:
        pivot_df[model_compare_name] = np.nan

    pivot_df = pivot_df.rename(
        columns={
            model_base_name: "model_base_score",
            model_compare_name: "model_compare_score",
        }
    )

    # Filter out 'Uncategorized' before normalization and plotting
    if "Uncategorized" in pivot_df.index:
        pivot_df = pivot_df.drop("Uncategorized")

    # Normalize: compare_score / base_score
    pivot_df["normalized_compare_score"] = (
        pivot_df["model_compare_score"] / pivot_df["model_base_score"]
    )
    pivot_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    pivot_df.index.name = "category"  # Set index name
    csv_data = pivot_df.reset_index().sort_values("category")

    # Save CSV
    csv_filename = f"{model_base_name}_vs_{model_compare_name}_category_comparison.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    try:
        csv_data.to_csv(csv_filepath, index=False)
        print(f"Saved Claude category comparison to {csv_filepath}")
    except Exception as e:
        print(f"Error saving Claude category comparison CSV: {e}")

    # Plotting
    plot_data = csv_data.dropna(subset=["normalized_compare_score"]).sort_values(
        "normalized_compare_score", ascending=False
    )
    if not plot_data.empty:
        plt.figure(figsize=(14, max(8, len(plot_data) * 0.4)))  # Adjusted height
        sns.barplot(
            x="normalized_compare_score",
            y="category",
            data=plot_data,
            hue="category",
            palette="mako",  # Use consistent palette
            legend=False,
        )
        plt.axvline(
            x=1.0,
            color="r",
            linestyle="--",
            label=f"{model_base_name} (Normalized to 1.0)",
        )
        # plt.title removed
        plt.xlabel(
            f"Computer-use relative weighted success rate\n({model_base_name}=1.0)",
            fontsize=18,
            weight="bold",
        )
        plt.ylabel("Task Category", fontsize=18, weight="bold")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(
            fontsize=16, loc="upper right", bbox_to_anchor=(1, 0.95)
        )  # Adjusted legend
        plt.grid(axis="x", visible=False)  # Removed linestyle and alpha
        plt.grid(axis="y", visible=False)
        plt.tight_layout()
        plot_filename = (
            f"{model_base_name}_vs_{model_compare_name}_category_comparison_plot"
        )
        save_plot(plt, output_dir, plot_filename)
        plt.close()
    else:
        print(
            f"No data to plot for {model_base_name} vs {model_compare_name} by category."
        )


def save_gemini_comparison_category_csv_and_plot(
    df: pd.DataFrame,
    output_dir: str,
    model_base_name: str,
    model_compare_name: str,
):
    """
    Compares two specific Gemini models based on weighted success rate per category.
    Normalizes model_base_name's score to 1 and reports model_compare_name's score relative to it.
    Saves a CSV and a bar plot.
    """
    relevant_models = [model_base_name, model_compare_name]
    df_filtered = df[df["model"].isin(relevant_models)].copy()

    if df_filtered.empty:
        print(
            f"No data found for models {model_base_name} or {model_compare_name}. Skipping category comparison."
        )
        return

    df_filtered["passed_numeric"] = df_filtered["passed"].astype(int)
    df_filtered["weighted_passed_score"] = (
        df_filtered["passed_numeric"] * df_filtered["weight"]
    )

    category_performance = (
        df_filtered.groupby(["model", "category"], observed=False)
        .agg(
            total_weighted_passed=("weighted_passed_score", "sum"),
            total_weight=("weight", "sum"),
        )
        .reset_index()
    )

    category_performance["weighted_success_rate"] = (
        category_performance["total_weighted_passed"]
        / category_performance["total_weight"]
    )
    # Handle cases where total_weight might be 0, leading to NaN or Inf
    category_performance["weighted_success_rate"] = category_performance[
        "weighted_success_rate"
    ].replace([np.inf, -np.inf], np.nan)
    category_performance["weighted_success_rate"] = category_performance[
        "weighted_success_rate"
    ].fillna(0)  # If NaN from 0/0, treat as 0 success

    # Pivot to get model scores side-by-side
    pivot_df = category_performance.pivot(
        index="category", columns="model", values="weighted_success_rate"
    )

    if model_base_name not in pivot_df.columns:
        pivot_df[model_base_name] = np.nan
    if model_compare_name not in pivot_df.columns:
        pivot_df[model_compare_name] = np.nan

    pivot_df = pivot_df.rename(
        columns={
            model_base_name: "model_base_score",
            model_compare_name: "model_compare_score",
        }
    )

    # Filter out 'Uncategorized' before normalization and plotting
    if "Uncategorized" in pivot_df.index:
        pivot_df = pivot_df.drop("Uncategorized")

    # Normalize: compare_score / base_score
    pivot_df["normalized_compare_score"] = (
        pivot_df["model_compare_score"] / pivot_df["model_base_score"]
    )
    pivot_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    pivot_df.index.name = "category"  # Set index name
    csv_data = pivot_df.reset_index().sort_values("category")

    # Save CSV
    csv_filename = f"{model_base_name}_vs_{model_compare_name}_category_comparison.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    try:
        csv_data.to_csv(csv_filepath, index=False)
        print(f"Saved Gemini category comparison to {csv_filepath}")
    except Exception as e:
        print(f"Error saving Gemini category comparison CSV: {e}")

    # Plotting
    plot_data = csv_data.dropna(subset=["normalized_compare_score"]).sort_values(
        "normalized_compare_score", ascending=False
    )
    if not plot_data.empty:
        plt.figure(figsize=(14, max(8, len(plot_data) * 0.4)))  # Adjusted height
        sns.barplot(
            x="normalized_compare_score",
            y="category",
            data=plot_data,
            hue="category",
            palette="mako",  # Use consistent palette
            legend=False,
        )
        plt.axvline(
            x=1.0,
            color="r",
            linestyle="--",
            label=f"{model_base_name} (Normalized to 1.0)",
        )
        # plt.title removed
        plt.xlabel(
            f"Text-only relative weighted success rate\n({model_base_name}=1.0)",
            fontsize=18,
            weight="bold",
        )
        plt.ylabel("Task Category", fontsize=18, weight="bold")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(
            fontsize=16, loc="upper right", bbox_to_anchor=(1, 0.95)
        )  # Adjusted legend
        plt.grid(axis="x", visible=False)  # Removed linestyle and alpha
        plt.grid(axis="y", visible=False)
        plt.tight_layout()
        plot_filename = (
            f"{model_base_name}_vs_{model_compare_name}_category_comparison_plot"
        )
        save_plot(plt, output_dir, plot_filename)
        plt.close()
    else:
        print(
            f"No data to plot for {model_base_name} vs {model_compare_name} by category."
        )


def save_gemini_comparison_difficulty_csv_and_plot(
    df: pd.DataFrame,
    output_dir: str,
    model_base_name: str,
    model_compare_name: str,
):
    """
    Compares two specific Gemini models based on success rate per difficulty.
    Normalizes model_base_name's score to 1 and reports model_compare_name's score relative to it.
    Saves a CSV and a bar plot.
    """
    relevant_models = [model_base_name, model_compare_name]
    df_filtered = df[df["model"].isin(relevant_models)].copy()

    if df_filtered.empty:
        print(
            f"No data found for models {model_base_name} or {model_compare_name}. Skipping difficulty comparison."
        )
        return

    # Calculate success rate per model and difficulty
    difficulty_summary = (
        df_filtered.groupby(["model", "difficulty"], observed=False)["passed"]
        .mean()
        .reset_index()
    )
    difficulty_summary.rename(columns={"passed": "success_rate"}, inplace=True)

    # Pivot to get model scores side-by-side
    pivot_df = difficulty_summary.pivot(
        index="difficulty", columns="model", values="success_rate"
    )

    # Ensure both models are columns, fill with NaN if one is missing for a difficulty
    if model_base_name not in pivot_df.columns:
        pivot_df[model_base_name] = np.nan
    if model_compare_name not in pivot_df.columns:
        pivot_df[model_compare_name] = np.nan

    pivot_df = pivot_df.rename(
        columns={
            model_base_name: "model_base_score",
            model_compare_name: "model_compare_score",
        }
    )

    # Normalize: compare_score / base_score
    # Handle division by zero or NaN in base_score
    pivot_df["normalized_compare_score"] = (
        pivot_df["model_compare_score"] / pivot_df["model_base_score"]
    )
    pivot_df.replace(
        [np.inf, -np.inf], np.nan, inplace=True
    )  # Handle true division by zero

    # Order difficulties
    difficulty_order = ["easy", "base", "hard"]
    pivot_df = pivot_df.reindex(difficulty_order)
    pivot_df.index.name = "difficulty"  # Set index name for reset_index()
    csv_data = pivot_df.reset_index()

    # Save CSV
    csv_filename = (
        f"{model_base_name}_vs_{model_compare_name}_difficulty_comparison.csv"
    )
    csv_filepath = os.path.join(output_dir, csv_filename)
    try:
        csv_data.to_csv(csv_filepath, index=False)
        print(f"Saved Gemini difficulty comparison to {csv_filepath}")
    except Exception as e:
        print(f"Error saving Gemini difficulty comparison CSV: {e}")

    # Plotting
    plot_data = csv_data.dropna(subset=["normalized_compare_score"])
    if not plot_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            y="difficulty",
            x="normalized_compare_score",
            data=plot_data,
            color=MODEL_PALETTE.get(model_compare_name, "lightgreen"),
            order=difficulty_order,  # Order for y-axis
            legend=False,  # No hue
        )
        plt.axvline(
            x=1.0,
            color="r",
            linestyle="--",
            label=f"{model_base_name} (Normalized to 1.0)",
        )
        # plt.title removed
        plt.ylabel("Difficulty", fontsize=18, weight="bold")  # Swapped
        plt.xlabel(  # Swapped
            f"Text-only relative success rate\n({model_base_name}=1.0)",
            fontsize=18,
            weight="bold",
        )
        plt.yticks(fontsize=16)  # Swapped
        plt.xticks(fontsize=16)  # Swapped
        plt.legend(
            fontsize=14,
            loc="lower right",  # Adjusted legend position
        )
        plt.grid(axis="x", visible=False)  # Swapped from y
        plt.grid(axis="y", visible=False)  # Swapped from x
        plt.tight_layout()
        plot_filename = (
            f"{model_base_name}_vs_{model_compare_name}_difficulty_comparison_plot"
        )
        save_plot(plt, output_dir, plot_filename)
        plt.close()
    else:
        print(
            f"No data to plot for {model_base_name} vs {model_compare_name} by difficulty."
        )


def main():
    # Set a nicer plot style
    sns.set_theme(style="whitegrid")

    # Create output directory if it doesn't exist
    output_dir = "outputs2"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Load and preprocess data
    processed_data = load_and_preprocess_data()

    if processed_data.empty:
        print("No data processed. Cannot generate plots.")
        return

    # Generate model-category performance table
    save_11_model_category_performance_csv_and_tex(processed_data, output_dir)

    # List unsolved tasks
    list_unsolved_tasks(processed_data, output_dir)

    # List tasks solved by exactly one model
    list_tasks_solved_by_one_model(processed_data, output_dir)

    # List unsolved task families
    list_unsolved_task_families(processed_data, output_dir)

    # Save average success rate per difficulty
    save_02_average_success_per_difficulty_globally_csv(processed_data, output_dir)
    save_03_success_vs_difficulty_per_model_csv(processed_data, output_dir, MODEL_ORDER)
    save_04_success_rate_per_category_csv(processed_data, output_dir)
    save_05_success_rate_per_difficulty_per_category_csv(processed_data, output_dir)
    save_06_weighted_unsolved_tasks_by_category_csv(processed_data, output_dir)
    save_07_model_performance_radar_data_csv(processed_data, output_dir)

    # List generation phase
    save_08_tasks_unsolved_by_any_model_csv(processed_data, output_dir)
    save_09_tasks_solved_by_exactly_one_model_csv(processed_data, output_dir)
    save_10_parent_tasks_unsolved_at_any_difficulty_csv(processed_data, output_dir)

    # Save overall success rate per model
    save_01_overall_success_per_model_csv(processed_data, output_dir, MODEL_ORDER)

    # Generate the new game_id vs model pass status CSV
    save_00_game_id_model_pass_status_csv(processed_data, output_dir, MODEL_ORDER)

    # Generate per-model category contributions for passed tasks
    save_per_model_category_contributions(
        processed_data, output_dir, CATEGORIES_CSV_PATH, MODEL_ORDER
    )

    # --- Plotting Phase (will read from CSVs) ---
    plot_01_overall_success_per_model_chart(output_dir, MODEL_ORDER)
    plot_03_success_vs_difficulty_per_model_chart(output_dir, MODEL_ORDER)
    plot_00_game_id_model_pass_status_heatmap(output_dir, MODEL_ORDER)
    plot_04_success_rate_per_category_chart(output_dir)
    plot_05_success_rate_per_difficulty_per_category_chart(output_dir)
    plot_06_weighted_unsolved_tasks_by_category_chart(output_dir)
    plot_07_model_performance_radar_chart(output_dir)

    print(
        f"\nAnalysis complete. Plots and lists saved to the '{output_dir}' directory."
    )

    # Specific Claude comparisons
    model_claude_base = "claude-3-7-sonnet-20250219"
    model_claude_computeruse = "claude-3-7-sonnet-20250219-computeruse"

    save_claude_comparison_difficulty_csv_and_plot(
        processed_data, output_dir, model_claude_base, model_claude_computeruse
    )
    save_claude_comparison_category_csv_and_plot(
        processed_data, output_dir, model_claude_base, model_claude_computeruse
    )

    print(
        f"\nAnalysis complete. Plots and lists saved to the '{output_dir}' directory."
    )

    # Specific Gemini comparisons
    model_gemini_base = "gemini-2.5-pro-preview-05-06"
    model_gemini_textonly = "gemini-2.5-pro-preview-05-06-textonly"

    save_gemini_comparison_category_csv_and_plot(
        processed_data, output_dir, model_gemini_base, model_gemini_textonly
    )
    save_gemini_comparison_difficulty_csv_and_plot(
        processed_data, output_dir, model_gemini_base, model_gemini_textonly
    )

    print(
        f"\nAnalysis complete. Plots and lists saved to the '{output_dir}' directory."
    )


if __name__ == "__main__":
    main()
