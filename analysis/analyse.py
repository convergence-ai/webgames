import dataclasses
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
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-04-17",
    "gpt-4o",
    "gpt-4o-mini",
    "qwen2.5-vl-72b-instruct",
    "qwen2.5-vl-32b-instruct",
    "qwen2.5-vl-7b-instruct",
]

# Define the model color palette for consistent plotting
# Keys must match model names in DataFrame (lowercase, as processed)
MODEL_PALETTE = {
    # Claude (Oranges/Browns from YlOrBr palette)
    "claude-3-7-sonnet-20250219": sns.color_palette("YlOrBr", 3)[1],  # Mid-shade
    # Gemini (Greens from Greens palette)
    "gemini-2.5-pro-preview-05-06": sns.color_palette("Greens", 3)[
        2
    ],  # Darker shade for Pro
    "gemini-2.5-flash-preview-04-17": sns.color_palette("Greens", 3)[
        1
    ],  # Lighter shade for Flash
    # GPT (Blues from Blues palette)
    "gpt-4o": sns.color_palette("Blues", 3)[2],  # Darker shade for 4o
    "gpt-4o-mini": sns.color_palette("Blues", 3)[1],  # Lighter shade for 4o-mini
    # Qwen (Purples from Purples palette)
    "qwen2.5-vl-72b-instruct": sns.color_palette("Purples", 4)[
        3
    ],  # Darkest shade for 72b
    "qwen2.5-vl-32b-instruct": sns.color_palette("Purples", 4)[
        2
    ],  # Mid-dark shade for 32b
    "qwen2.5-vl-7b-instruct": sns.color_palette("Purples", 4)[
        1
    ],  # Mid-light shade for 7b
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
    # or derive it from a numeric score column like `score_value`.
    # For this example, we'll try to find a column like 'score_passed'.
    # If not found, we will try to use 'score_accuracy' == 1 as passed.

    # Check for 'metadata.variant' and 'metadata.base_task' and rename if they have prefixes
    if "metadata.variant" in df_combined.columns:
        df_combined.rename(columns={"metadata.variant": "difficulty"}, inplace=True)
    elif "metadata_variant" in df_combined.columns:  # Common pattern from inspect_ai
        df_combined.rename(columns={"metadata_variant": "difficulty"}, inplace=True)
    else:
        print(
            "Warning: 'difficulty' column (metadata.variant or metadata_variant) not found. Assigning 'base' as default."
        )
        df_combined["difficulty"] = "base"

    if "metadata.base_task" in df_combined.columns:
        df_combined.rename(columns={"metadata.base_task": "game"}, inplace=True)
    elif "metadata_base_task" in df_combined.columns:
        df_combined.rename(columns={"metadata_base_task": "game"}, inplace=True)
    else:
        print(
            "Warning: 'game' column (metadata.base_task or metadata_base_task) not found."
        )
        # Cannot proceed without game for category merge
        return pd.DataFrame()

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
    df_categories_melted = df_categories_melted[df_categories_melted["weight"] > 0]

    # Merge with main data
    df_merged = pd.merge(df_combined, df_categories_melted, on="game", how="left")
    df_merged["category"] = df_merged["category"].fillna("Uncategorized")
    df_merged["weight"] = df_merged["weight"].fillna(0)

    # Save the processed data to cache
    try:
        with open(CACHE_FILE_PATH, "wb") as f:
            pickle.dump(df_merged, f)
            print(f"Saved processed data to cache: {CACHE_FILE_PATH}")
    except Exception as e:
        print(f"Error saving cache file: {e}")

    return df_merged


def save_plot(plt, output_dir: str, base_filename: str, dpi: int = 300):
    """
    Save a plot in both PDF and PNG formats.

    Args:
        plt: The matplotlib pyplot object
        output_dir: Directory to save the files in
        base_filename: Base filename without extension
        dpi: DPI for the PNG output (default: 300)
    """
    # Remove any existing extension from base_filename
    base_filename = os.path.splitext(base_filename)[0]

    # Save PDF
    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    plt.savefig(pdf_path)

    # Save PNG with high DPI
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_path, dpi=dpi)

    print(f"Generated {base_filename}.pdf and {base_filename}.png in {output_dir}")


def plot_success_vs_difficulty_per_model(df, output_dir):
    """Plots success rate vs. difficulty per model."""
    if (
        "difficulty" not in df.columns
        or "passed" not in df.columns
        or "model" not in df.columns
    ):
        print(
            "Skipping plot_success_vs_difficulty_per_model: Required columns missing."
        )
        return

    # Define a custom order for difficulty
    difficulty_order = ["easy", "base", "hard"]

    # Calculate success rate
    summary_df = df.groupby(["model", "difficulty"])["passed"].mean().reset_index()

    # Filter out 'unknown' difficulty before setting the categorical type
    summary_df = summary_df[summary_df["difficulty"] != "unknown"]

    summary_df["difficulty"] = pd.Categorical(
        summary_df["difficulty"], categories=difficulty_order, ordered=True
    )
    summary_df = summary_df.sort_values(["model", "difficulty"])

    # Get a sorted list of models for hue_order
    model_order = [model for model in MODEL_ORDER if model in df["model"].unique()]

    plt.figure(figsize=(12, 7))
    sns.barplot(
        x="difficulty",
        y="passed",
        hue="model",
        data=summary_df,
        palette=MODEL_PALETTE,
        hue_order=model_order,
    )
    # plt.title("Success Rate vs. Difficulty per Model")
    plt.ylabel("Success Rate (0.0 - 1.0)", fontsize=18, weight="bold")
    plt.xlabel("Difficulty", fontsize=18, weight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(axis="y", visible=False)  # Remove horizontal grid lines
    plt.tight_layout()
    save_plot(plt, output_dir, "success_vs_difficulty_per_model")
    plt.close()


def plot_success_rate_per_category(df, output_dir):
    """Plots overall success rate per category, weighted."""
    if (
        "category" not in df.columns
        or "passed" not in df.columns
        or "weight" not in df.columns
    ):
        print("Skipping plot_success_rate_per_category: Required columns missing.")
        return

    # Calculate weighted success for each category
    # For each sample, its contribution to category success is sample.passed * sample.weight
    # Total weighted success = sum(passed * weight)
    # Total weight = sum(weight)
    # Success rate = Total weighted success / Total weight

    # Ensure 'passed' is numeric (0 or 1) for multiplication
    df["passed_numeric"] = df["passed"].astype(int)
    df["weighted_passed_score"] = df["passed_numeric"] * df["weight"]

    category_summary = (
        df.groupby("category")
        .agg(
            total_weighted_passed_score=("weighted_passed_score", "sum"),
            total_weight=("weight", "sum"),
        )
        .reset_index()
    )

    category_summary["success_rate"] = (
        category_summary["total_weighted_passed_score"]
        / category_summary["total_weight"]
    )
    category_summary = category_summary[
        category_summary["total_weight"] > 0
    ]  # Avoid division by zero and irrelevant categories
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
    plt.title("Overall Weighted Success Rate per Game Category")
    plt.xlabel("Weighted Success Rate (0.0 - 1.0)")
    plt.ylabel("Game Category")
    plt.tight_layout()
    save_plot(plt, output_dir, "success_rate_per_category")
    plt.close()


def plot_overall_success_per_model(df, output_dir):
    """Plots overall success rate per model."""
    if "passed" not in df.columns or "model" not in df.columns:
        print("Skipping plot_overall_success_per_model: Required columns missing.")
        return

    model_summary = df.groupby("model")["passed"].mean().reset_index()
    # Sort models based on the predefined MODEL_ORDER
    model_summary["model"] = pd.Categorical(
        model_summary["model"], categories=MODEL_ORDER, ordered=True
    )
    model_summary = model_summary.sort_values("model")

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="passed",
        y="model",
        data=model_summary,
        hue="model",
        palette=MODEL_PALETTE,
        legend=False,
        order=model_summary["model"].tolist(),  # Uses the new sorted order
    )
    # plt.title("Overall Success Rate per Model", fontsize=18, weight="bold")
    plt.xlabel("Success Rate (0.0 - 1.0)", fontsize=18, weight="bold")
    plt.ylabel("Model", fontsize=18, weight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis="y", visible=False)  # Remove horizontal grid lines
    plt.grid(axis="x", visible=False)  # Add vertical grid lines
    plt.tight_layout()
    save_plot(plt, output_dir, "overall_success_per_model")
    plt.close()


def save_overall_success_per_model_csv(df: pd.DataFrame, output_dir: str):
    """Calculates and saves the overall success rate per model to a CSV file."""
    if "passed" not in df.columns or "model" not in df.columns:
        print(
            "Skipping save_overall_success_per_model_csv: Required columns ('passed', 'model') missing."
        )
        return

    if df.empty:
        print("Skipping save_overall_success_per_model_csv: Input DataFrame is empty.")
        return

    model_summary = df.groupby("model")["passed"].mean().reset_index()
    # Sort models based on the predefined MODEL_ORDER
    model_summary["model"] = pd.Categorical(
        model_summary["model"], categories=MODEL_ORDER, ordered=True
    )
    model_summary = model_summary.sort_values(
        "model"
    )  # Sort by model order, not success rate
    model_summary.rename(columns={"passed": "overall_success_rate"}, inplace=True)

    if model_summary.empty:
        print("No data to save for overall success rate per model.")
        return

    output_file_path = os.path.join(output_dir, "overall_success_per_model.csv")
    try:
        model_summary.to_csv(output_file_path, index=False)
        print(f"Overall success rate per model saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving overall success rate per model to CSV: {e}")


def plot_success_rate_per_difficulty_per_category(df, output_dir):
    """Plots success rate per difficulty per category."""
    if (
        "category" not in df.columns
        or "difficulty" not in df.columns
        or "passed" not in df.columns
    ):
        print(
            "Skipping plot_success_rate_per_difficulty_per_category: Required columns missing."
        )
        return

    difficulty_order = ["easy", "base", "hard"]

    # Filter out 'Uncategorized' category and 'unknown' difficulty for this specific plot
    # Also ensure 'passed' is numeric for catplot's default estimator (mean)
    plot_df = df[
        (df["category"] != "Uncategorized") & (df["difficulty"] != "unknown")
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    if plot_df.empty:
        print(
            "No data available for plotting success rate per difficulty per category after filtering."
        )
        return

    plot_df["passed"] = plot_df["passed"].astype(
        float
    )  # Ensure 'passed' is float for mean calculation
    plot_df["difficulty"] = pd.Categorical(
        plot_df["difficulty"], categories=difficulty_order, ordered=True
    )
    # Sort values to ensure consistent plot order if not inherently handled by catplot grouping
    plot_df = plot_df.sort_values(["category", "difficulty"])

    if plot_df.empty:
        print("No data to plot after filtering for difficulty and category.")
        return

    # Use catplot for faceted bar plots
    g = sns.catplot(
        x="difficulty",
        y="passed",  # catplot will calculate the mean of 'passed'
        hue="difficulty",  # Assign x to hue
        col="category",
        data=plot_df,
        kind="bar",
        col_wrap=3,  # Adjust number of columns for subplots
        palette="viridis",
        height=4,
        aspect=1.2,
        sharey=True,  # Keep y-axis consistent for comparison
        legend=False,  # Set legend to False
    )
    g.set_axis_labels("Difficulty", "Success Rate (0.0 - 1.0)")
    g.set_titles("{col_name}")  # Sets each subplot's title to the category name
    g.fig.suptitle(
        "Success Rate per Difficulty per Category", y=1.03, fontsize=16
    )  # Overall title
    plt.tight_layout(rect=(0, 0, 1, 0.97))  # Adjust rect to make space for suptitle

    save_plot(plt, output_dir, "success_rate_per_difficulty_per_category")
    plt.close(g.fig)  # Close the figure associated with catplot


def plot_weighted_unsolved_tasks_by_category(df: pd.DataFrame, output_dir: str):
    """Plots the weighted count of completely unsolved tasks by category."""
    required_cols = ["id", "passed", "category", "weight"]
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(
            f"Skipping plot_weighted_unsolved_tasks_by_category: Required columns missing: {missing_cols}."
        )
        return
    if df.empty:
        print(
            "Skipping plot_weighted_unsolved_tasks_by_category: Input DataFrame is empty."
        )
        return

    # Identify task 'id's that were not solved by any model
    solved_by_any_model = df.groupby("id")["passed"].any()
    unsolved_task_ids = solved_by_any_model[~solved_by_any_model].index

    if unsolved_task_ids.empty:
        print("No completely unsolved tasks found. Skipping plot.")
        return

    # Filter the DataFrame for details of these unsolved tasks
    unsolved_df_details = df[df["id"].isin(unsolved_task_ids)]

    if unsolved_df_details.empty:
        # This case should ideally not be reached if unsolved_task_ids is not empty,
        # but as a safeguard:
        print("No details found for unsolved tasks. Skipping plot.")
        return

    # For these unsolved tasks, get their unique (id, category, weight) contributions.
    # This ensures that each unsolved task contributes its weight to a category only once,
    # regardless of how many models failed it.
    unique_unsolved_task_category_weights = unsolved_df_details[
        ["id", "category", "weight"]
    ].drop_duplicates()

    # Sum these weights by category
    category_unsolved_summary = (
        unique_unsolved_task_category_weights.groupby("category")["weight"]
        .sum()
        .reset_index()
    )
    category_unsolved_summary = category_unsolved_summary.rename(
        columns={"weight": "total_weighted_unsolved_tasks"}
    )

    # Filter out categories with zero total weighted unsolved tasks
    category_unsolved_summary = category_unsolved_summary[
        category_unsolved_summary["total_weighted_unsolved_tasks"] > 0
    ]
    category_unsolved_summary = category_unsolved_summary.sort_values(
        "total_weighted_unsolved_tasks", ascending=False
    )

    if category_unsolved_summary.empty:
        print("No weighted unsolved tasks to plot by category after processing.")
        return

    plt.figure(figsize=(14, 8))
    sns.barplot(
        x="total_weighted_unsolved_tasks",
        y="category",
        data=category_unsolved_summary,
        hue="category",  # Assigns a different color to each category bar
        palette="viridis",  # You can choose any suitable palette
        legend=False,  # Legend is not necessary as y-axis labels categories
    )
    plt.title("Weighted Count of Completely Unsolved Tasks by Category")
    plt.xlabel("Total Weighted Unsolved Tasks")
    plt.ylabel("Game Category")
    plt.tight_layout()
    save_plot(plt, output_dir, "weighted_unsolved_tasks_by_category")
    plt.close()


def plot_model_performance_radar(df: pd.DataFrame, output_dir: str):
    """Plots a radar chart of normalized model performance per category."""
    required_cols = ["id", "model", "category", "weight", "passed"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Skipping radar plot: Missing required columns: {missing}")
        return
    if df.empty:
        print("Skipping radar plot: DataFrame is empty.")
        return

    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Ensure 'weighted_passed_score' column exists
    if "weighted_passed_score" not in df_copy.columns:
        if "passed" not in df_copy.columns or "weight" not in df_copy.columns:
            print(
                "Skipping radar plot: Cannot create 'weighted_passed_score' due to missing 'passed' or 'weight'."
            )
            return
        df_copy["passed_numeric"] = df_copy["passed"].astype(int)
        df_copy["weighted_passed_score"] = df_copy["passed_numeric"] * df_copy["weight"]

    # 1. Performance per model per category
    model_category_performance = (
        df_copy.groupby(["model", "category"])["weighted_passed_score"]
        .sum()
        .reset_index()
    )

    # 2. Total available score per category
    # Sum of weights of unique tasks (id) within each category
    unique_task_category_weights = df_copy[
        ["id", "category", "weight"]
    ].drop_duplicates()
    total_score_per_cat_df = (
        unique_task_category_weights.groupby("category")["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "total_possible_score"})
    )

    # Filter out categories where total_possible_score is not positive
    total_score_per_cat_df = total_score_per_cat_df[
        total_score_per_cat_df["total_possible_score"] > 0
    ]

    if total_score_per_cat_df.empty:
        print(
            "Skipping radar plot: No categories with positive total possible scores found."
        )
        return

    # 3. Merge and calculate normalized performance
    merged_data = pd.merge(
        model_category_performance,
        total_score_per_cat_df,
        on="category",
        how="inner",  # Use inner merge to keep only categories with a total possible score
    )

    if merged_data.empty:
        print(
            "Skipping radar plot: No data after merging model performance with category scores."
        )
        return

    merged_data["normalized_performance"] = (
        merged_data["weighted_passed_score"] / merged_data["total_possible_score"]
    )
    # Clip values to ensure they are between 0 and 1
    merged_data["normalized_performance"] = merged_data["normalized_performance"].clip(
        0, 1
    )

    # 4. Pivot table for plotting
    pivot_df = merged_data.pivot(
        index="model", columns="category", values="normalized_performance"
    ).fillna(0)  # Fill NaN with 0 for models not in certain categories after merge

    # Sort pivot_df by model name (index)
    pivot_df = pivot_df.sort_index(ascending=True)

    if pivot_df.empty:
        print("Skipping radar plot: Pivoted data is empty.")
        return

    categories = pivot_df.columns.tolist()
    if not categories:
        print("Skipping radar plot: No categories to plot.")
        return

    num_vars = len(categories)

    # Angles for radar plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    closed_angles = angles + angles[:1]  # Close the loop for plotting

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    # Determine the maximum value in the pivot_df for dynamic scaling
    # Ensure that max_val is at least a small positive number if all values are 0
    max_val = pivot_df.max().max()
    if (
        max_val <= 0
    ):  # If all performances are 0 or negative (though normalized should be >=0)
        max_val = 0.2  # Default to a small range like 0-0.2 to make the plot visible
    else:
        # Add a small buffer to the max_val so the plot doesn't touch the edge,
        # and round up to the nearest sensible tick interval (e.g. 0.1 or 0.2)
        if max_val <= 0.5:
            max_val = np.ceil(max_val * 10) / 10  # round up to nearest 0.1
        elif (
            max_val < 1.0
        ):  # for values between 0.5 and 1.0, round up to nearest 0.1 or 0.2
            max_val = np.ceil(max_val / 0.2) * 0.2  # round up to nearest 0.2
        else:  # for max_val >= 1 (e.g. if clipping was removed)
            max_val = np.ceil(max_val)  # round up to nearest integer

    # Plot each model
    for model_name in pivot_df.index:
        values = pivot_df.loc[model_name].tolist()
        closed_values = values + values[:1]  # Close the loop for plotting

        color = MODEL_PALETTE.get(str(model_name))
        if color is None:
            print(
                f"Warning: Color not found for model {model_name} in radar plot. Using default gray."
            )
            color = "#808080"  # Default gray

        ax.plot(
            closed_angles,
            closed_values,
            linewidth=1.5,
            label=str(model_name),
            color=color,
        )
        ax.fill(closed_angles, closed_values, alpha=0.2, color=color)

    # Formatting
    ax.set_xticks(angles)  # Set angle ticks to original (non-closed) angles
    ax.set_xticklabels(categories, fontsize=16, fontweight="bold")
    # Dynamic y-ticks based on max_val
    tick_step = max_val / 5
    # Sensible tick steps (e.g., 0.1, 0.2, 0.25, 0.5)
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
    ax.set_ylim(0, max_val)

    # Legend
    num_models = len(pivot_df.index)
    legend_ncol = max(1, num_models // 3 if num_models > 2 else num_models)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),  # Adjust position to be below the plot
        ncol=legend_ncol,
        fontsize=16,
    )

    # plt.title("Normalized Model Performance per Category (Radar Plot)", size=16, y=1.08) # Title removed
    # Adjust layout to make space for legend (rect might need adjustment if title space is reclaimed)
    fig.tight_layout(rect=(0, 0.05, 1, 1))  # Adjusted top boundary for removed title

    save_plot(plt, output_dir, "model_performance_radar")
    plt.close(fig)


def list_unsolved_tasks(df: pd.DataFrame, output_dir: str):
    """
    Identifies tasks that were not solved by any model and saves their IDs to a CSV file.
    A task is identified by its 'id'.
    """
    if "passed" not in df.columns or "id" not in df.columns:
        print("Skipping list_unsolved_tasks: 'passed' or 'id' column missing.")
        return

    if df.empty:
        print("Input DataFrame is empty. No unsolved tasks to list.")
        return

    # For each task 'id', check if it was solved by at least one model
    # Group by 'id' and check if 'any' of the 'passed' values are True for that group
    solved_by_any_model = df.groupby("id")["passed"].any()

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
    if not all(col in df.columns for col in ["id", "model", "passed"]):
        print(
            "Skipping list_tasks_solved_by_one_model: Required columns ('id', 'model', 'passed') missing."
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
    solved_counts = passed_tasks_df.groupby("id")["model"].nunique()

    # Get tasks solved by exactly one model
    tasks_solved_by_one_model_ids = solved_counts[solved_counts == 1].index

    if tasks_solved_by_one_model_ids.empty:
        print("No tasks were solved by exactly one model.")
        return

    # Get the details (model) for these tasks
    # We need to go back to passed_tasks_df and filter by these task ids
    result_df = (
        passed_tasks_df[passed_tasks_df["id"].isin(tasks_solved_by_one_model_ids)][
            ["id", "model"]
        ]
        .drop_duplicates()  # Should be unique by id already due to the filter, but good practice
        .rename(columns={"id": "task_id", "model": "solving_model"})
    )

    output_file_path = os.path.join(output_dir, "tasks_solved_by_exactly_one_model.csv")
    try:
        result_df.to_csv(output_file_path, index=False)
        print(f"List of tasks solved by exactly one model saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving tasks solved by one model to CSV: {e}")


def list_unsolved_task_families(df: pd.DataFrame, output_dir: str):
    """
    Identifies task families (e.g., 'brick-buster' based on the 'game' column)
    that were not solved at any difficulty level and saves their names to a CSV.
    A task family is considered unsolved if no task instance belonging to that
    family (e.g., 'game-easy', 'game-hard', 'game-base') was passed.
    """
    required_cols = ["game", "passed"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(
            f"Skipping list_unsolved_task_families: Required columns missing: {missing}."
        )
        return

    if df.empty:
        print("Skipping list_unsolved_task_families: Input DataFrame is empty.")
        return

    # Determine if any task within each 'game' (family) was solved.
    # The 'game' column is assumed to represent the task family identifier.
    family_solved_status = df.groupby("game")["passed"].any()

    # Identify families where no task was solved (all entries for that family are False)
    unsolved_families_series = family_solved_status[~family_solved_status]

    if unsolved_families_series.empty:
        print(
            "All task families have at least one solved instance at some difficulty level. No unsolved families to list."
        )
        return

    unsolved_families_list = unsolved_families_series.index.tolist()
    unsolved_families_df = pd.DataFrame(
        unsolved_families_list, columns=["unsolved_task_family"]
    )

    output_file_path = os.path.join(
        output_dir, "task_families_unsolved_at_any_difficulty.csv"
    )
    try:
        unsolved_families_df.to_csv(output_file_path, index=False)
        print(
            f"List of task families unsolved at any difficulty saved to: {output_file_path}"
        )
    except Exception as e:
        print(f"Error saving list of unsolved task families to CSV: {e}")


def plot_category_cooccurrence(csv_path, output_dir):
    """
    Calculates and plots the co-occurrence of task categories.
    Co-occurrence is defined as the number of tasks (games) that
    fall under both categories (have a weight > 0).
    """
    try:
        df_categories_raw = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Categories file not found at {csv_path}")
        return

    if "game" not in df_categories_raw.columns:
        print("Error: 'game' column not found in categories CSV.")
        return

    df_categories = df_categories_raw.set_index("game")

    # Ensure all category columns are numeric, coercing errors and filling NaNs.
    # Explicitly cast to float after processing to aid type checking.
    for col in df_categories.columns:  # These are category names
        df_categories[col] = pd.to_numeric(df_categories[col], errors="coerce")
        df_categories[col] = df_categories[col].fillna(0.0)  # Fill NaNs before casting
        df_categories[col] = df_categories[col].astype(
            float
        )  # Explicitly cast to float

    category_names = df_categories.columns.tolist()
    if not category_names:
        print("No category columns found in the CSV (excluding 'game' column).")
        return

    cooccurrence_matrix = pd.DataFrame(
        0.0,
        index=category_names,
        columns=category_names,
        dtype=float,
    )

    for game_index in df_categories.index:
        present_categories = []
        for cat in category_names:
            # Use .at for scalar float access, value should be float due to prior .astype(float)
            value = df_categories.at[game_index, cat]
            try:
                # If value is float (guaranteed by .astype(float) and .at), direct comparison is fine.
                # pd.notna is still useful for safety, though fillna(0.0) should prevent NaNs here.
                if pd.notna(value) and value > 0.0:
                    present_categories.append(cat)
            except (
                ValueError,
                TypeError,
            ):  # TypeError might occur if value is not comparable to float
                # Handle cases where conversion to float might fail for unexpected non-numeric data
                print(
                    f"Warning: Could not convert value '{value}' to float for game '{game_index}', category '{cat}'. Skipping."
                )
                continue

        for cat1 in present_categories:
            for cat2 in present_categories:
                current_val = cooccurrence_matrix.at[
                    cat1, cat2
                ]  # Use .at for scalar access
                try:
                    # current_val is float due to dtype=float and .at access
                    cooccurrence_matrix.at[cat1, cat2] = (
                        current_val + 1.0
                    )  # Use .at for assignment
                except (ValueError, TypeError):
                    print(
                        f"Warning: Could not convert matrix value '{current_val}' to float for categories ('{cat1}', '{cat2}'). Skipping increment."
                    )
                    continue

    if cooccurrence_matrix.empty:
        print("No co-occurrence data to plot.")
        return

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cooccurrence_matrix.astype(int),
        annot=True,
        fmt="d",
        cmap="viridis",
        linewidths=0.5,  # Cast to int for display
    )
    plt.title(
        "Co-occurrence Matrix of Task Categories\\n(Number of tasks sharing categories)"
    )
    plt.xlabel("Category")
    plt.ylabel("Category")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_plot(plt, output_dir, "category_cooccurrence_heatmap")
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


def generate_game_model_pass_matrix_and_plot(df: pd.DataFrame, output_dir: str):
    """
    Generates a CSV and a heatmap plot showing which game variants (game + difficulty)
    were passed by which model.
    Rows: Game Variants, Columns: Models, Values: Boolean (passed or not).
    """
    # Add 'difficulty' to required_cols for the new 'game_variant'
    required_cols = ["game", "difficulty", "model", "passed"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(
            f"Skipping game-variant-model pass matrix: Missing required columns: {missing}"
        )
        return
    if df.empty:
        print("Skipping game-variant-model pass matrix: DataFrame is empty.")
        return

    # Create a copy to safely add the new column
    df_copy = df.copy()
    # Ensure 'game' and 'difficulty' are strings before concatenation to avoid mixed type issues
    df_copy["game"] = df_copy["game"].astype(str)
    df_copy["difficulty"] = df_copy["difficulty"].astype(str)
    df_copy["game_variant"] = df_copy["game"] + "_" + df_copy["difficulty"]

    # Pivot the table using 'game_variant'
    try:
        pass_matrix_df = df_copy.pivot_table(
            index="game_variant", columns="model", values="passed", fill_value=False
        )
        # Sort columns (model names) according to MODEL_ORDER
        ordered_models_in_data = [m for m in MODEL_ORDER if m in pass_matrix_df.columns]
        pass_matrix_df = pass_matrix_df.reindex(columns=ordered_models_in_data)
    except Exception as e:
        print(f"Error creating pivot table for game-variant-model pass matrix: {e}")
        return

    if pass_matrix_df.empty:
        print("Pivoted game-variant-model pass matrix is empty. Skipping CSV and plot.")
        return

    # Save to CSV with a new name
    csv_filename = "game_variant_model_pass_status_matrix.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    try:
        pass_matrix_df.to_csv(csv_filepath)
        print(f"Generated {csv_filename} in {output_dir}")
    except Exception as e:
        print(f"Error saving game-variant-model pass matrix to CSV: {e}")

    # Generate heatmap plot
    # Dynamic figure size based on matrix dimensions
    num_game_variants = len(pass_matrix_df.index)  # Use game_variant count
    num_models = len(pass_matrix_df.columns)

    # Adjust multiplier for cell size; 0.2 for game_variants (height), 0.5 for models (width)
    # Min size: 10 for width, 8 for height.
    fig_width = max(10, num_models * 0.5)
    fig_height = max(8, num_game_variants * 0.25)  # Use game_variant count

    # Cap max size to prevent extremely large figures
    fig_width = min(fig_width, 40)  # Max width 40 inches
    fig_height = min(fig_height, 60)  # Max height 60 inches

    plt.figure(figsize=(fig_width, fig_height))
    try:
        sns.heatmap(
            pass_matrix_df.astype(
                bool
            ),  # Ensure boolean for cmap interpretation if needed
            annot=False,  # No annotations for density
            cmap="RdYlGn",  # Red for False, Green for True
            cbar=False,  # No colorbar for density
            linewidths=0.5,
            linecolor="gray",
        )
        # Updated title and ylabel
        plt.title(
            "Game Variant Pass Status by Model (Green=Passed, Red=Failed/Not Attempted)"
        )
        plt.xlabel("Model")
        plt.ylabel("Game Variant")  # Changed from "Game"
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Updated plot filename
        plot_filename = "game_variant_model_pass_status_heatmap.pdf"
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath)
        print(f"Generated {plot_filename} in {output_dir}")
    except Exception as e:
        print(f"Error generating heatmap for game-variant-model pass matrix: {e}")
    finally:
        plt.close()  # Ensure plot is closed even if errors occur


def generate_model_category_performance_table(df: pd.DataFrame, output_dir: str):
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
        df_copy.groupby(["model", "category"])
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
    output_file_path_csv = os.path.join(output_dir, "model_category_performance.csv")
    try:
        formatted_table_csv.to_csv(output_file_path_csv)
        print(f"Model-category performance table saved to: {output_file_path_csv}")
    except Exception as e:
        print(f"Error saving model-category performance table to CSV: {e}")

    # Prepare table for LaTeX
    latex_table_for_tex = pivot_table.copy()
    latex_table_for_tex = (latex_table_for_tex * 100).round(1)

    # Map values to strings with percent sign for LaTeX
    # Ensure that na_rep is handled before to_latex if necessary, or by to_latex itself.
    latex_table_percent_str_df = latex_table_for_tex.map(
        lambda x: f"{x:.1f}\\%" if pd.notnull(x) else "-"  # Escaped % for LaTeX
    )

    output_file_path_latex = os.path.join(output_dir, "model_category_performance.tex")
    try:
        # Convert DataFrame to LaTeX string
        # escape=False allows pre-formatted LaTeX like \\% and \\\\
        # Ensure `latex_table_percent_str_df` is purely strings before to_latex if type issues persist
        # Use Styler API for booktabs compatibility if direct argument fails
        latex_string = latex_table_percent_str_df.to_latex(  # type: ignore
            na_rep="-",
            escape=False,
            bold_rows=True,
            header=[
                # "Category",
                "Adversarial \\ Resistance",  # Added double backslash for LaTeX line break
                "Cognitive \\ Abilities",  # Added double backslash
                "Realtime \\ Responsiveness",  # Added double backslash
                "Technical \\ Fluency",  # Added double backslash
                "Visual \\ Comprehension",  # Added double backslash
                "Average",
            ],
        )

        # For just the table code:
        latex_table_code = latex_string

        with open(output_file_path_latex, "w") as f:
            f.write(latex_table_code)
        print(
            f"Model-category performance LaTeX table saved to: {output_file_path_latex}"
        )
    except Exception as e:
        print(f"Error saving model-category performance table to LaTeX: {e}")
        raise e


def main():
    # Set a nicer plot style
    sns.set_theme(style="whitegrid")

    # Create output directory if it doesn't exist
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Load and preprocess data
    processed_data = load_and_preprocess_data()

    if processed_data.empty:
        print("No data processed. Cannot generate plots.")
        return

    # Generate plots
    plot_success_vs_difficulty_per_model(processed_data, output_dir)
    plot_success_rate_per_category(processed_data, output_dir)
    plot_overall_success_per_model(processed_data, output_dir)
    plot_success_rate_per_difficulty_per_category(processed_data, output_dir)
    plot_weighted_unsolved_tasks_by_category(processed_data, output_dir)
    plot_model_performance_radar(processed_data, output_dir)

    # Generate game-model pass matrix and plot
    generate_game_model_pass_matrix_and_plot(processed_data, output_dir)

    # Generate model-category performance table
    generate_model_category_performance_table(processed_data, output_dir)

    # List unsolved tasks
    list_unsolved_tasks(processed_data, output_dir)

    # List tasks solved by exactly one model
    list_tasks_solved_by_one_model(processed_data, output_dir)

    # List unsolved task families
    list_unsolved_task_families(processed_data, output_dir)

    # Plot category co-occurrence
    plot_category_cooccurrence(CATEGORIES_CSV_PATH, output_dir)

    # Save average success rate per difficulty
    save_average_success_per_difficulty_csv(processed_data, output_dir)

    # Save overall success rate per model
    save_overall_success_per_model_csv(processed_data, output_dir)

    print(
        f"\nAnalysis complete. Plots, lists, and category co-occurrence heatmap saved to the '{output_dir}' directory."
    )


if __name__ == "__main__":
    main()
