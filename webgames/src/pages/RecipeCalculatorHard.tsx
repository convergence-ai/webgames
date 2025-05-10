import React, { useState } from "react";
import { Link } from "react-router-dom";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";
import { recipes } from "./recipes/recipes";

export const PASSWORD_RecipeCalculatorHard = "HARD_RECIPE_CALC_789";
export const TASK_ID_RecipeCalculatorHard = "recipe-calculator-hard";

const RecipeCalculatorHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_RecipeCalculatorHard);
  const [answer, setAnswer] = useState<string>("");
  const [message, setMessage] = useState<string>("");
  const [password, setPassword] = useState<string>("");

  // Target recipe is Beef Stew, calculate for 10 servings.
  // This recipe has more ingredients and might require more careful parsing.
  // The target ingredient is "beef chuck"
  const targetRecipe = recipes.find((r) => r.name === "Hearty Beef Stew");
  const targetServings = 10; // Hard: larger, less common serving size
  const targetIngredient = "beef chuck"; // Hard: specific cut of meat

  const calculateExpectedAmount = () => {
    if (!targetRecipe) return 0;
    const ingredient = targetRecipe.ingredients.find(
      (i) => i.item.toLowerCase() === targetIngredient.toLowerCase() // Case-insensitive match
    );
    if (
      !ingredient ||
      typeof ingredient.amount !== "number" ||
      typeof targetRecipe.servings !== "number" ||
      targetRecipe.servings === 0
    ) {
      // Add a console log for debugging if an ingredient is not found or types are wrong.
      console.error(
        "Recipe or ingredient data issue:",
        targetRecipe,
        ingredient
      );
      return 0;
    }
    // The beef chuck amount is in kg, task asks for grams. Original recipe serves 6.
    // Original amount (kg) * 1000 (g/kg) * targetServings / originalServings
    return (ingredient.amount * 1000 * targetServings) / targetRecipe.servings;
  };

  const handleSubmit = () => {
    const expectedAmount = calculateExpectedAmount();
    const userAnswer = parseFloat(answer);

    if (expectedAmount === 0 && targetRecipe) {
      setMessage(
        "Could not calculate expected amount. Please check recipe data."
      );
      return;
    }
    if (expectedAmount === 0 && !targetRecipe) {
      setMessage(
        "Target recipe 'Hearty Beef Stew' not found. Please check available recipes."
      );
      return;
    }

    if (Math.abs(userAnswer - expectedAmount) < 0.01) {
      // Allow for minor floating point differences
      setPassword(PASSWORD_RecipeCalculatorHard);
      setMessage(
        "Excellent! You've correctly calculated the amount of beef chuck needed for the grand feast!"
      );
      recordSuccess();
    } else {
      setMessage(
        `That's not quite right. You need to calculate for ${targetServings} people. The original 'Hearty Beef Stew' recipe serves ${
          targetRecipe?.servings || "unknown number of"
        } people. Don't forget to convert units if necessary!`
      );
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-2xl font-bold">
              Recipe Calculator Challenge (Hard)
            </h1>
          </div>

          <div className="mb-6">
            <p className="text-gray-700 mb-4">
              You're planning a large gathering for <strong>10 people</strong>{" "}
              and have decided to make "Hearty Beef Stew". This recipe is known
              for its rich flavor but requires precise ingredient scaling.
            </p>
            <p className="text-gray-700 mb-4">
              Consult the{" "}
              <Link to="/recipes" className="text-blue-600 hover:text-blue-800">
                recipe collection
              </Link>{" "}
              for the "Hearty Beef Stew" recipe. Your task is to calculate
              exactly how many <strong>grams</strong> of{" "}
              <strong>beef chuck</strong> are needed for{" "}
              <strong>10 servings</strong>. Pay close attention to the original
              serving size and any unit conversions that might be necessary.
            </p>
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              How many grams of beef chuck are needed?
            </label>
            <div className="flex gap-4">
              <input
                type="number"
                step="0.01" // Allow for decimals if conversion results in them
                className="flex-1 px-3 py-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                placeholder="Enter amount in grams..."
              />
              <button
                onClick={handleSubmit}
                className="px-6 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
              >
                Check Answer
              </button>
            </div>
          </div>

          {message && (
            <div
              className={`p-4 rounded-lg ${
                password ? "bg-green-50" : "bg-red-50"
              }`}
            >
              <p className={`${password ? "text-green-800" : "text-red-800"}`}>
                {message}
              </p>
              {password && (
                <p className="mt-2 font-bold text-green-800">
                  Password: {password}
                </p>
              )}
            </div>
          )}
        </div>
        <div className="flex justify-center mt-6">
          <Link
            to="/recipes"
            className="inline-flex items-center px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white font-medium rounded-lg transition-colors duration-200"
          >
            Go to recipe site →
          </Link>
        </div>
      </div>
    </div>
  );
};

export default RecipeCalculatorHard;
