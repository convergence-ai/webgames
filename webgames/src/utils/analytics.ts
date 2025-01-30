import { useFunctions } from "vite-plugin-cloudflare-functions/client";
import { TaskCompletion } from "../../functions/api/record-completion";

// eslint-disable-next-line react-hooks/rules-of-hooks
const client = useFunctions();

// Generate a random user ID if none exists
const getUserId = () => {
  const storedId = localStorage.getItem("user_id");
  if (storedId) return storedId;

  const newId =
    Math.random().toString(36).substring(2) + Date.now().toString(36);
  localStorage.setItem("user_id", newId);
  return newId;
};

// Ensure timestamp is a valid number
const getValidTimestamp = (timestamp?: number) => {
  if (
    typeof timestamp === "number" &&
    !isNaN(timestamp) &&
    isFinite(timestamp)
  ) {
    return timestamp;
  }
  return Date.now();
};

export async function recordTaskCompletion(
  partialCompletion: Partial<TaskCompletion>
): Promise<boolean> {
  try {
    const now = Date.now();
    // Get validated timestamps
    const validCompletionTime =
      getValidTimestamp(partialCompletion.completionTime) || now;
    const validStartTime =
      getValidTimestamp(partialCompletion.start_time) || now - 1000;

    const completion: TaskCompletion = {
      taskId: "unknown",
      completionTime: validCompletionTime,
      start_time: validStartTime,
      user_id: getUserId(),
      user_agent: navigator.userAgent,
      ip_address: "", // This will be populated by the server
      ...partialCompletion,
    };

    const response = await client.post("/api/record-completion", completion);
    const data = await response.json();
    return data.success;
  } catch (error) {
    console.error("Failed to record task completion:", error);
    return false;
  }
}
