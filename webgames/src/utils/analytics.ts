import { TaskCompletion } from "../../functions/api/record-completion";
import { TaskView } from "../../functions/api/record-view";

// Generate a random user ID if none exists
function getUserId() {
  const storedId = localStorage.getItem("user_id");
  if (storedId) return storedId;

  const newId = crypto.randomUUID();
  localStorage.setItem("user_id", newId);
  return newId;
}

// Ensure timestamp is a valid number
function getValidTimestamp(timestamp?: number) {
  if (
    typeof timestamp === "number" &&
    !isNaN(timestamp) &&
    isFinite(timestamp)
  ) {
    return timestamp;
  }
  return Date.now();
}

// Convert timestamp to ISO string and validate date range
function safeDate(timestamp: number): [string, boolean] {
  try {
    const date = new Date(timestamp);
    // Check if date is valid and within reasonable range
    if (
      isNaN(date.getTime()) ||
      date.getFullYear() < 2020 ||
      date.getFullYear() > 2100
    ) {
      return [new Date().toISOString(), false];
    }
    return [date.toISOString(), true];
  } catch {
    return [new Date().toISOString(), false];
  }
}

export async function recordTaskCompletion(
  taskId: string,
  completionTime: number,
  startTime: number
): Promise<boolean> {
  try {
    const [validCompletionTime, okCompletionTime] = safeDate(completionTime);
    if (!okCompletionTime) {
      console.error("Invalid completion time provided");
      return false;
    }

    const [validStartTime, okStartTime] = safeDate(startTime);

    if (!okStartTime) {
      console.error("Invalid start time provided");
      return false;
    }

    const completion: TaskCompletion = {
      taskId,
      completionTime: validCompletionTime,
      startTime: validStartTime,
      userId: getUserId(),
      host: window.location.host,
      url: window.location.href,
    };

    const response = await fetch("/api/record-completion", {
      method: "POST",
      body: JSON.stringify(completion),
    });
    const data = await response.json();
    return data.success;
  } catch (error) {
    console.error("Failed to record task completion:", error);
    return false;
  }
}

export async function recordTaskView(
  taskId: string,
  viewTime: number
): Promise<boolean> {
  try {
    const [viewTimeISO, okViewTime] = safeDate(getValidTimestamp(viewTime));
    if (!okViewTime) {
      console.error("Invalid view time provided");
      return false;
    }

    const view: TaskView = {
      taskId,
      viewTime: viewTimeISO,
      userId: getUserId(),
      host: window.location.host,
      url: window.location.href,
    };

    const response = await fetch("/api/record-view", {
      method: "POST",
      body: JSON.stringify(view),
    });
    const data = await response.json();
    return data.success;
  } catch (error) {
    console.error("Failed to record task view:", error);
    return false;
  }
}
