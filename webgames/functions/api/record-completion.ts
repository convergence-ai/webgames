import { D1Database } from "@cloudflare/workers-types";

interface Env {
  DB: D1Database;
}

export interface TaskCompletion {
  taskId: string;
  completionTime: number;
  user_id: string;
  start_time: number;
  user_agent: string;
  ip_address: string;
}

export const onRequestPost = async (context: {
  request: Request;
  env: Env;
}) => {
  // List all tables in the database
  const tables = await context.env.DB.prepare(
    `SELECT name FROM sqlite_master WHERE type='table'`
  ).all();

  console.log("Tables in database:", tables);

  try {
    const data: TaskCompletion = await context.request.json();
    const { taskId, completionTime, user_id, start_time } = data;

    // Get user agent and IP from headers
    const userAgent = context.request.headers.get("User-Agent") || "unknown";
    const ip = context.request.headers.get("CF-Connecting-IP") || "unknown";

    // Safely convert timestamps to ISO strings for SQLite
    const safeDate = (timestamp: number) => {
      try {
        const date = new Date(timestamp);
        // Check if date is valid and within reasonable range
        if (
          isNaN(date.getTime()) ||
          date.getFullYear() < 2020 ||
          date.getFullYear() > 2100
        ) {
          return new Date().toISOString();
        }
        return date.toISOString();
      } catch {
        return new Date().toISOString();
      }
    };

    const startTimeISO = safeDate(start_time);
    const completionTimeISO = safeDate(completionTime);

    // Insert into D1 database
    await context.env.DB.prepare(
      `INSERT INTO completions (task_id, start_time, completion_time, user_agent, ip_address, user_id)
       VALUES (?, ?, ?, ?, ?, ?)`
    )
      .bind(
        taskId,
        startTimeISO,
        completionTimeISO,
        userAgent,
        ip,
        user_id || "anonymous"
      )
      .run();

    return new Response(JSON.stringify({ success: true }), {
      headers: { "Content-Type": "application/json" },
      status: 200,
    });
  } catch (error) {
    console.error("Error saving task completion:", error);
    return new Response(
      JSON.stringify({
        success: false,
        error: "Failed to save task completion",
      }),
      {
        headers: { "Content-Type": "application/json" },
        status: 500,
      }
    );
  }
};
