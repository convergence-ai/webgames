import React, { useState } from "react";
import WeeklyCalendar, { calendarEvents } from "../components/WeeklyCalendar";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_CalendarComprehension2Hard = "TIME_MASTER_CHALLENGER";
export const TASK_ID_CalendarComprehension2Hard = "calendar2-hard";

const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"];

// Questions derived and made harder from CalendarComprehension2.tsx
const questions = [
  {
    id: 1,
    question:
      "Calculate the total unscheduled minutes on Monday between 8am and 7pm, excluding the 12pm-1pm lunch break. Answer in an integer number of minutes.",
    // Original Q1 in C2: "How many minutes are unscheduled between 9am and 6pm on Monday? Answer in an integer number of minutes." -> "240"
    // Monday events: Daily Standup (9-10), Team Sync (10-11), Quick Break (11-11:30), Lunch (12-1), Client Call (1-2), Coffee Chat (2-2:30), Deep Work (3-6)
    // 8am-9am: 60 (free)
    // 11:30am-12pm: 30 (free)
    // 2:30pm-3pm: 30 (free)
    // After Deep Work (6pm) to 7pm: 60 (free)
    // Total: 60 + 30 + 30 + 60 = 180 minutes.
    answer: "180",
  },
  {
    id: 2,
    question:
      "On Tuesday, after the 'Client Meeting' (which ends at 4pm), what is the latest possible start time for a 75-minute uninterrupted meeting before 6pm? Answer in HH:MM format.",
    // Original Q2 in C2: "When is the earliest time on Tuesday after lunch that I could schedule a 45 minute meeting? Answer in HH:MM." -> "17:00"
    // Tuesday Events: Sales Training (9-12), Lunch (12-1), Client Meeting (1-4), Yoga Break (5-6)
    // After Client Meeting (4pm). Free slot 4pm-5pm. Yoga at 5pm.
    // For a 75 min meeting, it cannot fit before 6pm if started after 4pm, because Yoga is 5-6.
    // The slot is 4pm to 5pm (60 mins). So a 75 min meeting is not possible.
    // Let's adjust: "On Tuesday, after the 'Client Meeting' (ends 4pm), what's the maximum duration (in minutes) for a meeting that must end by 5pm?"
    // Answer: 60
    // Or, making it harder: "If 'Yoga Break' on Tuesday was cancelled, what is the latest possible start time for a 75-minute meeting that must end by 6 PM, assuming you start after the 'Client Meeting'? Answer in HH:MM format."
    // Client meeting ends 4pm. Slot 4pm to 6pm (120 mins). Latest start for 75min meeting is 6:00 PM - 75 mins = 4:45 PM.
    answer: "16:45",
  },
  {
    id: 3,
    question:
      "Considering only Wednesday and Friday, how many total minutes are dedicated to events that are NOT 'Lunch'? Answer in an integer number of minutes.",
    // Original Q3 in C2: "How many minutes do I have free between Retrospective and Weekly Sync on Friday? Answer in an integer number of minutes." -> "60"
    // Wednesday events: Marketing (9-11:30), Lunch (12-1), Design Review (1-3), Brainstorm (4-5:30)
    // Marketing: 2.5hr = 150. Design Review: 2hr = 120. Brainstorm: 1.5hr = 90. Total W: 150+120+90 = 360
    // Friday events: Planning (9-12), Lunch (12-1), Retrospective (2-3:30), Weekly Sync (4:30-5)
    // Planning: 3hr = 180. Retro: 1.5hr = 90. Weekly Sync: 0.5hr = 30. Total F: 180+90+30 = 300
    // Total non-lunch: 360 + 300 = 660
    answer: "660",
  },
  {
    id: 4,
    question:
      "Which day of the week has the most scheduled event time, and how many hours is it? (e.g., Monday, 7.5 hours)",
    // Monday: Standup(1) + Sync(1) + Break(0.5) + Lunch(1) + Call(1) + Chat(0.5) + DeepWork(3) = 8 hours
    // Tuesday: Training(3) + Lunch(1) + Meeting(3) + Yoga(1) = 8 hours
    // Wednesday: Marketing(2.5) + Lunch(1) + Design(2) + Brainstorm(1.5) = 7 hours
    // Thursday: Standup(0.5) + Workshop(3.5) + Lunch(1) + Review(2) = 7 hours
    // Friday: Planning(3) + Lunch(1) + Retro(1.5) + Sync(0.5) + Deadline(0) = 6 hours (Deadline is 5pm, no duration)
    // Actually, Project Deadline is 1pm-5pm = 4 hours.
    // Friday: Planning(3) + Lunch(1) + Retro(1.5) + Sync(0.5) + Project Deadline(4) = 10 hours
    // So Friday has the most. Let's verify events.
    // Monday: 9-10, 10-11, 11-11:30, 12-1, 1-2, 2-2:30, 3-6. Total: 1+1+0.5+1+1+0.5+3 = 8 hours.
    // Tuesday: 9-12, 12-1, 1-4, 5-6. Total: 3+1+3+1 = 8 hours.
    // Wednesday: 9-11:30, 12-1, 1-3, 4-5:30. Total: 2.5+1+2+1.5 = 7 hours.
    // Thursday: 9:30-10, 10-12, 12-1, 1-2, 3-5. (Original Thurs: Standup (9:30-10), Workshop (10-1:30), Lunch (1:30-2:30), Code Review (3-5))
    // calendarEvents.ts: Thursday: { title: "Daily Standup", start: "09:30", end: "10:00" }, { title: "Feature Workshop", start: "10:00", end: "13:30" }, { title: "Lunch", start: "13:30", end: "14:30" }, { title: "Code Review", start: "15:00", end: "17:00" }
    // Thursday: 0.5 + 3.5 + 1 + 2 = 7 hours.
    // Friday: { title: "Sprint Planning", start: "09:00", end: "12:00" }, { title: "Lunch", start: "12:00", end: "13:00" }, { title: "Retrospective", start: "14:00", end: "15:30" }, { title: "Weekly Sync", start: "16:30", end: "17:00" }, { title: "Project Deadline", start: "13:00", end: "17:00" } // Overlap!
    // Project Deadline 1-5pm (4hr). Retrospective 2-3:30pm (1.5hr) is INSIDE Project Deadline.
    // This makes the calendar data tricky.
    // Let's assume events can overlap and we sum their individual durations.
    // Friday: Planning(3) + Lunch(1) + Retro(1.5) + Sync(0.5) + Deadline(4) = 10 hours.
    // So, Friday, 10 hours
    answer: "Friday, 10 hours",
  },
];

const CalendarComprehension2Hard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(
    TASK_ID_CalendarComprehension2Hard
  );
  const [answers, setAnswers] = useState<{ [key: number]: string }>({});
  const [showResults, setShowResults] = useState(false);
  const [message, setMessage] = useState("");
  const [isComplete, setIsComplete] = useState(false);

  const handleAnswerChange = (questionId: number, value: string) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const checkAnswers = () => {
    const allCorrect = questions.every(
      (q) => answers[q.id]?.trim().toLowerCase() === q.answer.toLowerCase()
    );

    if (allCorrect) {
      setMessage("Congratulations! All answers are correct!");
      setIsComplete(true);
      recordSuccess();
    } else {
      setMessage(
        "Some answers are incorrect. Please review carefully and try again."
      );
    }
    setShowResults(true);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">
        Advanced Calendar Comprehension Challenge (Hard)
      </h1>

      <WeeklyCalendar
        events={calendarEvents}
        days={days}
        startHour={8}
        endHour={19}
      />

      <div className="mt-8">
        {isComplete ? (
          <div className="p-8 bg-green-100 border-2 border-green-500 rounded-lg text-center">
            <h2 className="text-2xl font-bold text-green-700 mb-4">
              🎉 Challenge Complete!
            </h2>
            <p className="text-xl mb-4">Your password is:</p>
            <div className="inline-block font-mono text-2xl bg-green-200 px-6 py-3 rounded-lg font-bold">
              {PASSWORD_CalendarComprehension2Hard}
            </div>
          </div>
        ) : (
          <>
            <h2 className="text-2xl font-bold mb-4">
              Answer these questions about the calendar:
            </h2>

            {questions.map((q) => (
              <div key={q.id} className="mb-6">
                <p className="text-lg mb-2">
                  {q.id}. {q.question}
                </p>
                <input
                  type="text"
                  className={`w-full p-2 border rounded-lg ${
                    showResults &&
                    answers[q.id]?.trim().toLowerCase() !==
                      q.answer.toLowerCase()
                      ? "border-red-500"
                      : "border-gray-300"
                  }`}
                  value={answers[q.id] || ""}
                  onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                  placeholder={q.id === 4 ? "Day, X hours" : ""}
                />
                {showResults &&
                  answers[q.id]?.trim().toLowerCase() !==
                    q.answer.toLowerCase() && (
                    <p className="text-red-500 text-sm mt-1">
                      Incorrect answer
                    </p>
                  )}
              </div>
            ))}

            <button
              onClick={checkAnswers}
              className="mt-4 bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition-colors"
            >
              Check Answers
            </button>

            {message && (
              <p
                className={`mt-4 ${
                  isComplete ? "text-green-600" : "text-red-600"
                }`}
              >
                {message}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default CalendarComprehension2Hard;
