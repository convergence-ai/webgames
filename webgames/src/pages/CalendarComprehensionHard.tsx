import React, { useState } from "react";
import WeeklyCalendar, { calendarEvents } from "../components/WeeklyCalendar";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_CalendarComprehensionHard = "CALENDAR_GURU_2025";
export const TASK_ID_CalendarComprehensionHard = "calendar-comprehension-hard";

const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"];

const questions = [
  {
    id: 1,
    question:
      "What is the total number of hours scheduled for events on Wednesday?", // Requires summing durations
    answer: "5.5",
  },
  {
    id: 2,
    question:
      "Which event on Friday ends the latest, and what time does it end? (e.g., Event Name, HH:MM AM/PM)", // Requires finding latest and formatting
    answer: "Project Deadline, 5:00 PM",
  },
  {
    id: 3,
    question: "How many events are exactly 1 hour long across all days?",
    answer: "3",
  },
  {
    id: 4,
    question:
      "List all events that occur on Tuesday, in chronological order, separated by a comma and space.", // Requires listing and ordering
    answer: "Sales Training, Client Meeting, Yoga Break",
  },
  {
    id: 5,
    question:
      "What is the start time of the earliest event on Thursday? (HH:MM AM/PM)",
    answer: "9:30 AM",
  },
  {
    id: 6,
    question: "How many days have an event called 'Lunch'?",
    answer: "5",
  },
  {
    id: 7,
    question: "Which day has the most individual events scheduled?",
    answer: "Monday",
  },
];

const CalendarComprehensionHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_CalendarComprehensionHard);
  const [answers, setAnswers] = useState<{ [key: number]: string }>({});
  const [showResults, setShowResults] = useState(false);
  const [message, setMessage] = useState("");

  const handleAnswerChange = (questionId: number, value: string) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const checkAnswers = () => {
    const allCorrect = questions.every(
      (q) => answers[q.id]?.trim().toLowerCase() === q.answer.toLowerCase()
    );
    if (allCorrect) {
      setMessage(
        `Congratulations! All answers are correct! The password is: ${PASSWORD_CalendarComprehensionHard}`
      );
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
        Calendar Comprehension Challenge (Hard)
      </h1>

      <WeeklyCalendar
        events={calendarEvents} // Potentially, a more complex event set could be used for 'Hard'
        days={days}
        startHour={8} // Wider range to make scanning harder
        endHour={19}
      />

      <div className="mt-8">
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
                answers[q.id]?.trim().toLowerCase() !== q.answer.toLowerCase()
                  ? "border-red-500"
                  : "border-gray-300"
              }`}
              value={answers[q.id] || ""}
              onChange={(e) => handleAnswerChange(q.id, e.target.value)}
              placeholder={
                q.id === 2
                  ? "Event Name, HH:MM AM/PM"
                  : q.id === 4
                  ? "Event1, Event2, ..."
                  : ""
              }
            />
            {showResults &&
              answers[q.id]?.trim().toLowerCase() !==
                q.answer.toLowerCase() && (
                <p className="text-red-500 text-sm mt-1">Incorrect answer</p>
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
              message.includes("Congratulations")
                ? "text-green-600"
                : "text-red-600"
            }`}
          >
            {message}
          </p>
        )}
      </div>
    </div>
  );
};

export default CalendarComprehensionHard;
