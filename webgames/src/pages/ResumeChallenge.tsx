import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";
import generateResumePDF from "../utils/generateResumePDF";

export const TASK_ID_ResumeChallenge = "resume-challenge";
export const PASSWORD_ResumeChallenge = "PDFdetective";

// The expected answers based on the mock resume PDF
const EXPECTED_ANSWERS = {
  fullName: "Judy Hopps",
  email: "judy.hopps@zootopia.gov",
  phone: "(555) 123-4567",
  education: "Bunnyburrow University",
  degree: "Bachelor of Science in Criminal Justice",
  recentEmployer: "Zootopia Police Department",
  jobTitle: "Detective",
  skill1: "Criminal Profiling",
  skill2: "Digital Forensics",
  skill3: "Public Speaking",
  projectName: "Night Howler Investigation",
};

const ResumeChallenge: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ResumeChallenge);
  const [formData, setFormData] = useState({
    fullName: "",
    email: "",
    phone: "",
    education: "",
    degree: "",
    recentEmployer: "",
    jobTitle: "",
    skill1: "",
    skill2: "",
    skill3: "",
    projectName: "",
  });

  const [errors, setErrors] = useState<string[]>([]);
  const [showSuccess, setShowSuccess] = useState(false);
  const [isDownloaded, setIsDownloaded] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleDownload = () => {
    try {
      // Generate the PDF
      const pdfBlob = generateResumePDF();

      // Create a URL for the blob
      const url = URL.createObjectURL(pdfBlob);

      // Create a temporary link element
      const link = document.createElement("a");
      link.href = url;
      link.download = "Judy_Hopps_Resume.pdf";

      // Append to body, click to download, and remove
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Clean up the URL object
      URL.revokeObjectURL(url);

      // Set the downloaded flag
      setIsDownloaded(true);
    } catch (error) {
      console.error("Error generating or downloading PDF:", error);
      alert("Failed to download the resume. Please try again.");
    }
  };

  const validateForm = () => {
    const newErrors: string[] = [];

    Object.entries(EXPECTED_ANSWERS).forEach(([key, expectedValue]) => {
      const userValue = formData[key as keyof typeof formData];
      if (userValue.trim().toLowerCase() !== expectedValue.toLowerCase()) {
        newErrors.push(
          `Incorrect information for ${key
            .replace(/([A-Z])/g, " $1")
            .toLowerCase()}`
        );
      }
    });

    return newErrors;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    // Simulate network delay
    setTimeout(() => {
      const newErrors = validateForm();
      setErrors(newErrors);

      if (newErrors.length === 0) {
        setShowSuccess(true);
        recordSuccess();
      }

      setLoading(false);
    }, 1000);
  };

  return (
    <div className="max-w-4xl mx-auto mt-8 p-6 bg-white rounded-lg shadow-lg">
      <h1 className="text-2xl font-bold mb-4">Resume Detective Challenge</h1>

      <div className="mb-6">
        <p className="mb-4">
          Download the mock resume of Judy Hopps, carefully read through it, and
          fill out the form below with the information you find. Pay attention
          to specific details!
        </p>

        <div className="p-4 bg-blue-100 rounded mb-4">
          <h3 className="font-semibold text-blue-800 mb-2">
            Tips for Success:
          </h3>
          <ul className="list-disc pl-5 text-blue-800">
            <li>
              For education, use the highest degree obtained (university, not
              academy)
            </li>
            <li>Make sure to use exact spelling and formatting</li>
          </ul>
        </div>

        <button
          className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 transition-colors mb-4"
          onClick={handleDownload}
        >
          Download Resume PDF
        </button>

        {isDownloaded && (
          <div className="p-4 bg-green-100 rounded mb-4">
            <p className="text-green-800">
              Resume downloaded! Make sure to read it carefully before filling
              out the form.
            </p>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit}>
        <div className="space-y-6">
          <div>
            <h2 className="text-xl font-semibold mb-3">Personal Information</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label
                  htmlFor="fullName"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Full Name
                </label>
                <input
                  type="text"
                  id="fullName"
                  name="fullName"
                  value={formData.fullName}
                  onChange={handleChange}
                  placeholder="As shown on resume"
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label
                  htmlFor="email"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Email Address
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="Official email from resume"
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label
                  htmlFor="phone"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Phone Number
                </label>
                <input
                  type="text"
                  id="phone"
                  name="phone"
                  value={formData.phone}
                  onChange={handleChange}
                  placeholder="Including parentheses and dashes"
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-xl font-semibold mb-3">Education</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label
                  htmlFor="education"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  University/College
                </label>
                <input
                  type="text"
                  id="education"
                  name="education"
                  value={formData.education}
                  onChange={handleChange}
                  placeholder="Higher education institution name"
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label
                  htmlFor="degree"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Degree
                </label>
                <input
                  type="text"
                  id="degree"
                  name="degree"
                  value={formData.degree}
                  onChange={handleChange}
                  placeholder="Complete degree name including field"
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-xl font-semibold mb-3">Work Experience</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label
                  htmlFor="recentEmployer"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Current Employer
                </label>
                <input
                  type="text"
                  id="recentEmployer"
                  name="recentEmployer"
                  value={formData.recentEmployer}
                  onChange={handleChange}
                  placeholder="Organization's full name"
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label
                  htmlFor="jobTitle"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Current Job Title
                </label>
                <input
                  type="text"
                  id="jobTitle"
                  name="jobTitle"
                  value={formData.jobTitle}
                  onChange={handleChange}
                  placeholder="Current position title"
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-xl font-semibold mb-3">Skills</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label
                  htmlFor="skill1"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Investigation Skill
                </label>
                <input
                  type="text"
                  id="skill1"
                  name="skill1"
                  value={formData.skill1}
                  onChange={handleChange}
                  placeholder="First investigation skill listed"
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label
                  htmlFor="skill2"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Technology Skill
                </label>
                <input
                  type="text"
                  id="skill2"
                  name="skill2"
                  value={formData.skill2}
                  onChange={handleChange}
                  placeholder="First technology skill listed"
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label
                  htmlFor="skill3"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Communication Skill
                </label>
                <input
                  type="text"
                  id="skill3"
                  name="skill3"
                  value={formData.skill3}
                  onChange={handleChange}
                  placeholder="First communication skill listed"
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-xl font-semibold mb-3">Projects</h2>

            <div>
              <label
                htmlFor="projectName"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Featured Project Name
              </label>
              <input
                type="text"
                id="projectName"
                name="projectName"
                value={formData.projectName}
                onChange={handleChange}
                placeholder="Name of first project listed"
                className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              />
            </div>
          </div>

          <div className="mt-6">
            {errors.length > 0 && (
              <div className="p-4 bg-red-100 rounded mb-4">
                <p className="text-red-800 font-medium">
                  Please correct the following errors:
                </p>
                <ul className="list-disc pl-5 mt-1">
                  {errors.map((error, index) => (
                    <li key={index} className="text-red-700">
                      {error}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {showSuccess && (
              <div className="p-4 bg-green-100 rounded mb-4">
                <p className="text-green-800 font-medium">
                  Great job! You've correctly filled out all the information
                  from the resume.
                </p>
                <p className="text-green-800 font-bold mt-2">
                  The secret password is: {PASSWORD_ResumeChallenge}
                </p>
              </div>
            )}

            <button
              type="submit"
              className={`w-full py-2 px-4 rounded ${
                loading ? "bg-gray-400" : "bg-blue-500 hover:bg-blue-600"
              } text-white transition-colors`}
              disabled={loading || showSuccess}
            >
              {loading ? (
                <span>Processing...</span>
              ) : (
                <span>Submit Answers</span>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default ResumeChallenge;
