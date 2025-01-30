import { useEffect } from "react";
import { Link, useSearchParams } from "react-router-dom";
import Balancer from "react-wrap-balancer";
import Stopwatch from "../components/Stopwatch";
import { routes } from "../router/routes";
import { recordTaskView } from "../utils/analytics";

export default function Home() {
  useEffect(() => {
    (async () => {
      await recordTaskView("home", Date.now());
    })();
  }, []);

  const visibleRoutes = routes.filter((route) => !route.hidden);
  const [searchParams] = useSearchParams();
  const showDownloads = searchParams.get("showDownloads") === "true";

  const downloadChallengesJSONL = () => {
    const jsonl = visibleRoutes
      .map((route) =>
        JSON.stringify({
          id: route.path,
          title: route.title,
          description: route.description,
          path: route.path,
          password: route.password,
          tags: route.tags,
        })
      )
      .join("\n");

    const blob = new Blob([jsonl], { type: "application/x-jsonlines" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "webgames-v0-challenges.jsonl";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadChallengesCSV = () => {
    const headers = ["id", "title", "description", "path", "password", "tags"];
    const csvContent = [
      headers.join(","),
      ...visibleRoutes.map((route) => {
        return [
          route.path,
          `"${route.title.replace(/"/g, '""')}"`,
          `"${route.description.replace(/"/g, '""')}"`,
          route.path,
          route.password,
          `"${(route.tags || []).join(";")}"`,
        ].join(",");
      }),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "webgames-v0-challenges.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="bg-gradient-to-b from-[#1c9188]/10 via-[#1c9188]/5 to-white border-b border-[#1c9188]/10">
        <div className="container mx-auto px-2 py-16">
          <div className="flex flex-col sm:flex-row justify-between items-center mb-12 gap-4">
            <div className="flex flex-col sm:flex-row items-center sm:items-baseline gap-4">
              <h1 className="text-5xl font-bold bg-gradient-to-br from-[#1c9188] via-[#1c9188] to-[#1c9188]/90 bg-clip-text text-transparent">
                WebGames
              </h1>
              <div className="flex items-center gap-2">
                <span className="text-xl text-[#1c9188]">by</span>
                <img
                  src="/convergence.svg"
                  alt="Convergence Logo"
                  className="h-6"
                />
              </div>
            </div>
            <div className="flex gap-4 items-center">
              <a
                href="https://github.com/convergence-ai/webgames"
                target="_blank"
                rel="noopener noreferrer"
                className="hidden sm:block px-4 py-2 text-sm font-mono text-gray-400 hover:text-gray-600 transition-colors duration-200"
              >
                <span className="flex items-center gap-2">
                  <svg
                    className="w-4 h-4"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                  >
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                  </svg>
                  GitHub
                </span>
              </a>
              {showDownloads && (
                <div className="hidden sm:flex gap-2">
                  <button
                    onClick={downloadChallengesJSONL}
                    className="px-4 py-2 text-sm font-mono text-gray-400 hover:text-gray-600 transition-colors duration-200"
                  >
                    ↓ Challenges JSONL
                  </button>
                  <button
                    onClick={downloadChallengesCSV}
                    className="px-4 py-2 text-sm font-mono text-gray-400 hover:text-gray-600 transition-colors duration-200"
                  >
                    ↓ Challenges CSV
                  </button>
                </div>
              )}
            </div>
          </div>
          <div className="max-w-2xl mx-auto text-center">
            <p className="text-xl text-gray-600 mb-6">
              <span className="sm:hidden">
                <Balancer>
                  Welcome to WebGames - a collection of challenges designed for
                  testing general-purpose web-browsing AI agents.
                </Balancer>
              </span>
              <span className="hidden sm:inline">
                Welcome to WebGames - a collection of challenges designed for
                testing general-purpose web-browsing AI agents.
              </span>
            </p>
            <p className="text-xl text-gray-600 mb-6">
              <span className="sm:hidden">
                <Balancer>
                  These challenges are easy for humans but hard for AI agents to
                  complete. Each task provides a unique password to indicate
                  successful completion.
                </Balancer>
              </span>
              <span className="hidden sm:inline">
                These challenges are easy for humans but hard for AI agents to
                complete. Each task provides a unique password to indicate
                successful completion.
              </span>
            </p>
            <p className="text-xl text-gray-600 mb-6">
              You can use this handy stopwatch to time your progress:
            </p>
            <div className="mb-8">
              <span className="hidden sm:inline">
                <Stopwatch />
              </span>
            </div>
            <p className="text-xl text-gray-600 mb-6">
              <span className="sm:hidden">
                <Balancer>Let the WebGames begin!</Balancer>
              </span>
              <span className="hidden sm:inline">Let the WebGames begin!</span>
            </p>
          </div>
        </div>
      </div>
      <div className="bg-blue-50/40">
        <div className="container mx-auto px-2 py-8">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {visibleRoutes.map((route, index) => (
              <Link
                key={route.path}
                to={route.path}
                className="group flex gap-3 p-4 bg-white hover:bg-gray-50 border border-gray-200 hover:border-blue-500 rounded-lg shadow-md hover:shadow-lg transition-all duration-100 ease-in-out relative"
              >
                <div className="shrink-0 w-10 flex items-start pt-1">
                  <span
                    className="text-2xl group-hover:scale-110 transition-transform duration-100"
                    role="img"
                    aria-label={route.title}
                  >
                    {route.icon}
                  </span>
                </div>
                <div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-xs font-medium text-gray-400">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <h2 className="text-sm font-medium text-gray-900">
                      {route.title}
                    </h2>
                  </div>
                  <p className="mt-1 text-sm text-gray-500 line-clamp-2">
                    {route.description}
                  </p>
                </div>
              </Link>
            ))}
          </div>
          <footer className="mt-8 text-center text-sm text-gray-500">
            Built by <a href="https://convergence.ai">convergence.ai</a>
          </footer>
        </div>
      </div>
    </div>
  );
}
