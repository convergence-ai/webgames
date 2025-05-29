import asyncio
import json
import socket
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

from browser_use import Agent, Browser, BrowserConfig, BrowserContextConfig
from browser_use.agent.views import AgentHistoryList
from dotenv import load_dotenv
from inspect_ai import Task, eval_async, task
from inspect_ai._util.content import Content, ContentImage, ContentText
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ModelOutput
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

TASK_LIMIT: int | None = None

OPENAI_O4_MINI = "openai/o4-mini"
OPENAI_O3 = "openai/o3"
OPENAI_GPT_4O = "openai/gpt-4o"
OPENAI_GPT_4O_MINI = "openai/gpt-4o-mini"
ANTHROPIC_CLAUDE_3_7_SONNET = "anthropic/claude-3-7-sonnet-20250219"
GOOGLE_GEMINI_2_5_PRO_PREVIEW = "google/gemini-2.5-pro-preview-05-06"
GOOGLE_GEMINI_2_5_PRO_TEXTONLY = "google/gemini-2.5-pro-preview-05-06-textonly"
GOOGLE_GEMINI_2_5_FLASH_PREVIEW = "google/gemini-2.5-flash-preview-04-17"

QWEN_VL_7B_INSTRUCT = "none/Qwen2.5-VL-7B-Instruct"
QWEN_VL_32B_INSTRUCT = "none/Qwen2.5-VL-32B-Instruct"
QWEN_VL_72B_INSTRUCT = "none/Qwen2.5-VL-72B-Instruct"

MODELS = [
    OPENAI_O4_MINI,
    OPENAI_O3,
    ANTHROPIC_CLAUDE_3_7_SONNET,
    GOOGLE_GEMINI_2_5_PRO_PREVIEW,
    GOOGLE_GEMINI_2_5_FLASH_PREVIEW,
    QWEN_VL_7B_INSTRUCT,
    QWEN_VL_32B_INSTRUCT,
    QWEN_VL_72B_INSTRUCT,
]


def load_webgames_dataset(jsonl_file_path: str) -> MemoryDataset:
    """
    Loads tasks from a JSONL file and converts them into an InspectAI MemoryDataset.

    Args:
        jsonl_file_path: Path to the .jsonl file containing task definitions.

    Returns:
        An InspectAI MemoryDataset.
    """
    samples: List[Sample] = []
    task_prompt_template = (
        "go to https://webgames.convergence.ai/{task_id} and complete the challenge "
        "as described. On successful completion, the task will give you a password. "
        "Tell me what the password is."
    )

    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            task_data = json.loads(line)
            task_id = task_data.get("id")
            password = task_data.get("password")

            if not task_id or not password:
                # Or raise an error, or log a warning
                print(f"Skipping task due to missing id or password: {task_data}")
                continue

            prompt = task_prompt_template.format(task_id=task_id)

            metadata: Dict[str, Any] = {
                key: value
                for key, value in task_data.items()
                if key not in ["id", "password"]
            }

            sample = Sample(
                id=task_id,
                input=prompt,
                target=password,
                metadata=metadata,
            )
            samples.append(sample)

    if TASK_LIMIT is not None:
        return MemoryDataset(samples=samples[:TASK_LIMIT], name="WebGamesTasks")
    return MemoryDataset(samples=samples, name="WebGamesTasks")


def _agent_history_to_messages(
    agent_history: AgentHistoryList,
) -> Sequence[ChatMessage]:
    messages: list[ChatMessageAssistant] = []
    for history_item in agent_history.history:
        content_items: list[Content] = []

        try:
            if history_item.model_output:
                # Add current state/goal
                current_state = history_item.model_output.current_state
                content_items.append(
                    ContentText(text=f"Next Goal: {current_state.next_goal}")
                )
                if current_state.memory:
                    content_items.append(
                        ContentText(text=f"Memory: {current_state.memory}")
                    )

                # Add actions
                for action in history_item.model_output.action:
                    action_dict = action.model_dump(exclude_none=True)
                    # The action is usually a single key in the dict like {'go_to_url': {'url': '...'}}
                    action_name = next(iter(action_dict))
                    action_params = action_dict[action_name]
                    if isinstance(action_params, dict):
                        params_str = ", ".join(
                            f"{k}='{v}'" for k, v in action_params.items()
                        )
                        content_items.append(
                            ContentText(text=f"Action: {action_name}({params_str})")
                        )
                    else:
                        content_items.append(
                            ContentText(text=f"Action: {action_name}({action_params})")
                        )

            # Add results
            for result_item in history_item.result:
                if result_item.extracted_content:
                    content_items.append(
                        ContentText(text=f"Result: {result_item.extracted_content}")
                    )
                if result_item.error:
                    content_items.append(
                        ContentText(text=f"Error: {result_item.error}")
                    )
                if result_item.is_done is not None:
                    content_items.append(
                        ContentText(
                            text=f"Is Done: {result_item.is_done}, Success: {result_item.success}"
                        )
                    )

            # Add screenshot if available
            if history_item.state and history_item.state.screenshot:
                screenshot_data_url = (
                    f"data:image/png;base64,{history_item.state.screenshot}"
                )
                content_items.append(ContentImage(image=screenshot_data_url))

        except Exception as e:
            # If any error occurs during processing, add an error message
            content_items.append(
                ContentText(text=f"Error processing history item: {str(e)}")
            )

        # Always append a message, even if content_items is empty
        if content_items:
            messages.append(ChatMessageAssistant(content=content_items))
        else:
            # Fallback if no specific content could be extracted
            messages.append(ChatMessageAssistant(content=str(history_item)))

    return messages


def _get_free_port() -> int:
    """
    Get a free port number that is not in use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _llm_model_name_to_base_chat_model(llm_model_name: str) -> BaseChatModel:
    if "/" not in llm_model_name:
        raise ValueError(
            f"Unsupported model (must be in the format 'provider/model' e.g. 'openai/gpt-4o'): {llm_model_name}"
        )

    trimmed_model_name = llm_model_name.split("/")[-1]
    trimmed_model_name = trimmed_model_name.replace("-textonly", "")
    if llm_model_name.startswith("openai/"):
        return ChatOpenAI(
            model=trimmed_model_name,
        )
    elif llm_model_name.startswith("anthropic/"):
        return ChatAnthropic(
            model_name=trimmed_model_name,
            timeout=None,
            stop=None,
        )
    elif llm_model_name.startswith("google/"):
        return ChatGoogleGenerativeAI(
            model=trimmed_model_name,
        )
    elif llm_model_name.startswith("none/"):
        base_url: str
        if llm_model_name == QWEN_VL_7B_INSTRUCT:
            base_url = "http://slurmus-a3nodeset-2:8007/v1"
        elif llm_model_name == QWEN_VL_32B_INSTRUCT:
            base_url = "http://slurmus-a3nodeset-14:8032/v1"
        elif llm_model_name == QWEN_VL_72B_INSTRUCT:
            base_url = "http://slurmus-a3nodeset-2:8072/v1"
        else:
            raise ValueError(
                f"Unsupported Qwen model: {llm_model_name}. Must be one of {QWEN_VL_7B_INSTRUCT}, {QWEN_VL_32B_INSTRUCT}, or {QWEN_VL_72B_INSTRUCT}"
            )

        return ChatOpenAI(
            model="Qwen/" + trimmed_model_name,
            base_url=base_url,
            api_key=SecretStr("EMPTY"),
        )

    raise ValueError(f"Unsupported model: {llm_model_name}")


@solver
def browser_agent_solver() -> Solver:
    """
    An InspectAI solver that uses the browser_use.Agent to perform a task.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        llm = _llm_model_name_to_base_chat_model(str(state.model))
        text_only = str(state.model).endswith("-textonly")
        agent_instance = Agent(
            enable_memory=False,
            task=state.input_text,
            llm=llm,
            use_vision=not text_only,
            generate_gif=False,
            max_failures=30,
            retry_delay=10,
            browser=Browser(
                config=BrowserConfig(
                    chrome_remote_debugging_port=_get_free_port(),
                    new_context_config=BrowserContextConfig(
                        minimum_wait_page_load_time=0.25,
                    ),
                ),
            ),
        )

        try:
            agent_history: AgentHistoryList | None = await agent_instance.run(
                max_steps=20
            )
        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            state.messages = [ChatMessageAssistant(content=error_msg)]
            state.output = ModelOutput(
                model=str(state.model),
                error=error_msg,
            )
            return state

        if not agent_history:
            error_msg = "Agent run did not produce a history object."
            state.messages = [ChatMessageAssistant(content=error_msg)]
            state.output = ModelOutput(
                model=str(state.model),
                error=error_msg,
            )
            return state

        if not agent_history.is_done():
            error_msg = "Agent did not reach a 'done' state."
            messages = list(_agent_history_to_messages(agent_history))
            messages.append(ChatMessageAssistant(content=error_msg))
            state.messages = messages
            state.output = ModelOutput(
                model=str(state.model),
                error=error_msg,
            )
            return state

        if not agent_history.is_successful():
            raw_errors: List[Optional[str]] = agent_history.errors()
            actual_errors: List[str] = [err for err in raw_errors if err is not None]
            error_summary = (
                "; ".join(actual_errors) if actual_errors else "unknown error(s)"
            )
            error_msg = f"Agent finished unsuccessfully. Errors: {error_summary}"
            messages = list(_agent_history_to_messages(agent_history))
            messages.append(ChatMessageAssistant(content=error_msg))
            state.messages = messages
            state.output = ModelOutput(
                model=str(state.model),
                error=error_msg,
            )
            return state

        messages = list(_agent_history_to_messages(agent_history))
        final_answer = agent_history.final_result() or ""
        messages.append(ChatMessageAssistant(content=final_answer))
        state.messages = messages

        state.output = ModelOutput.from_content(
            model=str(state.model), content=final_answer
        )
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def webgames_scorer():
    async def score(state: TaskState, target: Target):
        answer = state.output.completion
        correct = target.text in answer
        return Score(value="C" if correct else "I", answer=answer)

    return score


@task()
def webgames_browser_agent_eval():
    return Task(
        dataset=load_webgames_dataset("webgames_tasks.jsonl"),
        solver=[
            browser_agent_solver(),
        ],
        scorer=webgames_scorer(),
    )


async def main():
    try:
        await eval_async(
            webgames_browser_agent_eval(),
            max_samples=10,
            max_tasks=10,
            model=[
                # OPENAI_O4_MINI,
                # OPENAI_O3,
                # OPENAI_GPT_4O,
                # OPENAI_GPT_4O_MINI,
                # ANTHROPIC_CLAUDE_3_7_SONNET,
                # GOOGLE_GEMINI_2_5_PRO_PREVIEW,
                GOOGLE_GEMINI_2_5_PRO_TEXTONLY,
                # GOOGLE_GEMINI_2_5_FLASH_PREVIEW,
                # QWEN_VL_7B_INSTRUCT,
                # QWEN_VL_32B_INSTRUCT,
                # QWEN_VL_72B_INSTRUCT,
            ],
            # epochs=Epochs(5, ["mean", "pass_at_5"]),
        )
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C. Shutting down gracefully...")
        # Get the current event loop
        loop = asyncio.get_running_loop()
        # Cancel all running tasks
        for task in asyncio.all_tasks(loop):
            task.cancel()
        # Wait for all tasks to be cancelled
        await asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True)
        print("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess terminated by user.")
