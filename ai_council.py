import requests
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# =======================
# Basic config
# =======================
# Ollama local chat endpoint + model tag pulled locally.
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1:8b"


@dataclass
class Agent:
    """Represents one council member and its simple performance stats."""
    name: str
    role: str
    system_prompt: str
    wins: int = 0
    total_answers: int = 0


class ConversationMemory:
    """
    Simple short-term memory for the council.

    Stores the last N (user, assistant) turns and can convert them into
    Ollama chat-style messages so each agent gets recent context.
    """
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.turns: List[Tuple[str, str]] = []  # (user_text, assistant_text)

    def add_turn(self, user_text: str, assistant_text: str):
        """Append a turn and keep only the most recent max_turns."""
        self.turns.append((user_text, assistant_text))
        # keep only the most recent max_turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert stored turns into chat messages: user, assistant, user, assistant, ..."""
        messages: List[Dict[str, str]] = []
        for user_text, assistant_text in self.turns:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        return messages


# Global council memory (shared across the session)
MEMORY = ConversationMemory(max_turns=10)


# Define your council members here
COUNCIL: List[Agent] = [
    Agent(
        name="Analyst",
        role="careful analyst",
        system_prompt=(
            "You are a calm, highly analytical assistant. You think step-by-step, avoid speculation, and clearly explain your reasoning."
        ),
    ),
    Agent(
        name="Optimist",
        role="optimistic strategist",
        system_prompt=(
            "You are optimistic but realistic. You look for upside and opportunities, while still mentioning key risks briefly."
        ),
    ),
    Agent(
        name="Skeptic",
        role="critical reviewer",
        system_prompt=(
            "You are a skeptical critic. You point out flaws, risks, and hidden assumptions, and you focus on what could go wrong."
        ),
    ),
]


# =======================
# LLM helpers
# =======================
def call_ollama(
    system_prompt: str,
    user_prompt: str,
    history_messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Call Ollama's /api/chat endpoint.

    Args:
        system_prompt: The system instruction for the model ("who you are").
        user_prompt: The current request (your constructed prompt).
        history_messages: Optional prior messages as [{"role": "...", "content": "..."}].

    Returns:
        The assistant's text response (trimmed).
    """
    if history_messages is None:
        history_messages = []

    # Build the chat payload: system -> history -> current user prompt
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": MODEL_NAME,
        "stream": False,  # keep it simple (no streaming parsing)
        "messages": messages,
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Ollama /api/chat returns: {"message": {"role": "...", "content": "..."}, ...}
    return data["message"]["content"].strip()


def ask_agent(agent: Agent, question: str, memory: ConversationMemory) -> str:
    """
    Ask a single agent the question using its system prompt.
    The agent sees the recent conversation history from MEMORY.
    """
    agent_prompt = (
        f"You are {agent.name}, a {agent.role} in a council of AIs.\n"
        f"You see the recent conversation above as context. "
        f"Answer the user's new question clearly and concisely.\n\n"
        f"User's current question: {question}"
    )
    history = memory.to_messages()
    answer = call_ollama(agent.system_prompt, agent_prompt, history_messages=history)
    agent.total_answers += 1
    return answer


def run_council(question: str, memory: ConversationMemory):
    """
    Ask all council members the question (with memory), collect their answers,
    then run a judge pass to pick the best answer.

    Returns:
        answers_by_agent: dict[agent_name] -> answer text
        winning_agent_name: name of the chosen agent
        winning_answer: the answer text
    """
    # 1 – Each agent answers, conditioned on conversation memory
    answers_by_agent: Dict[str, str] = {}
    for agent in COUNCIL:
        print(f"\n=== {agent.name} is thinking... ===")
        ans = ask_agent(agent, question, memory)
        answers_by_agent[agent.name] = ans
        print(f"{agent.name} answered.\n")

    # 2 – Judge: use the same model as a separate "meta-judge"
    judge_prompt = build_judge_prompt(question, answers_by_agent)
    judge_system = (
        "You are an impartial judge in an AI council.\n"
        "You receive a question and several candidate answers from different agents.\n"
        "Your task is to decide which answer is BEST overall.\n"
        "Rules:\n"
        "- Consider correctness, clarity, depth, and usefulness.\n"
        "- Respond ONLY with the agent's name, nothing else.\n"
        "- Do not explain your reasoning, do not add extra words."
    )

    # Judge does NOT use memory here to keep selection clean
    judge_output = call_ollama(judge_system, judge_prompt, history_messages=None)
    winning_agent_name = parse_winning_agent(judge_output, answers_by_agent.keys())
    if winning_agent_name is None:
        # fallback: pick first agent if judge fails
        winning_agent_name = next(iter(answers_by_agent.keys()))

    winning_answer = answers_by_agent[winning_agent_name]

    # Update win counter
    for agent in COUNCIL:
        if agent.name == winning_agent_name:
            agent.wins += 1
            break

    return answers_by_agent, winning_agent_name, winning_answer


def build_judge_prompt(question: str, answers_by_agent: Dict[str, str]) -> str:
    """
    Build a prompt listing all answers for the judge model.
    """
    text = f"Question:\n{question}\n\nAnswers:\n"
    for name, ans in answers_by_agent.items():
        text += f"--- Agent: {name} ---\n{ans}\n\n"
    text += (
        "Now, choose the single best answer overall.\n"
        "Reply with exactly the NAME of the winning agent and nothing else."
    )
    return text


def parse_winning_agent(judge_output: str, agent_names) -> Optional[str]:
    """
    Try to extract which agent the judge chose, based on the raw model output.
    We tolerate extra spaces and punctuation.
    """
    raw = judge_output.strip().lower()
    for name in agent_names:
        if name.lower() in raw:
            return name
    return None


# =======================
# Simple elimination logic (optional)
# =======================
def maybe_eliminate_and_replace(threshold_answers: int = 10, min_win_rate: float = 0.1):
    """
    Optional: After enough questions, you can 'cull' underperforming agents.

    Any agent with total_answers >= threshold_answers and
    win rate < min_win_rate will be replaced with a fresh 'Trainee' agent.
    """
    global COUNCIL
    new_council: List[Agent] = []
    eliminated = []

    for agent in COUNCIL:
        if agent.total_answers < threshold_answers:
            new_council.append(agent)
            continue

        win_rate = agent.wins / max(1, agent.total_answers)
        if win_rate < min_win_rate:
            eliminated.append((agent.name, win_rate))
        else:
            new_council.append(agent)

    # Replace eliminated agents with new generic ones
    for old_name, _ in eliminated:
        print(f"Eliminating underperforming agent: {old_name}")
        replacement = Agent(
            name=f"Trainee_{old_name}",
            role="new trainee council member",
            system_prompt=(
                "You are a fresh, eager AI trainee. Be concise, accurate, and helpful. "
                "You try to learn from better answers and improve over time."
            ),
        )
        new_council.append(replacement)
        print(f"Added replacement agent: {replacement.name}")

    COUNCIL = new_council


# =======================
# Interactive loop
# =======================
def main():
    print("=== Mini AI Council (Ollama, with memory) ===")
    print(f"Using model: {MODEL_NAME}")
    print("Type a question and press Enter. Type 'quit' to exit.\n")

    questions_asked = 0

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        questions_asked += 1

        answers_by_agent, winner_name, winner_answer = run_council(question, MEMORY)

        print("\n================= Council Answers =================")
        for name, ans in answers_by_agent.items():
            print(f"\n===== {name.upper()} =====")
            print(ans)

        print("\n================== Council Verdict =================")
        print(f"Winning agent: {winner_name}")
        print(f"\nAnswer:\n{winner_answer}")

        # Update memory with this turn (user question + winning answer)
        MEMORY.add_turn(question, winner_answer)

        # Optional elimination mechanic every N questions
        if questions_asked % 5 == 0:
            print("\n[Council maintenance] Checking for underperforming agents...")
            maybe_eliminate_and_replace(threshold_answers=10, min_win_rate=0.15)

        # Show quick stats
        print("\n[Stats]")
        for a in COUNCIL:
            win_rate = a.wins / max(1, a.total_answers)
            print(f"{a.name}: {a.wins} wins / {a.total_answers} answers (win rate {win_rate:.2%})")


if __name__ == "__main__":
    main()
