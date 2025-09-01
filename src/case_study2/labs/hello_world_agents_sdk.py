import os
from datetime import datetime
from agents import Agent, Runner, gen_trace_id, trace
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4o-mini"
OUTPUT = "./src/case_study2/labs/output/log_agent_hello_world.csv"


def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def check_installation():
    """Check if the correct packages are installed"""
    try:
        from agents import Agent, Runner

        print("✅ OpenAI Agents SDK is correctly installed!")
        return True
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nTo fix this:")
        print("1. Uninstall any conflicting packages:")
        print("   pip uninstall agents")
        print("2. Install the correct OpenAI Agents SDK:")
        print("   pip install openai-agents")
        print("3. Set your API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return False


def simple_hello_world():
    """Simple Hello World without sessions (most basic example)"""

    try:
        from agents import Agent, Runner, trace

        # Create a simple agent
        agent = Agent(
            name="Hello World Assistant",
            instructions="You are a friendly assistant. Greet users warmly and be helpful.",
            model=MODEL,
        )
        print("🤖 Simple Hello World - OpenAI Agents SDK")
        print("=" * 40)
        with trace("agent_sdk"):
            message = "Hello! Can you introduce yourself?"

            print("✅ Agent created successfully")

            # Run the agent synchronously (simpler than async)
            print("🏃 Running agent...")
            result = Runner.run_sync(agent, message)
            line = f"{time_now()}|{gen_trace_id()}|{message}|{result.last_agent}\n"
            print(line)
            with open(OUTPUT, "a") as f:
                f.write(line)

            # Display the result
            print("\n💬 Conversation:")
            print("-" * 30)
            print(f"You: Hello! Can you introduce yourself?")
            print(f"Assistant: {result.final_output}")

            print("\n✅ Simple conversation completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


def multi_turn_conversation():
    """Example with multiple turns (without sessions for simplicity)"""

    try:
        from agents import Agent, Runner

        print("\n🚀 Multi-turn Conversation")
        print("=" * 40)

        # Create agent
        agent = Agent(
            name="Chat Assistant",
            instructions="You are a helpful assistant. Be concise but friendly.",
            model="gpt-4",
        )

        # Simulate multiple turns by building context manually
        conversation_history = []

        messages = [
            "Hi, what's your name?",
            "Can you tell me a fun fact about space?",
            "That's interesting! What about the ocean?",
        ]

        for i, user_message in enumerate(messages, 1):
            print(f"\nTurn {i}:")
            print(f"You: {user_message}")

            # Build the full conversation context
            if conversation_history:
                # Include previous context in the message
                context = "\n".join(
                    [
                        f"User: {h['user']}\nAssistant: {h['assistant']}"
                        for h in conversation_history
                    ]
                )
                full_message = f"Previous conversation:\n{context}\n\nCurrent message: {user_message}"
            else:
                full_message = user_message

            result = Runner.run_sync(agent, full_message)

            print(f"Assistant: {result.final_output}")

            # Store for next turn
            conversation_history.append(
                {"user": user_message, "assistant": result.final_output}
            )

        print("\n✅ Multi-turn conversation completed!")

    except Exception as e:
        print(f"❌ Error in multi-turn example: {e}")


if __name__ == "__main__":
    # Check if everything is installed correctly
    if not check_installation():
        exit(1)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)

    # Run examples
    simple_hello_world()
    multi_turn_conversation()

    print("\n🎉 All examples completed!")
    print("\nNext steps:")
    print("- Try the examples in the OpenAI Agents SDK documentation")
    print("- Explore sessions, handoffs, and tools")
    print("- Check out: https://openai.github.io/openai-agents-python/")
