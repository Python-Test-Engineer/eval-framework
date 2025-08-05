import pytest
from src.openai_agent_logging import SimpleAgent


class TestSimpleAgent:
    """Test suite for the SimpleAgent"""

    @pytest.fixture
    def agent(self):
        """Create a SimpleAgent instance for testing"""
        return SimpleAgent()

    def test_process_alpha_data(self, agent):
        """Test processing alpha data"""
        result = agent.run("Process this alpha data")
        assert result == "Tool A processed: Process this alpha data"

    def test_handle_beta_information(self, agent):
        """Test handling beta information"""
        result = agent.run("Handle beta information")
        assert result == "Tool B processed: Handle beta information"

    def test_work_with_gamma_values(self, agent):
        """Test working with gamma values"""
        result = agent.run("Work with gamma values")
        assert result == "Tool C processed: Work with gamma values"

    def test_random_input(self, agent):
        """Test random input processing"""
        result = agent.run("Random input")
        assert result == "Tool A processed: Random input"


# # Alternative: Test all cases in a single parameterized test
# @pytest.mark.parametrize("input_text,expected_output", [
#     ("Process this alpha data", "Tool A processed: Process this alpha data"),
#     ("Handle beta information", "Tool B processed: Handle beta information"),
#     ("Work with gamma values", "Tool C processed: Work with gamma values"),
#     ("Random input", "Tool A processed: Random input")
# ])
# def test_agent_routing(input_text, expected_output):
#     """Parameterized test for agent routing logic"""
#     agent = SimpleAgent()
#     result = agent.run(input_text)
#     assert result == expected_output


# # Individual test functions matching the original test cases
# def test_alpha_case():
#     """Test case: Process this alpha data"""
#     agent = SimpleAgent()
#     result = agent.run("Process this alpha data")
#     assert result == "Tool A processed: Process this alpha data"


# def test_beta_case():
#     """Test case: Handle beta information"""
#     agent = SimpleAgent()
#     result = agent.run("Handle beta information")
#     assert result == "Tool B processed: Handle beta information"


# def test_gamma_case():
#     """Test case: Work with gamma values"""
#     agent = SimpleAgent()
#     result = agent.run("Work with gamma values")
#     assert result == "Tool C processed: Work with gamma values"


# def test_random_case():
#     """Test case: Random input"""
#     agent = SimpleAgent()
#     result = agent.run("Random input")
#     assert result == "Tool A processed: Random input"
