"""Unit tests for LLM client setup (create_llm_client).

Tests that ChatNVIDIA is initialized with the correct model parameter
and that structured output binding is applied via .with_structured_output().

Requirements: 4.2, 4.3
"""

from unittest.mock import MagicMock, patch

from config import ActionModel


class TestCreateLlmClientDefaultModel:
    """Test that create_llm_client initializes ChatNVIDIA with the default model."""

    @patch("llm_client.ChatNVIDIA")
    def test_initializes_with_default_model(self, mock_chat_nvidia):
        """ChatNVIDIA is called with model='meta/llama3-8b-instruct' by default."""
        mock_instance = MagicMock()
        mock_chat_nvidia.return_value = mock_instance
        mock_instance.with_structured_output.return_value = MagicMock()

        from llm_client import create_llm_client

        create_llm_client()

        mock_chat_nvidia.assert_called_once_with(model="meta/llama-3.1-8b-instruct")


class TestCreateLlmClientCustomModel:
    """Test that create_llm_client initializes ChatNVIDIA with a custom model."""

    @patch("llm_client.ChatNVIDIA")
    def test_initializes_with_custom_model(self, mock_chat_nvidia):
        """ChatNVIDIA is called with the provided custom model identifier."""
        mock_instance = MagicMock()
        mock_chat_nvidia.return_value = mock_instance
        mock_instance.with_structured_output.return_value = MagicMock()

        from llm_client import create_llm_client

        create_llm_client(model="nvidia/llama-3.1-nemotron-70b-instruct")

        mock_chat_nvidia.assert_called_once_with(
            model="nvidia/llama-3.1-nemotron-70b-instruct"
        )


class TestCreateLlmClientStructuredOutput:
    """Test that structured output binding is applied with ActionModel."""

    @patch("llm_client.ChatNVIDIA")
    def test_with_structured_output_called_with_action_model(self, mock_chat_nvidia):
        """with_structured_output is called with ActionModel on the ChatNVIDIA instance."""
        mock_instance = MagicMock()
        mock_chat_nvidia.return_value = mock_instance
        mock_instance.with_structured_output.return_value = MagicMock()

        from llm_client import create_llm_client

        create_llm_client()

        mock_instance.with_structured_output.assert_called_once_with(ActionModel)

    @patch("llm_client.ChatNVIDIA")
    def test_returns_structured_llm_result(self, mock_chat_nvidia):
        """create_llm_client returns the result of with_structured_output()."""
        mock_instance = MagicMock()
        mock_chat_nvidia.return_value = mock_instance
        expected_structured_llm = MagicMock(name="structured_llm")
        mock_instance.with_structured_output.return_value = expected_structured_llm

        from llm_client import create_llm_client

        result = create_llm_client()

        assert result is expected_structured_llm
