import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

@patch('core_logic.llm_handler.get_openai_client')
def test_generate_response_with_context(mock_get_client):
    """Generate LLM response using context from retrieved documents"""
    from core_logic.llm_handler import generate_response

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Mock OpenAI ChatCompletion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = 'This is the generated answer based on the context.'
    mock_client.chat.completions.create.return_value = mock_response

    query = "What is the capital of France?"
    context = "France is a country in Europe. Its capital is Paris."

    response = generate_response(query, context)

    assert response is not None
    assert 'generated answer' in response.lower()
    mock_client.chat.completions.create.assert_called_once()

    # Verify the prompt includes both query and context
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs.get('messages', [])
    assert len(messages) > 0
    prompt = messages[0]['content']
    assert query.lower() in prompt.lower()
    assert context.lower() in prompt.lower()

def test_construct_prompt_with_context():
    """Construct proper RAG prompt with query and context"""
    from core_logic.llm_handler import construct_prompt

    query = "What is RAG?"
    context_docs = [
        "RAG stands for Retrieval Augmented Generation.",
        "It combines retrieval with LLM generation."
    ]

    prompt = construct_prompt(query, context_docs)

    assert query in prompt
    assert all(doc in prompt for doc in context_docs)
    # Should have instruction to use context
    assert 'context' in prompt.lower() or 'information' in prompt.lower()

def test_construct_prompt_french():
    """Test French prompt construction"""
    import os
    from core_logic.llm_handler import construct_prompt_balanced_fr

    query = "Quels sont les types d'airbag ?"
    context_docs = [
        "La Tesla Model Y dispose de plusieurs airbags.",
        "Il y a des airbags frontaux et latéraux."
    ]

    prompt = construct_prompt_balanced_fr(query, context_docs)

    assert query in prompt
    assert all(doc in prompt for doc in context_docs)
    # Should have French instructions
    assert 'Réponds' in prompt or 'réponse' in prompt.lower()
    assert 'contexte' in prompt.lower()

def test_language_routing():
    """Test that prompts route correctly based on RESPONSE_LANGUAGE"""
    import os
    from unittest.mock import patch
    from core_logic.llm_handler import construct_prompt

    query = "Test question"
    context_docs = ["Test context"]

    # Test French routing
    with patch.dict(os.environ, {'RESPONSE_LANGUAGE': 'fr'}):
        prompt_fr = construct_prompt(query, context_docs)
        assert 'Réponds' in prompt_fr or 'Question' in prompt_fr

    # Test English routing
    with patch.dict(os.environ, {'RESPONSE_LANGUAGE': 'en'}):
        prompt_en = construct_prompt(query, context_docs)
        assert 'Answer' in prompt_en or 'Question' in prompt_en

@patch('core_logic.llm_handler.get_openai_client')
def test_generate_response_without_context(mock_get_client):
    """Generate response when no relevant context is found"""
    from core_logic.llm_handler import generate_response

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = 'I don\'t have enough information to answer that question.'
    mock_client.chat.completions.create.return_value = mock_response

    query = "What is the meaning of life?"
    context = ""  # No context

    response = generate_response(query, context)

    assert response is not None
    assert 'information' in response.lower() or 'answer' in response.lower()

@patch('core_logic.llm_handler.get_openai_client')
def test_llm_uses_correct_model(mock_get_client):
    """Verify LLM uses the configured model"""
    from core_logic.llm_handler import generate_response

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = 'Test response'
    mock_client.chat.completions.create.return_value = mock_response

    generate_response("test query", "test context")

    call_args = mock_client.chat.completions.create.call_args
    model = call_args.kwargs.get('model', '')
    assert model is not None
    assert len(model) > 0  # Should have a model specified

@patch('core_logic.llm_handler.get_openai_client')
def test_handle_llm_error(mock_get_client):
    """Handle errors from OpenAI API gracefully"""
    from core_logic.llm_handler import generate_response

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Simulate connection error
    mock_client.chat.completions.create.side_effect = Exception("Connection failed")

    with pytest.raises(Exception, match="Connection failed"):
        generate_response("test query", "test context")
