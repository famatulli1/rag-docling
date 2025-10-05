import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_vision_enabled_check():
    """Test vision enabled configuration"""
    from core_logic.vision_config import get_vision_enabled

    # Test enabled
    with patch.dict(os.environ, {'ENABLE_PICTURE_DESCRIPTION': 'true'}):
        assert get_vision_enabled() is True

    # Test disabled
    with patch.dict(os.environ, {'ENABLE_PICTURE_DESCRIPTION': 'false'}):
        assert get_vision_enabled() is False

    # Test default (should be false)
    with patch.dict(os.environ, {}, clear=True):
        assert get_vision_enabled() is False


def test_vision_pipeline_options_disabled():
    """Test PdfPipelineOptions when vision is disabled"""
    from core_logic.vision_config import get_vision_pipeline_options

    with patch.dict(os.environ, {'ENABLE_PICTURE_DESCRIPTION': 'false'}):
        options = get_vision_pipeline_options()
        assert options.do_picture_description is False


def test_vision_pipeline_options_enabled():
    """Test PdfPipelineOptions when vision is enabled"""
    from core_logic.vision_config import get_vision_pipeline_options

    with patch.dict(os.environ, {
        'ENABLE_PICTURE_DESCRIPTION': 'true',
        'PICTURE_DESCRIPTION_PROMPT': 'Custom prompt',
        'PICTURE_AREA_THRESHOLD': '0.1',
        'PICTURE_BATCH_SIZE': '8'
    }):
        options = get_vision_pipeline_options()
        assert options.do_picture_description is True
        assert options.picture_description_options.prompt == 'Custom prompt'
        assert options.picture_description_options.picture_area_threshold == 0.1
        assert options.picture_description_options.batch_size == 8
        assert options.generate_picture_images is True
        assert options.images_scale == 2.0


@patch('core_logic.document_processor.DocumentConverter')
@patch('core_logic.document_processor.get_vision_enabled')
def test_process_pdf_with_vision_enabled(mock_vision_enabled, mock_converter_class):
    """Test PDF processing with vision enabled"""
    from core_logic.document_processor import process_document

    # Mock vision enabled
    mock_vision_enabled.return_value = True

    # Mock DocumentConverter
    mock_converter = MagicMock()
    mock_converter_class.return_value = mock_converter

    mock_result = MagicMock()
    mock_document = MagicMock()
    mock_document.export_to_markdown.return_value = "# Document\n\n![Image](image.png)\n\nImage description: A beautiful sunset."
    mock_result.document = mock_document
    mock_converter.convert.return_value = mock_result

    # Process a PDF
    with patch.dict(os.environ, {
        'ENABLE_PICTURE_DESCRIPTION': 'true',
        'PICTURE_DESCRIPTION_PROMPT': 'Describe this image'
    }):
        result = process_document("/test/document.pdf")

    # Verify DocumentConverter was used
    mock_converter_class.assert_called_once()
    mock_converter.convert.assert_called_once_with("/test/document.pdf")

    # Verify result contains image description
    assert "Image description" in result
    assert "![Image]" in result


@patch('core_logic.document_processor.DoclingLoader')
@patch('core_logic.document_processor.get_vision_enabled')
def test_process_pdf_with_vision_disabled(mock_vision_enabled, mock_loader_class):
    """Test PDF processing with vision disabled"""
    from core_logic.document_processor import process_document

    # Mock vision disabled
    mock_vision_enabled.return_value = False

    # Mock DoclingLoader
    mock_loader = MagicMock()
    mock_loader_class.return_value = mock_loader

    mock_doc = MagicMock()
    mock_doc.page_content = "# Document\n\nRegular text content."
    mock_loader.load.return_value = [mock_doc]

    # Process a PDF
    with patch.dict(os.environ, {'ENABLE_PICTURE_DESCRIPTION': 'false'}):
        result = process_document("/test/document.pdf")

    # Verify DoclingLoader was used (not DocumentConverter)
    mock_loader_class.assert_called_once()
    assert "Regular text content" in result


@patch('core_logic.document_processor.DocumentConverter')
@patch('core_logic.document_processor.get_vision_enabled')
def test_chunk_pdf_with_vision_metadata(mock_vision_enabled, mock_converter_class):
    """Test that vision chunks include metadata"""
    from core_logic.document_processor import chunk_document_from_file

    # Mock vision enabled
    mock_vision_enabled.return_value = True

    # Mock DocumentConverter
    mock_converter = MagicMock()
    mock_converter_class.return_value = mock_converter

    mock_result = MagicMock()
    mock_document = MagicMock()
    mock_document.export_to_markdown.return_value = "Text with image ![Image](img.png) description."
    mock_result.document = mock_document
    mock_converter.convert.return_value = mock_result

    # Mock HybridChunker
    mock_chunk = MagicMock()
    mock_chunk.text = "Text with image ![Image](img.png) description."

    with patch('core_logic.document_processor.HybridChunker') as mock_chunker_class:
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mock_chunk]
        mock_chunker_class.return_value = mock_chunker

        with patch.dict(os.environ, {'ENABLE_PICTURE_DESCRIPTION': 'true'}):
            chunks = chunk_document_from_file("/test/document.pdf")

    # Verify chunks have vision metadata
    assert len(chunks) == 1
    assert chunks[0]['metadata']['has_vision_content'] is True  # Contains markdown image syntax
    assert "![Image]" in chunks[0]['text']


@patch('core_logic.document_processor.DoclingLoader')
@patch('core_logic.document_processor.get_vision_enabled')
def test_chunk_non_pdf_ignores_vision(mock_vision_enabled, mock_loader_class):
    """Test that non-PDF files don't use vision even if enabled"""
    from core_logic.document_processor import chunk_document_from_file

    # Mock vision enabled
    mock_vision_enabled.return_value = True

    # Mock DoclingLoader for DOCX
    mock_loader = MagicMock()
    mock_loader_class.return_value = mock_loader

    mock_doc = MagicMock()
    mock_doc.page_content = "Regular DOCX content"
    mock_doc.metadata = {}
    mock_loader.load.return_value = [mock_doc]

    # Process a DOCX (not PDF)
    with patch.dict(os.environ, {'ENABLE_PICTURE_DESCRIPTION': 'true'}):
        chunks = chunk_document_from_file("/test/document.docx")

    # Verify DoclingLoader was used (vision only for PDFs)
    mock_loader_class.assert_called_once()
    assert len(chunks) == 1
    assert "Regular DOCX content" in chunks[0]['text']
