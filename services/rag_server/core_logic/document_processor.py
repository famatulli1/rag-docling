from pathlib import Path
from typing import List, Dict
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from core_logic.vision_config import get_vision_enabled, get_vision_pipeline_options
import logging

logger = logging.getLogger(__name__)

# Docling supports many more formats than the previous implementation
SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.pdf', '.docx', '.pptx', '.xlsx',
    '.html', '.htm', '.asciidoc', '.adoc'
}

# Tokenizer for chunking - use sentence-transformers/all-MiniLM-L6-v2 (compatible with nomic-embed-text)
# This is a HuggingFace model that HybridChunker can download
EMBED_MODEL_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"

def process_document(file_path: str) -> str:
    logger.info(f"[DOCLING] Starting to process document: {file_path}")
    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()
    logger.info(f"[DOCLING] File extension: {extension}")

    if extension not in SUPPORTED_EXTENSIONS:
        error_msg = f"Unsupported file type: {extension}"
        logger.error(f"[DOCLING] {error_msg}")
        raise ValueError(error_msg)

    # For PDFs with vision enabled, use DocumentConverter for image description
    if extension == '.pdf' and get_vision_enabled():
        logger.info(f"[DOCLING] PDF with vision enabled - using DocumentConverter")
        pipeline_options = get_vision_pipeline_options()

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

        result = converter.convert(str(file_path))
        full_text = result.document.export_to_markdown()
        logger.info(f"[DOCLING] Extracted {len(full_text)} characters with vision (includes image descriptions)")
        return full_text

    # Use DoclingLoader with MARKDOWN export for other formats or PDFs without vision
    logger.info(f"[DOCLING] Initializing DoclingLoader with MARKDOWN export")
    loader = DoclingLoader(
        file_path=str(file_path),
        export_type=ExportType.MARKDOWN
    )

    logger.info(f"[DOCLING] Loading document with DoclingLoader")
    docs = loader.load()
    logger.info(f"[DOCLING] Loaded {len(docs)} document sections")

    if not docs:
        error_msg = f"Could not load document: {file_path}"
        logger.error(f"[DOCLING] {error_msg}")
        raise ValueError(error_msg)

    # Combine all document chunks into single text
    # DoclingLoader may split into multiple docs, so join them
    full_text = "\n\n".join(doc.page_content for doc in docs)
    logger.info(f"[DOCLING] Extracted {len(full_text)} characters of text")

    return full_text


def chunk_document_from_file(file_path: str, chunk_size: int = 500) -> List[Dict]:
    logger.info(f"[DOCLING] chunk_document_from_file called for: {file_path}")
    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()
    logger.info(f"[DOCLING] File extension: {extension}, chunk_size: {chunk_size}")

    if extension not in SUPPORTED_EXTENSIONS:
        error_msg = f"Unsupported file type: {extension}"
        logger.error(f"[DOCLING] {error_msg}")
        raise ValueError(error_msg)

    # For PDFs with vision enabled, use DocumentConverter then chunk the result
    if extension == '.pdf' and get_vision_enabled():
        logger.info(f"[DOCLING] PDF with vision enabled - using DocumentConverter for chunking")
        pipeline_options = get_vision_pipeline_options()

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

        result = converter.convert(str(file_path))

        # Log picture elements detected in the document
        if hasattr(result.document, 'pictures'):
            logger.info(f"[DOCLING] Document has {len(result.document.pictures)} picture elements")

        full_text = result.document.export_to_markdown()
        logger.info(f"[DOCLING] Extracted {len(full_text)} characters with vision")

        # Log if images/pictures were found and described
        image_count = full_text.count('![')
        logger.info(f"[DOCLING] Found {image_count} image descriptions in markdown")
        if image_count > 0:
            # Log first 500 chars to see image descriptions
            logger.info(f"[DOCLING] Document preview with images: {full_text[:500]}...")
        else:
            # Log why no images were found
            logger.warning(f"[DOCLING] No image descriptions generated despite vision being enabled. Check PICTURE_AREA_THRESHOLD setting.")

        # Use HybridChunker to chunk the vision-enriched markdown
        chunker = HybridChunker(
            tokenizer=EMBED_MODEL_TOKENIZER,
            max_tokens=chunk_size
        )

        # Chunk the full document text
        doc_chunks = list(chunker.chunk(result.document))
        logger.info(f"[DOCLING] HybridChunker created {len(doc_chunks)} chunks from vision-enriched document")

        # Convert to our format
        chunks = []
        for i, chunk in enumerate(doc_chunks):
            text = chunk.text.strip()
            if text:
                chunk_preview = text[:80] + "..." if len(text) > 80 else text
                logger.debug(f"[DOCLING] Chunk {i}: {len(text)} chars - {chunk_preview}")

                # Add metadata indicating if this chunk likely contains image descriptions
                metadata = {
                    "chunk_index": i,
                    "has_vision_content": "![" in text  # Markdown image syntax indicates vision content
                }

                chunks.append({
                    'text': text,
                    'metadata': metadata
                })

        logger.info(f"[DOCLING] Created {len(chunks)} vision-enriched chunks")
        return chunks

    # Use DoclingLoader with DOC_CHUNKS for efficient chunking (non-PDF or vision disabled)
    logger.info(f"[DOCLING] Initializing DoclingLoader with DOC_CHUNKS export")
    logger.info(f"[DOCLING] HybridChunker settings: tokenizer={EMBED_MODEL_TOKENIZER}, max_tokens={chunk_size}")
    loader = DoclingLoader(
        file_path=str(file_path),
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(
            tokenizer=EMBED_MODEL_TOKENIZER,
            max_tokens=chunk_size
        )
    )

    logger.info(f"[DOCLING] Loading and chunking document")
    docs = loader.load()
    logger.info(f"[DOCLING] DoclingLoader returned {len(docs)} chunks")

    # Convert LangChain documents to our format
    chunks = []
    for i, doc in enumerate(docs):
        if doc.page_content.strip():
            chunk_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
            logger.debug(f"[DOCLING] Chunk {i}: {len(doc.page_content)} chars - {chunk_preview}")
            chunks.append({
                'text': doc.page_content,
                'metadata': doc.metadata
            })

    logger.info(f"[DOCLING] Created {len(chunks)} valid chunks (filtered out empty chunks)")
    return chunks


def extract_metadata(file_path: str) -> Dict[str, str]:
    file_path_obj = Path(file_path)

    return {
        "file_name": file_path_obj.name,
        "file_type": file_path_obj.suffix,
        "path": str(file_path_obj.parent)
    }
