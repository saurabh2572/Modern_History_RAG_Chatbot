from pathlib import Path
import logging
import json
import re
import yaml

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------
# Logging
# ---------------------------------------------------
Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename="logs/all_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("all_errors")

# ---------------------------------------------------
# Load Configuration
# ---------------------------------------------------
with open("./config.yaml", "r", encoding="utf-8") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        logger.error(f"Error loading config.yaml: {exc}")
        raise

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
RAW_DIR = Path(CONFIG["PATH"]["RAW_DATA_DIR"])
OUTPUT_FILE = Path(CONFIG["PATH"]["CHUNKED_DATA"])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# Text Splitter
# ---------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=250,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        "; ",
        ", ",
        " ",
        "",
    ],
)

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_chunk_text(text: str) -> str:
    return " ".join(text.split())


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def detect_heading_type(line: str):
    line = line.strip()

    if not line:
        return None

    lower = line.lower()

    chapter_patterns = [
        r"^chapter\s+\d+",
        r"^chapter\s+[ivxlcdm]+",
        r"^part\s+\d+",
        r"^part\s+[ivxlcdm]+",
        r"^book\s+\d+",
        r"^book\s+[ivxlcdm]+",
    ]

    section_patterns = [
        r"^\d+\.\s+[A-Z]",
        r"^\d+\.\d+\s+[A-Z]",
        r"^[A-Z][A-Z\s,'’:-]{5,}$",
        r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,8}$",
    ]

    for pattern in chapter_patterns:
        if re.match(pattern, lower):
            return "chapter"

    if len(line.split()) <= 12:
        for pattern in section_patterns:
            if re.match(pattern, line):
                return "section"

    return None


def build_sections(pages, source, book_title):
    sections = []

    current_chapter = None
    current_section = None
    current_header = None
    current_text = []
    current_page_start = None

    for page in pages:
        page_number = page["page_number"]
        page_text = clean_text(page["text"])

        if not page_text:
            continue

        for raw_line in page_text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            heading_type = detect_heading_type(line)

            if heading_type:
                if current_text:
                    sections.append(
                        {
                            "source": source,
                            "book_title": book_title,
                            "chapter": current_chapter,
                            "section": current_section,
                            "header": current_header,
                            "page_start": current_page_start,
                            "page_end": page_number,
                            "text": "\n".join(current_text).strip(),
                        }
                    )

                current_text = [line]
                current_page_start = page_number
                current_header = line

                if heading_type == "chapter":
                    current_chapter = line
                    current_section = None

                elif heading_type == "section":
                    current_section = line

            else:
                if current_page_start is None:
                    current_page_start = page_number

                current_text.append(line)

    if current_text:
        sections.append(
            {
                "source": source,
                "book_title": book_title,
                "chapter": current_chapter,
                "section": current_section,
                "header": current_header,
                "page_start": current_page_start,
                "page_end": pages[-1]["page_number"],
                "text": "\n".join(current_text).strip(),
            }
        )

    return sections


# ---------------------------------------------------
# Extract, Detect Sections & Chunk PDFs into JSON
# ---------------------------------------------------
try:
    pdf_files = list(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {RAW_DIR}")

    all_chunks = []
    global_chunk_index = 1

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        reader = PdfReader(pdf_path)

        pages = []

        for page_index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()

            if page_text:
                pages.append(
                    {
                        "page_number": page_index,
                        "text": page_text,
                    }
                )

        sections = build_sections(
            pages=pages,
            source=pdf_path.name,
            book_title=pdf_path.stem,
        )

        for section_index, section_data in enumerate(sections, start=1):
            chunks = splitter.split_text(section_data["text"])

            for local_chunk_index, chunk in enumerate(chunks, start=1):
                cleaned_chunk = normalize_chunk_text(chunk)

                if not cleaned_chunk:
                    continue

                chunk_data = {
                    "id": f"{pdf_path.stem}_{global_chunk_index:06d}",
                    "source": pdf_path.name,
                    "book_title": pdf_path.stem,
                    "text": cleaned_chunk,
                    "metadata": {
                        "file_name": pdf_path.name,
                        "file_path": str(pdf_path),
                        "book_title": pdf_path.stem,
                        "chapter": section_data["chapter"],
                        "section": section_data["section"],
                        "header": section_data["header"],
                        "section_index": section_index,
                        "chunk_index": global_chunk_index,
                        "local_chunk_index": local_chunk_index,
                        "page_start": section_data["page_start"],
                        "page_end": section_data["page_end"],
                        "char_count": len(cleaned_chunk),
                        "estimated_token_count": estimate_tokens(cleaned_chunk),
                        "chunk_size": 1800,
                        "chunk_overlap": 250,
                    },
                }

                all_chunks.append(chunk_data)
                global_chunk_index += 1

        print(f"Created chunks from {len(sections)} detected sections")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        json.dump(all_chunks, outfile, ensure_ascii=False, indent=2)

    print(f"\nJSON chunks saved successfully to:\n{OUTPUT_FILE}")
    print(f"Total chunks created: {len(all_chunks)}")

except Exception as e:
    logger.error(f"Error during PDF extraction/chunking: {e}", exc_info=True)
    raise