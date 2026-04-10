import pymupdf


def read_pdf_file(pdf_file):
    doc = pymupdf.open(pdf_file)
    for page in doc:
        text = page.get_text()

    return text