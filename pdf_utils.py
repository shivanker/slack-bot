import os
from io import BytesIO, StringIO

import requests  # type: ignore
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
cache: dict[str, str] = {}


def extract_text_from_pdf(pdf_url):
    if pdf_url in cache:
        return cache[pdf_url]
    response = requests.get(pdf_url, headers={"Authorization": f"Bearer {BOT_TOKEN}"})
    if response.status_code != 200:
        raise Exception(
            f"Error downloading file [{pdf_url}]. HTTP request status: {response.status_code}"
        )
    pdf_content = BytesIO(response.content)
    parser = PDFParser(pdf_content)
    document = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    text_output = StringIO()
    device = TextConverter(rsrcmgr, text_output, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(document):
        interpreter.process_page(page)

    text = text_output.getvalue()
    device.close()
    text_output.close()

    cache[pdf_url] = text
    if len(cache) > 10:
        oldest_url = next(iter(cache))
        del cache[oldest_url]
    return text
