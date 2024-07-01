import os
from io import BytesIO, StringIO

import requests  # type: ignore
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

import s3_cache

BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
CACHE_NAMESPACE = "pdf_utils"
s3_cache.set_max_size(CACHE_NAMESPACE, 100)


def extract_text_from_pdf(pdf_url):
    cached_text = s3_cache.get_cache(CACHE_NAMESPACE, pdf_url)
    if cached_text:
        return cached_text

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

    s3_cache.set_cache(CACHE_NAMESPACE, pdf_url, text)
    return text
