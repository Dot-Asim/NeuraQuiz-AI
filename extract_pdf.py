import PyPDF2
with open('AL2002_LabProject.pdf', 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

with open('pdf_content.txt', 'w', encoding='utf-8') as f:
    f.write(text)
