import io
import os
from typing import Union, Optional

import PIL
import fitz  # PyMuPDF
import PyPDF2
import tiktoken


def extract_text_and_images_with_positioning(pdf_path: os.PathLike, save_img_path: Optional[os.PathLike] = None):
    """
    Extracts text and images from a PDF file and returns them as a list of strings and PIL images.

    Args:
        pdf_path: The path to the PDF file.
        save_img_path: The path to save the images to. If None, images are not saved.

    Returns:
        all_blocks_pages: A list of lists of tuples, where each tuple contains the block's rectangle, type, and content.
        combined_content_pages: A list of lists of strings, where each string is the combined content of a page.
    """
    doc = fitz.open(pdf_path)
    combined_content_pages, all_blocks_pages = [], []

    for page_num, page in enumerate(doc):
        text_blocks = page.get_text("blocks")
        text_blocks.sort(
            key=lambda block: (block[1], block[0]))  # Sort primarily by vertical, then by horizontal position

        image_blocks = []
        for img_index, img in enumerate(page.get_images(full=True)):
            # Extracting and sorting image blocks requires getting their rectangles on the page
            xref = img[0]
            image_bytes = doc.extract_image(xref)["image"]
            img_rect = page.get_image_rects(xref)[0]  # Assuming one rect per image
            image_blocks.append((img_rect, image_bytes))
            if save_img_path:
                # Save the image
                image_filename = f"{save_img_path}/image_{page_num + 1}_{img_index + 1}.png"
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)

        # Sort image blocks like text blocks
        image_blocks.sort(key=lambda block: (block[0].y0, block[0].x0))

        # Merge and sort all blocks
        all_blocks = [(block[:4], 'text', block[4]) for block in text_blocks] + \
                     [(img_block[0], 'image', img_block[1]) for img_block in image_blocks]
        all_blocks.sort(key=lambda block: (block[0][1], block[0][0]))  # Sort by vertical then horizontal
        all_blocks_pages.append(all_blocks)
        combined_content = [block[2].strip() if block[1] == 'text' else block[2] if block[1] == 'image' else None for block in all_blocks]
        combined_content_pages.append(combined_content)

    doc.close()
    return all_blocks_pages, combined_content_pages


def extract_text_images(pdf_path, output_folder):
    """
    Extracts text and images from a PDF file and saves them to a folder.
    Args:
        pdf_path: The path to the PDF file.
        output_folder: The folder to save the extracted text and images to.

    Returns:

    """
    # Open the PDF file
    doc = fitz.open(pdf_path)

    # Make sure the output folder exists
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    text_filename = os.path.join(output_folder, "extracted_text.txt")
    with open(text_filename, "w") as text_file:
        image_count = 1  # Image counter

        for page_num, page in enumerate(doc):
            # Extract text from each page and write to the text file
            text = page.get_text("text")
            text_file.write(f"Page {page_num + 1}\n{text}\n")
            text_file.write("Images:\n")

            # Extract images
            image_list = page.get_images(full=True)
            for image_index, img in enumerate(image_list):
                # The image itself is the last item in the tuple
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Save the image
                image_filename = f"image_{page_num + 1}_{image_count}.png"
                image_filepath = os.path.join(output_folder, image_filename)
                with open(image_filepath, "wb") as image_file:
                    image_file.write(image_bytes)

                # Write a reference to the image in the text file
                text_file.write(f"[Image {image_count}] {image_filename}\n")
                image_count += 1

            text_file.write("\n")  # Add space between pages

    # Close the document
    doc.close()
    print(f"Extraction completed. Text and images are saved in '{output_folder}'.")


def extract_text(pdf_path, add_page_num: bool = False) -> list[str]:
    """
    Extracts text from a PDF file and returns it as a list of strings.
    Args:
        pdf_path: The path to the PDF file.
        add_page_num: Whether to add the page number to the beginning of each page's text.

    Returns:
        texts: A list of strings, where each string is the text from a page.
    """
    # Open the PDF file
    texts = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        # Iterate through each page and extract text
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            text = f"Page {page_num + 1}:\n{text}\n" if add_page_num else text + "\n"
            texts.append(text)
    return texts


def extract_text_and_fill_in_images(pdf_path, save_img_path=None, add_page_num: bool = False) -> list[Union[str, PIL.Image.Image]]:
    """
    Extracts text and images from a PDF file and returns them as a list of strings and PIL images.
    Args:
        pdf_path: The path to the PDF file.
        save_img_path: The path to save the images to. If None, images are not saved.
        add_page_num: Whether to add the page number to the beginning of each page's text.

    Returns:
        all_contents: A list of strings and PIL images, where each string is the text from a page and each image is a PIL image.
    """
    all_contents = []
    texts_from_pypdf = extract_text(pdf_path, add_page_num)
    blocks_from_pymupdf, contents_from_pymupdf = extract_text_and_images_with_positioning(pdf_path, save_img_path)

    for page_num, text in enumerate(texts_from_pypdf):
        all_contents.append(text)
        for i, block in enumerate(blocks_from_pymupdf[page_num]):
            if block[1] == "image":
                # img_cookie = {'mime_type': 'image/png', 'data': block[2]}
                img_cookie = PIL.Image.open(io.BytesIO(block[2]))
                all_contents.append(img_cookie)
    return all_contents


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def markdown_format_prompt(prompt):
    if type(prompt) == list:
        return "\n\n".join([f"**{message['role']}**: {message['content']}" for message in prompt])
    else:
        return prompt
