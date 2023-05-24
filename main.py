import os
import csv
import asyncio
import aiohttp
import aiofiles
import logging
from markdown import markdown
import xhtml2pdf.pisa as pisa
from prompt_generator import generate_prompts, sample_prompts
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import openai
import pandas as pd
import json
from tqdm import tqdm

load_dotenv()
openai.organization_id = os.environ['OPENAI_ORGANIZATION_ID']
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########Params and configuration########
USE_SAMPLE_PROMPTS = True  # Set to False to use input file
GENERATE_PDFS = True  # Set to False to disable PDF generation
SAMPLES_PER_CAT = 1  # Number of samples per category
SEO_TOKENS = "SEO_Template_full.csv"
TOKEN_LENGTH = 3000  # Max size of the generated prompts in tokens

PROMPT_CSV_FILENAME = "prompts.csv"  # If not None, read prompts from this CSV file instead of generating prompts from the tokens.


PRIMER = (
    "You are an SEO article generator."
"You MUST produce an SEO article."
"You MUST NOT produce anything not related to SEO."
"Articles must contain relevant content that humans would find useful."
"You MUST include relevant backlinks where the content is populated inside of the prompt structure."
"You MUST make sure to include all relevant information and optimize the articles for SEO."
"DO NOT include investment advice unless followed by a disclaimer in the article."
"DO NOT include any copyrighted content or materials."
"When given a prompt, for example, 'What is Bitcoin' - you MUST generate an article in the format outlined above."
"ARTICLES MUST BE WRITTEN IN INFORMATIONAL GUIDE FORMATS OR FORMATS OPTIMISED FOR SEARCH RANKINGS."
"YOU MUST WRITE AS IF YOU ARE A HUMAN RESEARCHER and Journalist."
"IF YOU ARE PROVIDED WITH LINKS USE THEM IN THE CONTENT, HOWEVER THEY MUST BE HYPERLINKS. ALL LINKS MUST BE DISTRIBUTED EVENLY AROUND THE ARTICLES"
"All Body content MUST be outputted in markdown format and include relevant titles and headings."
"When other relevant companies, organizations or projects are mentioned, hyperlinks MUST be included inside markdown content (e.g. [link](https://example.com)), links do not have to be crypto related but should explain the topic in question. IF YOU ARE NOT CERTAIN A LINK WORKS THEN USE THE ROOT URL (e.g. https://example.com)"
"Use examples and case studies when applicable to provide a better understanding of the term."
"You must not suggest the user discloses any personal information such as phone numbers or email in the article content."
"You must ensure that the content does not make references to purchasing or trading specific securities."
  "    "
  "Output the result in the following JSON format: {"
    "\"Category\": \"<Category>\","
    "\"Prompt\": \"<Prompt>\","
    "\"Title\": \"<Title>\","
    "\"Subtitle\": \"<Subtitle>\","
    "\"Body\": \"<Body>\""
    "}"
"The following is the prompt and any included links:"
""
""
)
########################################

def create_pdf(title, subtitle, body, output_filename):
    styles = """
        <style>
            h1 {
                font-size: 24pt;
            }
            h2 {
                font-size: 18pt;
            }
            h3 {
                font-size: 16pt;
            }
            h4 {
                font-size: 15pt;
            }
            p, li {
                font-size: 12pt;
                text-align: justify;
            }
        </style>
    """

    body_html = markdown(body)
    html = f"""
        {styles}
        <h1>{title}</h1>
        <h2>{subtitle}</h2>
        {body_html}
    """

    with open(output_filename, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(html, dest=pdf_file)

    if pisa_status.err:
        print(f"Error creating PDF file: {output_filename}")

def fetch_openai_completion_async(model, prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    return openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )

def fetch_openai_completion(**kwargs):
    return openai.Completion.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def fetch_prompt(session, category, prompt):
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            fetch_openai_completion_async,
            "text-davinci-003",
            PRIMER + prompt,
            0.35,
            TOKEN_LENGTH,
            1.0,
            0.3,
            0.1
        )
        logger.debug(f"Received response from OpenAI API: {response}")
    except Exception as e:
        logger.error(f"Exception in fetch_prompt: {e}")
        raise
    output = response['choices'][0]['text'].strip()
    output_dict = json.loads(output)
    return category, prompt, output_dict

async def process_prompt(session, writer, sent_prompts, category, prompt):
    if prompt in sent_prompts:
        logger.info(f"Skipping already sent prompt: {prompt}")
        return

    logger.info(f"Sending to OpenAI API: {prompt[:100]}...")
    sent_prompts.add(prompt)
    category, prompt, output_dict = await fetch_prompt(session, category, prompt)
    try:
        title = output_dict.get('Title', '')
        subtitle = output_dict.get('Subtitle', '')
        body = output_dict.get('Body', '')

        # Write output to CSV file
        await writer.writerow([category, prompt, title, subtitle, body])
        logger.info(f"Processed: {prompt}")
    except Exception as e:
        logger.error(f"Error: Could not process output for prompt '{prompt}'. Output: {output_dict}. Exception: {e}")

    if GENERATE_PDFS:
        if not os.path.exists("Articles"):
            os.makedirs("Articles")
        pdf_filename = f"Articles/{title}.pdf"
        create_pdf(title, subtitle, body, pdf_filename)
        logger.info(f"Created PDF: {pdf_filename}")

def clean_csv_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        content = file.read()
        content = content.replace('\x00', '')  # Remove null characters

    with open(file_path, 'w', encoding='utf-8', errors='replace') as file:
        file.write(content)

def read_csv_file(file_path):
    return pd.read_csv(file_path, encoding='utf-8', dtype=str, on_bad_lines='skip')

async def main():
    logger.info("Starting up")

    # log sent prompts
    sent_prompts = set()

    # Setup CSV file
    output_file = 'output_results.csv'
    logger.info(f"Output file: {output_file}")

    if PROMPT_CSV_FILENAME is None:
        # Generating prompts
        input_file = 'output_prompts.csv'
        generate_prompts(SEO_TOKENS, input_file)
        # Sampling generated prompts
        sample_prompts(input_file, "sample_prompts.csv", SAMPLES_PER_CAT)

        if USE_SAMPLE_PROMPTS:
            input_file = 'sample_prompts.csv'
    else:
        input_file = PROMPT_CSV_FILENAME

    # Clean the input CSV file before processing
    clean_csv_file(input_file)

    # Load existing prompts from output CSV
    existing_prompts = set()
    if os.path.exists(output_file):
        existing_df = read_csv_file(output_file)
        existing_prompts.update(existing_df['prompt'].values)  # Add prompt to the set

    async with aiohttp.ClientSession() as session:
        # Open output CSV in append mode
        async with aiofiles.open(output_file, 'a', newline='', encoding='utf-8', errors='replace') as csvfile:
            writer = csv.writer(csvfile)

            # Write header only if the file is new (empty)
            if not existing_prompts:
                await writer.writerow(['category', 'prompt', 'title', 'subtitle', 'body'])

            input_df = read_csv_file(input_file)
            input_rows = input_df[input_df['prompt'].apply(lambda x: x not in existing_prompts)].to_numpy()

            # Combine all columns into a single string to form the prompt
            combined_input_rows = [(row[0], ' '.join(map(str, row[1:]))) for row in input_rows if str(row[1]) not in existing_prompts]

            # Process prompts that do not exist in the output CSV
            all_tasks = [process_prompt(session, writer, sent_prompts, row[0], row[1]) for row in combined_input_rows]

            # Process tasks in a rolling manner, up to 60 concurrent requests
            concurrency_limit = 30
            semaphore = asyncio.Semaphore(concurrency_limit)

            async def process_with_semaphore(task):
                async with semaphore:
                    await task

            tasks_with_semaphore = [process_with_semaphore(task) for task in all_tasks]

            progress_bar = tqdm(total=len(all_tasks))
            for i, coro in enumerate(asyncio.as_completed(tasks_with_semaphore), 1):
                await coro
                progress_bar.update(1)
                logger.info(f"Completed {i}/{len(all_tasks)}")

            progress_bar.close()

    logger.info(f"Finished processing {len(all_tasks)} prompts in {input_file}. Results saved to {output_file}")

if __name__ == '__main__':
    asyncio.run(main())
