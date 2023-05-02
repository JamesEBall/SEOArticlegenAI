import os
import csv
import re
import asyncio
import aiohttp
import aiofiles
import logging
from markdown import markdown
import xhtml2pdf.pisa as pisa
from markdown import markdown
from prompt_generator import generate_prompts, sample_prompts
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import openai

load_dotenv()
openai.organization_id = os.environ['OPENAI_ORGANIZATION_ID']
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########Params and configuration########
USE_SAMPLE_PROMPTS = True  # Set to False to use input file
GENERATE_PDFS = True  # Set to False to disable PDF generation
SAMPLES_PER_CAT = 1  # Number of samples per category
SEO_TOKENS = "example_SEO_Template.csv"


PRIMER = (
    "You are ChatGPT, an AI language model, and your task is to create 300-500 word "
    "articles for a knowledge base. Please prioritize "
    "the accuracy and relevance of the information in the articles. If you do not have enough "
    "information to fill in the prompt, provide a brief explanation and suggest alternative "
    "resources for further research. While SEO optimization is important, avoid keyword-stuffing "
    "and focus on providing valuable information to the reader.\\n\\n"
    "If the prompted question requests information on an impossible task, mention that in "
    "the article, explain why it's not possible, and suggest an alternative task for the user. "
    "The article must be outputted in Markdown and include at least one (but ideally at least 3) relevant external URL either inside of the article and the end."
    "for additional information. Avoid any content that could be misconstrued as investment advice.\\n\\n"
    "Use examples and case studies, when applicable, to provide a better understanding of the topic.\\n\\n"
    "Output the prompt in the following format: Category:, Prompt:, Title:, Subtitle:, and Body:. "
    "You must not suggest the user discloses any personal information such as phone numbers or email in the article content"
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
            800,
            1.0,
            0.3,
            0.1
        )
    except Exception as e:
        print(f"Exception in fetch_prompt: {e}")
        raise
    return category, prompt, response['choices'][0]['text'].strip()




# Update the main function with additional logging
async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for category, prompt_list in PROMPTS.items():
            for prompt in prompt_list:
                print(f"Creating task for: {category} - {prompt}")
                tasks.append(asyncio.ensure_future(process_prompt(session, category, prompt)))
        print("All tasks created")
        await asyncio.gather(*tasks)


async def process_prompt(session, writer, category, prompt):
    logger.info(f"Sending to OpenAI API: {prompt}")
    category, prompt, output = await fetch_prompt(session, category, prompt)
    try:
        output_dict = {}
        for key, value in re.findall(r'(\w+):\s*(.*(?:\n(?!\w+:).*)*)', output):
            output_dict[key.strip()] = value.strip()

        title = output_dict.get('Title', '')
        subtitle = output_dict.get('Subtitle', '')
        body = output_dict.get('Body', '')

        # Write output to CSV file
        await writer.writerow([category, prompt, title, subtitle, body])
        logger.info(f"Processed: {prompt}")
    except Exception as e:
        logger.error(f"Error: Could not process output for prompt '{prompt}'. Output: {output}. Exception: {e}")

    if GENERATE_PDFS:
        if not os.path.exists("Articles"):
            os.makedirs("Articles")
        pdf_filename = f"Articles/{title}.pdf"
        create_pdf(title, subtitle, body, pdf_filename)



async def main():
    logger.info("Starting up")
    # Setup CSV file
    
    input_file = 'output_prompts.csv'
    output_file = 'output_results.csv'
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    #generating prompts
    generate_prompts(SEO_TOKENS, input_file)
  #sampling generated prompts
    sample_prompts(input_file, "sample_prompts.csv", SAMPLES_PER_CAT)

    # Modify this block
    if USE_SAMPLE_PROMPTS:
        input_file = 'sample_prompts.csv'
    else:
        input_file = input_file
    
    async with aiohttp.ClientSession() as session:
        async with aiofiles.open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            await writer.writerow(['category', 'prompt', 'title', 'subtitle', 'body'])

            with open(input_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # skip the header

                tasks = [process_prompt(session, writer, row[0], row[1]) for row in reader if GENERATE_PDFS]
                await asyncio.gather(*tasks)

    logger.info(f"Finished processing {len(tasks)} prompts in {input_file}. Results saved to {output_file}")
    

if __name__ == '__main__':
    asyncio.run(main())