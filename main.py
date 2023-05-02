import os
import csv
import json
import re
import asyncio
import aiohttp
import aiofiles
import logging
from prompt_generator import generate_prompts, sample_prompts
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import openai

load_dotenv()
openai.organization_id = os.environ['OPENAI_ORGANIZATION_ID']
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Params and configuration
USE_SAMPLE_PROMPTS = True  # Set to False to use input file
SAMPLES_PER_CAT = 5  # Number of samples per category


PRIMER = ("You are SEOGPT, your job is to create 300-500 word articles for a knowledge base based on the supplied prompt. "
          "Make sure that your knowledge is general. If you do not know how to fill in the prompt because you do not have "
          "enough information, return an empty response. The articles Must be SEO optimised, so keyword stuff them. If the prompted question is an impossible task, then state in the article that it is not possible to do the task and suggest another task for the user.The body *must be outputted in Markdown*. Make sure to include as many relevant links in the body as possible to relevant webpages and external content (you must include at least 1 external hu). Do not include anything that can be misconstrued as investment advice. Output the prompt in the following format: "
          "Category: , Prompt: , Title: , Subtitle: , and Body: . "
          "The prompts are all crypto and finance related.\n\n")

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


async def main():
    logger.info("Starting up")
    # Setup CSV file
    
    input_file = 'output_prompts.csv'
    output_file = 'output_results.csv'
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    #generating prompts
    generate_prompts("SEO_Template.csv", input_file)
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

                tasks = [process_prompt(session, writer, row[0], row[1]) for row in reader]
                await asyncio.gather(*tasks)

    logger.info(f"Finished processing {len(tasks)} prompts in {input_file}. Results saved to {output_file}")

if __name__ == '__main__':
    asyncio.run(main())