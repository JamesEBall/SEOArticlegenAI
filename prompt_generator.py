import csv
import random
from tqdm import tqdm

def generate_prompts(input_file, output_file):
    prompts = {}
    objects = {}
    
    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            category = row['Category'].strip().lower()
            obj = row['object'].strip().lower()
            obj_type = row['type'].strip().lower()

            if category not in prompts:
                prompts[category] = []

            if obj_type not in objects:
                objects[obj_type] = []

            if obj_type == "question":
                prompts[category].append(obj)
            else:
                objects[obj_type].append(obj)

    output = []

    total_prompts = sum(len(prompt_list) for prompt_list in prompts.values())

    with tqdm(total=total_prompts, desc="Generating prompts", ncols=80, unit="prompt") as progress_bar:
        for category, prompt_list in prompts.items():
            for prompt in prompt_list:
                # Check if there are multiple placeholders in the prompt
                if sum("{" in s for s in prompt.split()) > 1:
                    temp_prompts = [prompt]
                    for obj_type, obj_list in objects.items():
                        new_prompts = []
                        for temp_prompt in temp_prompts:
                            if "{" + obj_type + "}" in temp_prompt:
                                for obj in obj_list:
                                    new_prompt = temp_prompt.replace("{" + obj_type + "}", obj, 1)
                                    new_prompts.append(new_prompt)
                            else:
                                new_prompts.append(temp_prompt)
                        temp_prompts = new_prompts

                    for temp_prompt in temp_prompts:
                        output.append({"category": category, "prompt": temp_prompt})
                else:
                    for obj_type, obj_list in objects.items():
                        if "{" + obj_type + "}" in prompt:
                            for obj in obj_list:
                                generated_prompt = prompt.replace("{" + obj_type + "}", obj)
                                output.append({"category": category, "prompt": generated_prompt})

                progress_bar.update(1)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['category', 'prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in output:
            writer.writerow(row)

def sample_prompts(input_file, output_file, sample_count):
    prompts_by_category = {}

    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            category = row['category']
            prompt = row['prompt']

            if category not in prompts_by_category:
                prompts_by_category[category] = []

            prompts_by_category[category].append(prompt)

    sampled_prompts = []

    for category, prompt_list in prompts_by_category.items():
        if len(prompt_list) > sample_count:
            samples = random.sample(prompt_list, sample_count)
        else:
            samples = prompt_list

        for prompt in samples:
            sampled_prompts.append({"category": category, "prompt": prompt})

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['category', 'prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for prompt in sampled_prompts:
            writer.writerow(prompt)

