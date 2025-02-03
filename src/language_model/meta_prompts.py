gradient_instruction_writer_system_prompt = """
You are a user prompt writer tasked with improving a language model's user prompt for a specific task. Your goal is to identify the shortcomings of the current prompt and provide comprehensive suggestions for improvement.
""".strip()

gradient_for_instruction_writer_template = """
Here are the inputs you will be working with:

### System prompt:
{system_prompt}

### User prompt:
{instruction}

### This prompt gets the following responses wrong:
{examples}

### Remember to focus solely on discussing and improving the user prompt.

### Wrap the analysis of the user prompt in the <Analysis></Analysis> Tags.
""".strip()

optimizer_instruction_writer_system_prompt = """
You are a user prompt writer tasked with improving a language model's user prompt for a specific task. Your goal is to create an improved user prompt that enhances the model's performance.
""".strip()

optimizer_instruction_writer_template = """
Here are the inputs you will be working with:

### System prompt:
{system_prompt}

### User prompt:
{instruction}

### Wrong examples of the model's responses:
{examples}

### Analysis of the issues with this user prompt:
{analysis}

### Address any problems observed in the examples based on analysis.

### Ensure the instruction contains the <Question>{question}</Question> where the actual question will be placed.

### The new user prompt should be wrapped with <improved_instruction_prompt></improved_instruction_prompt> Tags.
""".strip()

gradient_system_writer_system_prompt = """
You are a system prompt writer tasked with improving a language model's system prompt for general tasks. Your goal is to analyze why the current system prompt fails to respond correctly in the given examples.
""".strip()

gradient_for_system_writer_template = """
Follow these instructions carefully:

### Review the current system prompt:
{system_prompt}

### Wrong responses:
{examples}

### Remember to focus solely on discussing and improving the system prompt.

### Wrap the analysis of the system prompt in the <Analysis></Analysis> Tags.
""".strip()

optimizer_system_writer_system_prompt = """
You are a system prompt writer tasked with improving a language model's system prompt. Your goal is to write a better system prompt that can be generalized for various tasks. 
""".strip()

optimizer_system_writer_template = """
Follow these instructions carefully:

### Review the current system prompt:
{system_prompt}

### Analysis of the current system prompt:
{analysis}

### Based on the information provided, write an improved system prompt.

### The new system prompt should be wrapped with <improved_system_prompt></improved_system_prompt> Tags.
""".strip()


ape_resampling_template = """
Generate a variation of the following instruction while keeping the semantic meaning.
<instruction>
{instruction}
</instruction>

Ensure the instruction contains the <Question>{question}</Question> where the actual question will be placed.

The new user prompt should be wrapped with <improved_instruction_prompt> and </improved_instruction_prompt>.
""".strip()

ape_generation_template = """
I gave a friend an instruction and inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs:

{demo}

Based on the above input-output pairs, write an instruction.
The new instruction should be wrapped with <instruction> and </instruction>.
""".strip()

