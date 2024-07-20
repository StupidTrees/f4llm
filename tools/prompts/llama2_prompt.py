

LLAMA_ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "## Instruction:\n{instruction}\n## Input:\n{input}\n## Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "## Instruction:\n{instruction}\n## Response:\n"
    ),
}

LLAMA_CHAT_PROMPT = """<s>[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, 
toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not 
correct. If you don’t know the answer to a question, please don’t share false information.<</SYS>> {Instruction} [
/INST]\nResponse:<\s> """


LLAMA_PROMPTS = {
    "llama_alpaca": LLAMA_ALPACA_PROMPT_DICT,
    "llama_chat_default": LLAMA_CHAT_PROMPT
}
