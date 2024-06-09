def parse_output(output: str) -> str:
    return output.split("ASSISTANT:")[-1].strip()