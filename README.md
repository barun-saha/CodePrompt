# CodePrompt

A collection of prompts to investigate code generation using [PaLM 2](https://ai.google/discover/palm2/). 

The dataset consists of 30 coding problems in Python. These range from code generation to completion and troubleshooting errors. A summary of these problems is available in the accompanying spreadsheet.

The CodePrompt dataset takes into consideration the sequential nature of trial and error that is often required with Large Language Models (LLMs) in order to obtain the desired output (or the output in a desired format). For example, some problems can be solved at the first attempt. Some might need a second attempt by refining the input prompt, a third attempt after refining further refining the input prompt, and so on. Accordingly, three maximum attempts are considered here. In case no correct output is generated even after the third attempt, a comment is provided in the spreadsheet explaining the reason.

A conference paper detailing the experimental results using CodePrompt has been accepted and will appear in 2024.
