# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Slack bot that provides conversational AI capabilities through multiple LLM providers (OpenAI, Anthropic, Google, Meta, Deepseek). It runs as an AWS Lambda function and supports content extraction from URLs, PDFs, and YouTube videos.

## Development Commands

### Build and Deploy
```bash
# Build Docker image for Lambda deployment
docker build -t slack-bot .

# Update requirements-dev.txt from current environment
./update_requirements.sh
```

### Code Quality
The project uses Python development tools but doesn't have pre-configured commands:
- Format code with `black`
- Sort imports with `isort`
- Type check with `mypy` (boto3-stubs installed)

## Architecture

### Core Components

1. **Lambda Handler** (`lambda_function.py`): Entry point that processes Slack events and manages conversation threads
2. **Session Management** (`session.py`): Central logic for chat sessions, LLM interactions, and command processing
3. **Content Extraction**:
   - `web_reader.py`: Web scraping using trafilatura
   - `pdf_utils.py`: PDF text extraction
   - `ytsubs.py`: YouTube transcript extraction
   - `s3_cache.py`: Caching layer for extracted content

### Key Design Patterns

- **Unified LLM Interface**: Uses LiteLLM to abstract multiple providers (OpenAI, Anthropic, Google, etc.)
- **Agent Framework**: Integrates Agno for tool-enabled conversations (web search, calculator, etc.)
- **Streaming Responses**: Implements token-by-token streaming with Slack message updates
- **State Persistence**: User settings stored in DynamoDB, conversation history managed per thread

### Environment Configuration

Required environment variables:
- AWS credentials (for Lambda, S3, DynamoDB access)
- Slack app tokens and signing secret
- LLM API keys (stored in AWS Systems Manager Parameter Store)

### Testing Approach

No test suite exists. When adding features:
1. Test locally with direct function calls
2. Deploy to Lambda test environment
3. Verify through Slack interactions

## Important Conventions

- All LLM model references use the `LiteLLMs` enum from `lite_llms.py`
- Message handling uses data classes from `messages.py`
- S3 caching follows size limits defined in `s3_cache.py`
- Slack commands start with backslash (e.g., `\help`, `\model`)