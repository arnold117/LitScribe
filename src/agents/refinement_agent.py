"""Refinement Agent for LitScribe (Phase 9.3).

Parses user refinement instructions and modifies an existing review.
Supports: add_content, remove_content, modify_content, rewrite_section.
add_papers is deferred to a future phase.

Design:
- Two LLM calls: classify instruction + execute modification
- Preserves citation format [Author, Year]
- Returns updated review_text
- Never blocks workflow (fallback returns original text on failure)
"""

import json
import logging
from typing import Any, Dict, List, Optional

from agents.errors import LLMError
from agents.prompts import (
    REFINEMENT_CLASSIFY_PROMPT,
    REFINEMENT_EXECUTE_PROMPT,
    format_papers_for_prompt,
)
from agents.state import LitScribeState, RefinementInstruction
from agents.tools import call_llm

logger = logging.getLogger(__name__)


async def classify_instruction(
    instruction_text: str,
    review_text: str,
    model: Optional[str] = None,
    tracker=None,
) -> RefinementInstruction:
    """Classify a user instruction into a structured action.

    Args:
        instruction_text: Raw user instruction
        review_text: Current review text (for context)
        model: LLM model to use

    Returns:
        RefinementInstruction with classified action
    """
    # Use first 2000 chars as excerpt (handles CJK where split() undercounts)
    excerpt = review_text[:2000]
    if len(review_text) > 2000:
        excerpt += " ..."

    prompt = REFINEMENT_CLASSIFY_PROMPT.format(
        instruction=instruction_text,
        review_excerpt=excerpt,
    )

    response = await call_llm(prompt, model=model, temperature=0.2, max_tokens=500, tracker=tracker, agent_name="refinement", task_type="refinement")

    # Parse JSON (use robust extraction for reasoning models)
    from agents.tools import extract_json
    data = extract_json(response)

    return RefinementInstruction(
        instruction_text=instruction_text,
        action_type=data.get("action_type", "modify_content"),
        target_section=data.get("target_section"),
        details=data.get("details", instruction_text),
    )


async def execute_refinement(
    instruction: RefinementInstruction,
    current_review: str,
    analyzed_papers: List[Dict[str, Any]],
    research_question: str,
    language: str = "en",
    model: Optional[str] = None,
    tracker=None,
) -> str:
    """Execute a refinement instruction on the current review.

    Args:
        instruction: Classified instruction
        current_review: Current review text
        analyzed_papers: Available paper summaries (for citation context)
        research_question: Original research question
        language: Output language
        model: LLM model

    Returns:
        Updated review text
    """
    papers_context = format_papers_for_prompt(analyzed_papers, max_chars=4000)

    target_section = instruction.get("target_section") or "entire review"

    prompt = REFINEMENT_EXECUTE_PROMPT.format(
        research_question=research_question,
        current_review=current_review,
        papers_context=papers_context,
        action_type=instruction["action_type"],
        target_section=target_section,
        details=instruction["details"],
    )

    response = await call_llm(prompt, model=model, temperature=0.4, max_tokens=8000, tracker=tracker, agent_name="refinement", task_type="refinement")

    # Clean up: remove any markdown code fences if LLM wraps the output
    response = response.strip()
    if response.startswith("```markdown"):
        response = response[len("```markdown"):].strip()
    if response.startswith("```"):
        response = response[3:].strip()
    if response.endswith("```"):
        response = response[:-3].strip()

    return response


async def refinement_agent(state: LitScribeState) -> Dict[str, Any]:
    """Main entry point for the Refinement Agent.

    Can be called programmatically (not through the main graph).

    Args:
        state: Current workflow state with refinement_instruction populated

    Returns:
        State updates with modified synthesis
    """
    instruction = state.get("refinement_instruction")
    synthesis = state.get("synthesis")
    analyzed_papers = state.get("analyzed_papers", [])
    research_question = state["research_question"]
    language = state.get("language", "en")
    errors = list(state.get("errors", []))
    llm_config = state.get("llm_config", {})
    model = llm_config.get("model")
    from utils.token_tracker import get_tracker
    tracker = get_tracker()

    if instruction is None:
        error_msg = "No refinement instruction provided"
        logger.warning(error_msg)
        return {"errors": errors + [error_msg]}

    if synthesis is None:
        error_msg = "No synthesis to refine"
        logger.warning(error_msg)
        return {"errors": errors + [error_msg]}

    current_review = synthesis.get("review_text", "")

    logger.info(f"Refinement Agent starting: action={instruction.get('action_type', '?')}")

    try:
        # Check for unsupported action
        if instruction.get("action_type") == "add_papers":
            error_msg = (
                "add_papers is not yet supported. "
                "Please run a new review with additional queries."
            )
            return {"errors": errors + [error_msg]}

        # Execute refinement
        new_review = await execute_refinement(
            instruction=instruction,
            current_review=current_review,
            analyzed_papers=analyzed_papers,
            research_question=research_question,
            language=language,
            model=model,
            tracker=tracker,
        )

        # Update synthesis
        updated_synthesis = dict(synthesis)
        updated_synthesis["review_text"] = new_review
        from agents.synthesis_agent import count_words
        updated_synthesis["word_count"] = count_words(new_review)

        logger.info(
            f"Refinement complete: {count_words(current_review)} → "
            f"{updated_synthesis['word_count']} words"
        )

        return {
            "synthesis": updated_synthesis,
            "refinement_instruction": instruction,
        }

    except (json.JSONDecodeError, LLMError) as e:
        error_msg = f"Refinement failed: {e}"
        logger.warning(error_msg)
        errors.append(error_msg)
        return {"errors": errors}
    except Exception as e:
        error_msg = f"Refinement unexpected error: {e}"
        logger.warning(error_msg)
        errors.append(error_msg)
        return {"errors": errors}


__all__ = [
    "refinement_agent",
    "classify_instruction",
    "execute_refinement",
]
