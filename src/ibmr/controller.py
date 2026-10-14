"""
Agent Controller — Orchestrator for iterative bidirectional querying.

Implements the prefrontal cortex analogue in the CLS bicameral architecture:
    - WorkingMemory:            Short-term buffer for active diagnostic session
    - ConsolidationScheduler:   Manages offline consolidation lifecycle
    - CLSAgentController:       LLM-based orchestration layer managing the
                                iterative bidirectional querying protocol

The iterative bidirectional protocol:
    1. Parse user query → extract structured fields (field, crop, symptom)
    2. Initial parallel query (episodic KG + semantic ML)
    3. Reconciliation: compare episodic facts vs. semantic patterns
    4. Refinement: semantic → episodic (semantic priors guide KG search)
                   OR episodic → semantic (anomalous facts revise semantic beliefs)
    5. Loop until convergence or max iterations
    6. Generate response with provenance tracking
"""

import json
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Any, Optional

import torch

from base import BaseAgentController
from config import AgentControllerConfig, SemanticMLConfig
from data_model import (
    KGSubgraph,
    TemporalPathQuery,
    SpatialProximityQuery,
    SemanticInferenceResult,
    ReconciliationResult,
    DiagnosticResponse,
    DiagnosticContext,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Working Memory
# ═══════════════════════════════════════════════════════════════════════════════

class WorkingMemory:
    """Short-term working memory buffer for active diagnostic session.

    Holds current query context, iteration-level results (episodic + semantic
    per cycle), reconciliation log, and intermediate reasoning state.

    Eviction policy prevents context overflow. Session-scoped: cleared at
    end of each diagnostic session.

    This is not a PyTorch module — it's a lightweight data buffer used by
    the agent controller for maintaining conversational state.
    """

    def __init__(self, max_tokens: int = 16_000, eviction_policy: str = "lru"):
        self.max_tokens = max_tokens
        self.eviction_policy = eviction_policy
        self.items: dict[str, Any] = {}
        self.access_log: list[tuple[str, datetime]] = []

    def add(self, key: str, value: Any) -> None:
        """Add item, evicting if over capacity.

        Args:
            key: Identifier for the memory item.
            value: Any serializable value.
        """
        tokens = self._estimate_tokens(str(value))
        while self._total_tokens() + tokens > self.max_tokens:
            self._evict_one()
        self.items[key] = value
        self.access_log.append((key, datetime.now()))

    def get(self, key: str) -> Optional[Any]:
        """Retrieve item, updating access log for LRU eviction.

        Args:
            key: Identifier for the memory item.

        Returns:
            The stored value, or None if not found.
        """
        self.access_log.append((key, datetime.now()))
        return self.items.get(key)

    def remove(self, key: str) -> None:
        """Remove a specific item from working memory."""
        self.items.pop(key, None)

    def clear(self) -> None:
        """Clear working memory for a new session."""
        self.items.clear()
        self.access_log.clear()

    def get_all(self) -> dict[str, Any]:
        """Return all items as a dict (for LLM context building)."""
        return dict(self.items)

    def __contains__(self, key: str) -> bool:
        return key in self.items

    def __len__(self) -> int:
        return len(self.items)

    def _total_tokens(self) -> int:
        """Estimate total token count of all stored items."""
        return sum(self._estimate_tokens(str(v)) for v in self.items.values())

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 characters per token."""
        return len(text) // 4 + 1

    def _evict_one(self) -> None:
        """Evict one item based on eviction policy."""
        if not self.items:
            return

        if self.eviction_policy == "lru":
            # Find least recently accessed item
            if self.access_log:
                # Get the item accessed least recently (not counting current add)
                access_counts = OrderedDict()
                for key, _ in reversed(self.access_log):
                    if key in self.items:
                        access_counts[key] = access_counts.get(key, 0) + 1

                if access_counts:
                    # Evict the item accessed the fewest times
                    lru_key = min(access_counts, key=lambda k: (access_counts[k], k))
                    del self.items[lru_key]
                    return

        elif self.eviction_policy == "token_count":
            # Evict the largest item by token count
            largest_key = max(
                self.items,
                key=lambda k: self._estimate_tokens(str(self.items[k])),
            )
            del self.items[largest_key]
            return

        # Fallback: evict first item
        if self.items:
            self.items.pop(next(iter(self.items)))


# ═══════════════════════════════════════════════════════════════════════════════
# Consolidation Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

class ConsolidationScheduler:
    """Manages the offline consolidation lifecycle for the semantic ML layer.

    Consolidation frequency is configurable (default: daily). A warmup period
    prevents consolidation before sufficient episodic data has accumulated.

    Consolidation triggers:
        - Scheduled: every consolidation_frequency_minutes
        - Threshold-based: when pending consolidation buffer is large
        - On-demand: explicitly triggered by agent controller after a
          high-value diagnostic session
    """

    def __init__(
        self,
        config: SemanticMLConfig,
        episodic_kg,
        semantic_memory,
    ):
        self.config = config
        self.episodic_kg = episodic_kg
        self.semantic_memory = semantic_memory
        self.system_start_time = datetime.now()
        self.consolidation_count = 0

    def should_consolidate(self) -> bool:
        """Check whether consolidation should run."""
        # Warmup period check
        hours_elapsed = (datetime.now() - self.system_start_time).total_seconds() / 3600
        if hours_elapsed < self.config.consolidation_warmup_hours:
            return False

        # Time-based check
        if self.semantic_memory.last_consolidation_time is not None:
            minutes_since = (
                datetime.now() - self.semantic_memory.last_consolidation_time
            ).total_seconds() / 60
            if minutes_since < self.config.consolidation_frequency_minutes:
                return False

        return True

    def step(self) -> dict:
        """Called periodically. Returns consolidation stats or empty dict if skipped.

        Returns:
            dict with consolidation statistics, or {"skipped": True}.
        """
        if self.should_consolidate():
            batch = self.episodic_kg.extract_consolidation_batch(
                max_samples=self.config.consolidation_batch_size * 10
            )
            if len(batch) >= self.config.consolidation_batch_size:
                stats = self.semantic_memory.consolidate(batch)
                self.consolidation_count += 1

                # Update KG subgraph embeddings for semantic-prior retrieval
                embeds = []
                for sg in batch:
                    result = self.semantic_memory.infer_pattern(sg)
                    embeds.append(result.pattern_embed)
                self.episodic_kg.update_subgraph_embeddings(embeds)

                # Optionally clear consolidated subgraphs
                self.episodic_kg.clear_pending_consolidation()

                stats["consolidation_count"] = self.consolidation_count
                return stats

        return {"skipped": True}

    def force_consolidation(self) -> dict:
        """Force an immediate consolidation cycle regardless of schedule."""
        batch = self.episodic_kg.extract_consolidation_batch(max_samples=10_000)
        if batch:
            stats = self.semantic_memory.consolidate(batch)
            self.consolidation_count += 1
            return stats
        return {"skipped": True, "reason": "no_matured_subgraphs"}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Interface (Stub)
# ═══════════════════════════════════════════════════════════════════════════════

class LLMInterface:
    """Lightweight interface for LLM calls in the agent controller.

    Provides a uniform API for text completion that can be backed by
    any LLM provider (OpenAI, Anthropic, local models, or a simple
    rule-based fallback for testing).

    The stub implementation provides deterministic responses for
    testing and smoke tests. In production, replace with the actual
    LLM provider SDK.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Send a completion request to the LLM.

        In the stub implementation, returns a deterministic JSON response
        for reconciliation prompts and a template response for other prompts.

        Args:
            prompt: The prompt to send.
            temperature: Optional override for temperature.

        Returns:
            str: The completion text.
        """
        # Stub: return deterministic responses for known prompt patterns
        if "Output JSON" in prompt:
            return json.dumps({
                "consistency_score": 0.85,
                "gaps": ["humidity data not found for the critical period"],
                "contradictions": [],
                "semantic_prior_needed": True,
                "episodic_refinement_needed": True,
                "refined_query_suggestion": "query for humidity readings near field 7 days before onset",
            })

        # Default stub response
        return (
            "Based on the episodic observations and semantic patterns, here is my analysis:\n\n"
            "1. The observed symptoms match the semantic pattern for early-stage powdery mildew.\n"
            "2. High humidity conditions preceding the outbreak are consistent with known risk factors.\n"
            "3. Recommended action: apply targeted fungicide treatment within 72 hours.\n\n"
            "Confidence: HIGH (episodic evidence supports semantic pattern)."
        )

    def __call__(self, prompt: str, **kwargs) -> str:
        """Callable interface for convenience."""
        return self.complete(prompt, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# CLS Agent Controller
# ═══════════════════════════════════════════════════════════════════════════════

class CLSAgentController(BaseAgentController):
    """
    LLM-based orchestration layer for CLS bicameral memory.

    Implements the prefrontal cortex analogue managing the iterative
    bidirectional querying protocol between episodic KG and semantic ML.

    The iterative protocol:
        1. Parse natural language query → structured diagnostic context
        2. Initial parallel query (episodic KG + semantic ML)
        3. Reconciliation: assess consistency, gaps, contradictions
        4. Semantic → Episodic refinement (if semantic prior needed)
        5. Episodic → Semantic refinement (if episodic revision needed)
        6. Loop until convergence or max_iterative_cycles
        7. Generate response with provenance tracking

    This is NOT a PyTorch nn.Module — it orchestrates nn.Module-based
    subsystems via their high-level APIs.
    """

    def __init__(
        self,
        config: AgentControllerConfig,
        episodic_kg,
        semantic_memory,
    ):
        super().__init__()
        self.config = config
        self.episodic_kg = episodic_kg
        self.semantic_memory = semantic_memory
        self.llm = LLMInterface(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        self.working_memory = WorkingMemory(
            max_tokens=config.working_memory_max_tokens,
            eviction_policy=config.working_memory_eviction,
        )

        # Statistics
        self._diagnosis_count = 0
        self._total_iterations = 0

    # ─── Main Entry Point ─────────────────────────────────────────────────────

    def diagnose(
        self,
        query: str,
        context: Optional[DiagnosticContext] = None,
    ) -> DiagnosticResponse:
        """
        Main entry point for agricultural diagnostics.

        Implements the iterative bidirectional querying protocol:
        1. Parse query → extract structured fields
        2. Initial parallel query (episodic KG + semantic ML)
        3-6. Iterative reconciliation loop (max_iterative_cycles)
        7. Generate response with provenance tracking

        Args:
            query: Natural language diagnostic query (e.g., "Why is my wheat
                   showing powdery mildew in field A-42?")
            context: DiagnosticContext with field, crop, season info.
                     If None, a minimal context is inferred from the query.

        Returns:
            DiagnosticResponse with answer, provenance, confidence, and evidence.
        """
        self._diagnosis_count += 1

        # Initialize working memory for this session
        self.working_memory.clear()
        self.working_memory.add("query", query)

        if context is None:
            context = self._infer_context(query)
        self.working_memory.add("context", context)

        # ── Step 1: Parse query ──
        parsed = self._parse_diagnostic_query(query, context)
        self.working_memory.add("parsed_query", parsed)

        # ── Step 2: Initial parallel query ──
        # Write the current query/observation to episodic KG
        observation_id = self.episodic_kg.fast_write({
            "type": "observation",
            "timestamp": datetime.now(),
            "field_id": context.field_id,
            "query": query,
            "summary": f"Query: {query}",
        })

        # Query episodic KG for relevant history
        kg_results = self._initial_kg_query(context, parsed)

        # Infer semantic pattern
        semantic_results = self.semantic_memory.infer_pattern(query)

        # Store in working memory
        self.working_memory.add("observation_id", observation_id)
        self.working_memory.add("kg_results_iter_0", kg_results)
        self.working_memory.add("semantic_results_iter_0", semantic_results)

        # ── Steps 3-6: Iterative reconciliation loop ──
        reconciliation_log: list[ReconciliationResult] = []
        iteration = 0

        while iteration < self.config.max_iterative_cycles:
            iteration += 1

            # Store iteration results in working memory
            self.working_memory.add(f"kg_results_iter_{iteration}", kg_results)
            self.working_memory.add(f"semantic_results_iter_{iteration}", semantic_results)

            # Reconciliation: compare episodic and semantic findings
            reconciliation = self._reconcile(
                episodic_facts=kg_results,
                semantic_pattern=semantic_results,
                iteration=iteration,
            )
            reconciliation_log.append(reconciliation)
            self.working_memory.add(f"reconciliation_iter_{iteration}", reconciliation)

            # Check early exit
            if reconciliation.confidence >= self.config.early_exit_confidence:
                logger.debug(f"Early exit at iteration {iteration} "
                             f"(confidence={reconciliation.confidence:.2f})")
                break

            # Decide refinement direction
            refinement_made = False

            # Semantic → Episodic refinement
            if (reconciliation.semantic_prior_needed
                    and self.config.enable_semantic_prior_routing
                    and iteration < self.config.max_iterative_cycles):
                refined_kg = self.semantic_memory.query_episodic_via_semantic_prior(
                    semantic_results,
                    self.episodic_kg,
                    k=self.semantic_memory.config.few_shot_k,
                )
                if refined_kg:
                    kg_results = refined_kg
                    refinement_made = True
                    logger.debug(f"Iteration {iteration}: Semantic → Episodic refinement "
                                 f"(found {len(refined_kg)} additional subgraphs)")

            # Episodic → Semantic refinement
            if (reconciliation.episodic_refinement_needed
                    and self.config.enable_episodic_revision
                    and iteration < self.config.max_iterative_cycles):
                refined_semantic = self.semantic_memory.few_shot_adapt(
                    support_subgraphs=kg_results[:self.semantic_memory.config.few_shot_k],
                    query=self._build_query_subgraph(query, context),
                )
                if refined_semantic["confidence"] > semantic_results.confidence:
                    semantic_results = SemanticInferenceResult(
                        pattern_embed=refined_semantic["pattern_embed"],
                        matched_prototype_idx=None,  # few-shot, not prototype
                        confidence=refined_semantic["confidence"],
                        provenance="few_shot_adapted",
                        summary=(f"Few-shot adapted from {len(kg_results)} support subgraphs; "
                                 f"similarity={refined_semantic.get('similarity', 0):.3f}"),
                    )
                    refinement_made = True
                    logger.debug(f"Iteration {iteration}: Episodic → Semantic refinement "
                                 f"(confidence={refined_semantic['confidence']:.3f})")

            if not refinement_made:
                # No further refinement possible
                logger.debug(f"Iteration {iteration}: No refinement possible; stopping.")
                break

        # ── Step 7: Generate response with provenance ──
        response_data = self._generate_response(
            query=query,
            reconciliation_log=reconciliation_log,
            kg_results=kg_results,
            semantic_results=semantic_results,
        )

        self._total_iterations += iteration

        return DiagnosticResponse(
            answer=response_data.get("answer", "Diagnostic analysis complete."),
            provenance=response_data.get("provenance", []),
            num_iterations=iteration,
            confidence=(
                reconciliation_log[-1].confidence
                if reconciliation_log
                else semantic_results.confidence
            ),
            evidence=response_data.get("evidence", []),
        )

    # ─── Reconciliation ───────────────────────────────────────────────────────

    def _reconcile(
        self,
        episodic_facts: list[KGSubgraph],
        semantic_pattern: SemanticInferenceResult,
        iteration: int,
    ) -> ReconciliationResult:
        """
        Compare episodic facts against semantic pattern to identify:
            - Consistency: do the facts match the pattern?
            - Gaps: does the pattern suggest factors not in the KG?
            - Contradictions: do facts contradict the pattern?

        Args:
            episodic_facts: Results from episodic KG queries.
            semantic_pattern: Results from semantic ML inference.
            iteration: Current iteration number.

        Returns:
            ReconciliationResult with scores, gaps, contradictions,
            refinement flags, and confidence.
        """
        if self.config.reconciliation_method == "llm_judge":
            return self._llm_reconciliation(episodic_facts, semantic_pattern, iteration)
        elif self.config.reconciliation_method == "weighted_vote":
            return self._weighted_vote_reconciliation(episodic_facts, semantic_pattern, iteration)
        else:
            return self._confidence_max_reconciliation(episodic_facts, semantic_pattern, iteration)

    def _llm_reconciliation(
        self,
        episodic_facts: list[KGSubgraph],
        semantic_pattern: SemanticInferenceResult,
        iteration: int,
    ) -> ReconciliationResult:
        """Use LLM to reconcile episodic facts with semantic pattern."""
        prompt = (
            f"Compare the following episodic facts against the semantic pattern.\n"
            f"Assess consistency, identify gaps, and flag contradictions.\n\n"
            f"Episodic facts:\n{self._format_subgraphs(episodic_facts)}\n\n"
            f"Semantic pattern (confidence: {semantic_pattern.confidence:.2f}):\n"
            f"{self._format_pattern(semantic_pattern)}\n\n"
            f"Output JSON:\n"
            f"{{\n"
            f'    "consistency_score": <0.0-1.0>,\n'
            f'    "gaps": [<list of missing information>],\n'
            f'    "contradictions": [<list of contradictions>],\n'
            f'    "semantic_prior_needed": <true/false>,\n'
            f'    "episodic_refinement_needed": <true/false>,\n'
            f'    "refined_query_suggestion": "<optional revised query>"\n'
            f"}}"
        )

        try:
            result_str = self.llm.complete(prompt, temperature=0.0)
            result = json.loads(result_str)
            return ReconciliationResult(
                consistency_score=result.get("consistency_score", 0.5),
                gaps=result.get("gaps", []),
                contradictions=result.get("contradictions", []),
                semantic_prior_needed=result.get("semantic_prior_needed", False),
                episodic_refinement_needed=result.get("episodic_refinement_needed", False),
                refined_query_suggestion=result.get("refined_query_suggestion"),
                confidence=result.get("consistency_score", 0.5),
                iteration=iteration,
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback to confidence-based if LLM response is malformed
            logger.warning("LLM reconciliation returned malformed JSON; falling back to confidence-max.")
            return self._confidence_max_reconciliation(episodic_facts, semantic_pattern, iteration)

    def _weighted_vote_reconciliation(
        self,
        episodic_facts: list[KGSubgraph],
        semantic_pattern: SemanticInferenceResult,
        iteration: int,
    ) -> ReconciliationResult:
        """Weighted vote: combine episodic confidence with semantic confidence."""
        # Episodic confidence = mean of subgraph confidences
        ep_conf = (
            sum(sg.confidence for sg in episodic_facts) / max(len(episodic_facts), 1)
        )
        # Weighted combination (equal weight)
        combined = 0.5 * ep_conf + 0.5 * semantic_pattern.confidence

        return ReconciliationResult(
            consistency_score=combined,
            gaps=[],
            contradictions=[],
            semantic_prior_needed=(
                semantic_pattern.confidence < self.config.early_exit_confidence
                and len(episodic_facts) > 0
            ),
            episodic_refinement_needed=(
                semantic_pattern.confidence < self.config.early_exit_confidence
                and len(episodic_facts) > 0
            ),
            confidence=combined,
            iteration=iteration,
        )

    def _confidence_max_reconciliation(
        self,
        episodic_facts: list[KGSubgraph],
        semantic_pattern: SemanticInferenceResult,
        iteration: int,
    ) -> ReconciliationResult:
        """Confidence-max: use the higher of episodic and semantic confidence."""
        ep_conf = (
            max(sg.confidence for sg in episodic_facts)
            if episodic_facts else 0.0
        )
        combined = max(ep_conf, semantic_pattern.confidence)

        return ReconciliationResult(
            consistency_score=combined,
            gaps=[],
            contradictions=[],
            semantic_prior_needed=(
                semantic_pattern.confidence < self.config.early_exit_confidence
                and len(episodic_facts) > 0
            ),
            episodic_refinement_needed=(
                semantic_pattern.confidence < self.config.early_exit_confidence
                and len(episodic_facts) > 0
            ),
            confidence=combined,
            iteration=iteration,
        )

    # ─── Response Generation ─────────────────────────────────────────────────

    def _generate_response(
        self,
        query: str,
        reconciliation_log: list[ReconciliationResult],
        kg_results: list[KGSubgraph],
        semantic_results: SemanticInferenceResult,
    ) -> dict:
        """
        Generate final diagnostic response with provenance tracking.

        Every claim in the output is tagged with its source layer
        (episodic KG, semantic ML, or both) and confidence.

        Args:
            query: Original user query.
            reconciliation_log: History of reconciliation steps.
            kg_results: Final episodic KG results.
            semantic_results: Final semantic ML results.

        Returns:
            dict with keys:
                - answer (str): Natural language diagnosis
                - provenance (list[dict]): Per-claim source tags
                - evidence (list[dict]): Supporting evidence from each layer
        """
        # Build provenance items
        provenance = []
        evidence = []

        if self.config.provenance_tracking:
            # Tag episodic facts
            for sg in kg_results:
                claim = sg.summary or f"Subgraph at {sg.timestamp.isoformat()}"
                provenance.append({
                    "claim": claim,
                    "source": "episodic_kg",
                    "confidence": sg.confidence,
                    "timestamp": sg.timestamp.isoformat(),
                })
                evidence.append({
                    "type": "episodic",
                    "content": claim,
                    "confidence": sg.confidence,
                    "query_type": sg.query_type,
                })

            # Tag semantic pattern
            provenance.append({
                "claim": (
                    f"Semantic pattern: {semantic_results.summary}"
                    if semantic_results.summary
                    else f"Pattern prototype #{semantic_results.matched_prototype_idx}"
                ),
                "source": "semantic_ml",
                "confidence": semantic_results.confidence,
                "provenance": semantic_results.provenance,
            })
            evidence.append({
                "type": "semantic",
                "content": semantic_results.summary,
                "confidence": semantic_results.confidence,
                "provenance": semantic_results.provenance,
            })

        # Generate answer using LLM
        prompt = (
            f"You are an agricultural diagnostic agent with a bicameral memory system.\n"
            f"Given the user's query, the episodic facts (specific observations), and the semantic\n"
            f"pattern (generalized agricultural knowledge), produce a diagnosis.\n\n"
            f"User query: {query}\n\n"
            f"Episodic facts (specific, timestamped):\n"
            f"{self._format_subgraphs(kg_results)}\n\n"
            f"Semantic pattern (generalized, confidence {semantic_results.confidence:.2f}):\n"
            f"{self._format_pattern(semantic_results)}\n\n"
            f"Reconciliation history ({len(reconciliation_log)} iterations):\n"
            f"{self._format_reconciliation_log(reconciliation_log)}\n\n"
            f"Provide:\n"
            f"1. A clear diagnosis\n"
            f"2. Confidence level\n"
            f"3. Supporting evidence from each memory system\n"
            f"4. Any caveats or alternative explanations"
        )

        answer = self.llm.complete(prompt, temperature=self.config.llm_temperature)

        return {
            "answer": answer,
            "provenance": provenance,
            "evidence": evidence,
        }

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _parse_diagnostic_query(self, query: str, context: DiagnosticContext) -> dict:
        """Parse a natural language query into structured fields.

        Extracts field ID, crop type, symptom, and time range references.

        Args:
            query: Natural language query (e.g., "Why is my wheat showing
                   powdery mildew in field A-42?")
            context: DiagnosticContext with known field/crop/season info.

        Returns:
            dict with keys: field_id, crop_type, symptom, time_range, raw_query.
        """
        # Simple rule-based parsing (in production, use LLM-based extraction)
        query_lower = query.lower()
        parsed = {
            "raw_query": query,
            "field_id": context.field_id,
            "crop_type": context.crop_type,
            "symptom": "",
            "time_range": {},
        }

        # Extract crop type (try to match known crops)
        known_crops = ["wheat", "corn", "maize", "rice", "soybean", "cotton",
                       "tomato", "potato", "grape", "apple", "chilli", "pepper"]
        for crop in known_crops:
            if crop in query_lower:
                parsed["crop_type"] = crop
                break

        # Extract symptom (words following "showing", "with", "exhibiting")
        symptom_keywords = ["showing", "with", "exhibiting", "displaying",
                           "suffering from", "signs of"]
        for kw in symptom_keywords:
            if kw in query_lower:
                idx = query_lower.find(kw) + len(kw)
                remainder = query_lower[idx:].strip().split(" in")[0].split(".")[0]
                if remainder:
                    parsed["symptom"] = remainder.strip()
                    break

        # Extract field ID
        import re
        field_match = re.search(r'(field\s+)?([A-Za-z]-?\d+)', query_lower)
        if field_match:
            parsed["field_id"] = field_match.group(2).upper()

        return parsed

    def _infer_context(self, query: str) -> DiagnosticContext:
        """Infer diagnostic context from query when none is provided."""
        parsed = self._parse_diagnostic_query(query, DiagnosticContext(field_id="unknown"))
        return DiagnosticContext(
            field_id=parsed["field_id"],
            crop_type=parsed["crop_type"],
            season_start=None,
            season_end=None,
        )

    def _initial_kg_query(
        self,
        context: DiagnosticContext,
        parsed: dict,
    ) -> list[KGSubgraph]:
        """Execute initial parallel KG queries for relevant history.

        Runs temporal path queries and spatial proximity queries in
        sequence (for the stub implementation; in production these
        would be parallelized).

        Args:
            context: DiagnosticContext with field, crop, season info.
            parsed: Parsed query dict.

        Returns:
            list[KGSubgraph] — Aggregated KG query results.
        """
        results = []

        # Temporal path query: find the treatment/disease sequence for the field
        if context.field_id != "unknown":
            temporal_results = self.episodic_kg.temporal_path_query(
                TemporalPathQuery(
                    start_node_id=f"field_{context.field_id}",
                    relation_sequence=("occurred_during", "treated_with", "followed_by"),
                    from_date=context.season_start,
                    to_date=context.season_end,
                    max_hops=3,
                )
            )
            results.extend(temporal_results)

            # Spatial proximity query: find nearby disease events
            spatial_results = self.episodic_kg.spatial_proximity_query(
                SpatialProximityQuery(
                    center_field_id=context.field_id,
                    radius_m=500.0,
                    from_date=context.season_start,
                    to_date=context.season_end,
                )
            )
            results.extend(spatial_results)

        return results

    def _build_query_subgraph(self, query: str, context: DiagnosticContext) -> KGSubgraph:
        """Build a minimal subgraph representing the current query for few-shot adaptation.

        Args:
            query: Natural language query.
            context: DiagnosticContext.

        Returns:
            KGSubgraph with a single tensor-featurized node. [bench-fix] The
            node carries explicit features so episodic→semantic refinement
            (few_shot_adapt) can encode it; an unfeaturized subgraph would
            crash SubgraphBatch with a 0-dim feature matrix.
        """
        import hashlib
        node_id = f"query_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        sg = KGSubgraph(
            root_node_id=node_id,
            query_type="query",
            summary=f"Current query: {query}",
            metadata={"query": query, "field_id": context.field_id},
        )
        # [bench-fix] Single node + self-loop edge so SubgraphBatch gets a
        # non-zero feature matrix; a nodeless/edgeless subgraph crashes
        # SubgraphBatch with 0-dim tensors during few_shot_adapt encoding.
        from data_model import KGNode, KGEdge
        from datetime import timedelta
        sg.nodes = [KGNode(
            node_id=node_id,
            node_type="observation",
            timestamp=datetime.now(),
            attributes={"summary": f"Current query: {query}", "field_id": context.field_id},
        )]
        sg.edges = [KGEdge(
            source=node_id, target=node_id, relation="self",
            timestamp=datetime.now() + timedelta(seconds=1),
        )]
        self.episodic_kg._populate_subgraph_tensors(sg)
        if sg.node_features is None:
            sg.node_features = torch.zeros(1, self.episodic_kg.config.node_embed_dim)
        return sg

    # ─── Formatting Helpers for LLM Prompts ───────────────────────────────────

    def _format_subgraphs(self, subgraphs: list[KGSubgraph]) -> str:
        """Format KG subgraphs as readable text for LLM prompts.

        Args:
            subgraphs: List of KGSubgraph objects.

        Returns:
            str: Formatted text representation.
        """
        if not subgraphs:
            return "  (no episodic facts found)"

        lines = []
        for i, sg in enumerate(subgraphs):
            lines.append(f"  [{i + 1}] {sg.summary}")
            lines.append(f"      Type: {sg.query_type} | "
                         f"Nodes: {sg.num_nodes} | "
                         f"Edges: {sg.num_edges} | "
                         f"Confidence: {sg.confidence:.2f}")
            lines.append(f"      Timestamp: {sg.timestamp.isoformat()}")

            # List nodes if available
            for node in sg.nodes:
                attrs = node.attributes
                disease = attrs.get("disease_name", "")
                severity = attrs.get("severity", "")
                if disease:
                    lines.append(f"      - Disease: {disease} (severity: {severity})")

        return "\n".join(lines)

    def _format_pattern(self, semantic_result: SemanticInferenceResult) -> str:
        """Format semantic pattern result as readable text for LLM prompts.

        Args:
            semantic_result: SemanticInferenceResult object.

        Returns:
            str: Formatted text representation.
        """
        return (
            f"  Pattern: {semantic_result.summary}\n"
            f"  Confidence: {semantic_result.confidence:.2f}\n"
            f"  Provenance: {semantic_result.provenance}\n"
            f"  Prototype index: {semantic_result.matched_prototype_idx}"
        )

    def _format_reconciliation_log(self, log: list[ReconciliationResult]) -> str:
        """Format reconciliation history as readable text for LLM prompts.

        Args:
            log: List of ReconciliationResult objects.

        Returns:
            str: Formatted text representation.
        """
        if not log:
            return "  (no reconciliation performed)"

        lines = []
        for r in log:
            lines.append(f"  Iteration {r.iteration}:")
            lines.append(f"    Consistency: {r.consistency_score:.2f}")
            lines.append(f"    Confidence: {r.confidence:.2f}")
            lines.append(f"    Gaps: {', '.join(r.gaps) if r.gaps else 'none'}")
            lines.append(f"    Contradictions: {', '.join(r.contradictions) if r.contradictions else 'none'}")
            lines.append(f"    Semantic prior needed: {r.semantic_prior_needed}")
            lines.append(f"    Episodic refinement needed: {r.episodic_refinement_needed}")

        return "\n".join(lines)

    def get_statistics(self) -> dict:
        """Return controller usage statistics."""
        return {
            "diagnosis_count": self._diagnosis_count,
            "total_iterations": self._total_iterations,
            "avg_iterations_per_diagnosis": (
                self._total_iterations / max(self._diagnosis_count, 1)
            ),
        }
