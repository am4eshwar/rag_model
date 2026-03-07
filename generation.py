"""
Generation Module

Generates structured verification reports using LLM based on retrieved context.

DESIGN PHILOSOPHY:
- Deterministic generation: Low temperature for consistent, factual reports
- Structured output: JSON format for easy parsing and display
- Citation-based: Every claim must reference a retrieved chunk
- Confidence scoring: Provide uncertainty estimates
- Modular LLM: Support both local models and API-based (OpenAI, Anthropic, etc.)

PROMPT ENGINEERING PRINCIPLES:
1. Clear task definition: "You are a teacher's assistant verifying student work"
2. Structured input: Query + Retrieved chunks + Metadata
3. Explicit output format: JSON schema with required fields
4. Citation requirement: Force model to cite chunk IDs
5. Confidence calibration: Ask for self-assessment of certainty

GENERATION PARAMETERS:
- temperature=0.1: Very deterministic (0.0 may be too repetitive)
- max_tokens=1024: Enough for detailed report (~700 words)
- top_p=0.9: Slight diversity while staying focused
- presence_penalty=0: Allow repeated citations/terms
- frequency_penalty=0: Don't penalize technical term reuse

WHEN TO ADJUST:
- Increase temperature to 0.3 if reports are too rigid/repetitive
- Increase max_tokens to 2048 if reports are truncated
- Decrease temperature to 0.0 for maximum consistency (testing)
"""

import logging
from typing import List, Dict, Optional
import json
from dataclasses import dataclass
from datetime import datetime

from config import (
    GENERATOR_MODEL_NAME,
    GENERATOR_TYPE,
    GENERATION_TEMPERATURE,
    GENERATION_MAX_TOKENS,
    GENERATION_TOP_P,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

VERIFICATION_SYSTEM_PROMPT = """You are an AI assistant helping teachers verify and analyze student submissions.

Your task:
1. Analyze the teacher's question/request about a student submission
2. Use the provided retrieved text chunks as evidence
3. Generate a structured verification report

Requirements:
- Be factual and objective
- Cite specific chunks using [chunk_id] notation
- Provide confidence scores (0.0-1.0) for your assessments
- If information is insufficient, clearly state that
- Do not make claims without citing supporting evidence

Output format: JSON with these fields:
{
  "summary": "Brief overview (2-3 sentences)",
  "findings": [
    {
      "statement": "A specific finding",
      "evidence": ["chunk_id_1", "chunk_id_2"],
      "confidence": 0.9
    }
  ],
  "answer": "Direct answer to the teacher's question",
  "confidence": 0.85,
  "limitations": "Any limitations or missing information",
  "recommendations": "Optional recommendations for the teacher"
}"""

VERIFICATION_USER_PROMPT_TEMPLATE = """Teacher's Question:
{query}

Student Submission Metadata:
- Document ID: {doc_id}
- Filename: {filename}
- Pages: {page_count}

Retrieved Relevant Chunks:
{chunks}

Please analyze the submission and provide a structured verification report in JSON format."""


def format_chunks_for_prompt(chunks: List) -> str:
    """
    Format retrieved chunks for inclusion in prompt
    
    Args:
        chunks: List of RetrievalResult objects
        
    Returns:
        Formatted string with chunks
    """
    formatted = []
    
    for chunk in chunks:
        chunk_info = [
            f"[{chunk.chunk_id}]",
            f"Score: {chunk.score:.3f}",
        ]
        
        # Add metadata if available
        if chunk.metadata:
            if "doc_id" in chunk.metadata:
                chunk_info.append(f"Document: {chunk.metadata['doc_id']}")
            if "page_num" in chunk.metadata:
                chunk_info.append(f"Page: {chunk.metadata['page_num']}")
        
        formatted.append(" | ".join(chunk_info))
        formatted.append(f'Text: "{chunk.text}"')
        formatted.append("")  # Blank line
    
    return "\n".join(formatted)


# ==============================================================================
# REPORT DATACLASS
# ==============================================================================

@dataclass
class VerificationReport:
    """
    Structured verification report
    
    Attributes:
        query: Original teacher question
        doc_id: Document being analyzed
        summary: Brief overview
        findings: List of specific findings with evidence
        answer: Direct answer to query
        confidence: Overall confidence (0-1)
        limitations: Any caveats or missing info
        recommendations: Optional suggestions
        retrieved_chunks: Original chunks used
        generation_metadata: Model info, timestamp, etc.
    """
    query: str
    doc_id: str
    summary: str
    findings: List[Dict]
    answer: str
    confidence: float
    limitations: str
    recommendations: Optional[str] = None
    retrieved_chunks: Optional[List] = None
    generation_metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "query": self.query,
            "doc_id": self.doc_id,
            "summary": self.summary,
            "findings": self.findings,
            "answer": self.answer,
            "confidence": self.confidence,
            "limitations": self.limitations,
            "recommendations": self.recommendations,
            "retrieved_chunks": [
                chunk.to_dict() if hasattr(chunk, "to_dict") else chunk
                for chunk in (self.retrieved_chunks or [])
            ],
            "generation_metadata": self.generation_metadata,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Convert to human-readable markdown"""
        md = []
        md.append("# Verification Report")
        md.append(f"**Query:** {self.query}")
        md.append(f"**Document:** {self.doc_id}")
        md.append(f"**Confidence:** {self.confidence:.1%}")
        md.append("")
        
        md.append("## Summary")
        md.append(self.summary)
        md.append("")
        
        md.append("## Answer")
        md.append(self.answer)
        md.append("")
        
        if self.findings:
            md.append("## Findings")
            for i, finding in enumerate(self.findings, 1):
                md.append(f"### {i}. {finding['statement']}")
                md.append(f"**Confidence:** {finding['confidence']:.1%}")
                md.append(f"**Evidence:** {', '.join(finding['evidence'])}")
                md.append("")
        
        if self.limitations:
            md.append("## Limitations")
            md.append(self.limitations)
            md.append("")
        
        if self.recommendations:
            md.append("## Recommendations")
            md.append(self.recommendations)
            md.append("")
        
        return "\n".join(md)


# ==============================================================================
# GENERATOR CLASSES
# ==============================================================================

class BaseGenerator:
    """Base class for LLM generators"""
    
    def generate(
        self,
        query: str,
        chunks: List,
        doc_metadata: Optional[Dict] = None,
    ) -> VerificationReport:
        """
        Generate verification report
        
        Args:
            query: Teacher's question
            chunks: Retrieved RetrievalResult objects
            doc_metadata: Metadata about the document
            
        Returns:
            VerificationReport object
        """
        raise NotImplementedError


class GroqGenerator(BaseGenerator):
    """
    Generator using Groq API via LangChain
    
    SETUP:
    - Requires Groq API key in environment: GROQ_API_KEY
    - Get API key from: https://console.groq.com/
    - Install: pip install langchain-groq
    
    MODELS:
    - llama-3.3-70b-versatile: Llama 3.3 70B (recommended, best quality)
    - llama-3.1-70b-versatile: Llama 3.1 70B
    - gemma2-9b-it: Google Gemma 2 9B instruction-tuned
    
    ADVANTAGES:
    - Very fast inference (often faster than OpenAI)
    - Good quality models
    - Competitive pricing
    """
    
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = GENERATION_TEMPERATURE,
        max_tokens: int = GENERATION_MAX_TOKENS,
    ):
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "langchain-groq not installed. Run: pip install langchain-groq"
            )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Groq chat model via LangChain
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        logger.info(f"Initialized Groq generator with model: {model_name}")
    
    def generate(
        self,
        query: str,
        chunks: List,
        doc_metadata: Optional[Dict] = None,
    ) -> VerificationReport:
        """Generate report using Groq API via LangChain"""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        doc_metadata = doc_metadata or {}
        
        # Format chunks for prompt
        chunks_text = format_chunks_for_prompt(chunks)
        
        # Build user prompt
        user_prompt = VERIFICATION_USER_PROMPT_TEMPLATE.format(
            query=query,
            doc_id=doc_metadata.get("doc_id", "unknown"),
            filename=doc_metadata.get("filename", "unknown"),
            page_count=doc_metadata.get("page_count", "unknown"),
            chunks=chunks_text,
        )
        
        # Create messages
        messages = [
            SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        
        # Call Groq API via LangChain
        logger.info(f"Generating response with Groq ({self.model_name})...")
        response = self.llm.invoke(messages)
        content = response.content
        
        # Parse response
        try:
            # Try to extract JSON from response
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                report_data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response: {content[:500]}...")
            # Fallback
            report_data = {
                "summary": "Error parsing response",
                "findings": [],
                "answer": content,
                "confidence": 0.5,
                "limitations": f"JSON parsing failed: {e}",
            }
        
        # Create report object
        report = VerificationReport(
            query=query,
            doc_id=doc_metadata.get("doc_id", "unknown"),
            summary=report_data.get("summary", ""),
            findings=report_data.get("findings", []),
            answer=report_data.get("answer", ""),
            confidence=report_data.get("confidence", 0.5),
            limitations=report_data.get("limitations", ""),
            recommendations=report_data.get("recommendations"),
            retrieved_chunks=chunks,
            generation_metadata={
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "provider": "groq",
            },
        )
        
        return report


class LocalGenerator(BaseGenerator):
    """
    Generator using local Hugging Face model
    
    SETUP:
    - Install: pip install transformers torch accelerate
    """
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        temperature: float = GENERATION_TEMPERATURE,
        max_tokens: int = GENERATION_MAX_TOKENS,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                import torch
                
                logger.info(f"Loading local model: {self.model_name}")
                
                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
                
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device_map="auto" if device == "cuda" else None,
                    device=-1 if device == "cpu" else 0,
                    torch_dtype=torch_dtype,
                    max_new_tokens=self.max_tokens,
                )
                logger.info(f"Model loaded on {device}")
            except ImportError:
                raise ImportError(
                    "Local generation requires 'transformers', 'torch', and 'accelerate'. "
                    "Run: pip install transformers torch accelerate"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load local model: {e}")
        return self._pipeline
    
    def generate(
        self,
        query: str,
        chunks: List,
        doc_metadata: Optional[Dict] = None,
    ) -> VerificationReport:
        """
        Generate report using local model
        """
        doc_metadata = doc_metadata or {}
        chunks_text = format_chunks_for_prompt(chunks)
        
        # Build prompt suitable for chat models
        # Simplify system prompt for smaller models to ensure they follow JSON instructions
        system_instruction = (
            "You are a helpful assistant. "
            "Analyze the retrieved text chunks and the user's query. "
            "Output your answer ONLY in valid JSON format. "
            "Do not include any explanation before or after the JSON."
        )
        
        user_prompt = VERIFICATION_USER_PROMPT_TEMPLATE.format(
            query=query,
            doc_id=doc_metadata.get("doc_id", "unknown"),
            filename=doc_metadata.get("filename", "unknown"),
            page_count=doc_metadata.get("page_count", "unknown"),
            chunks=chunks_text,
        )
        
        # Construct messages for chat-based models
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ]
        
        logger.info(f"Generating with local model: {self.model_name}")
        
        # Helper to apply chat template keying if available, else fallback
        try:
             # Most modern HF pipelines handle list-of-dicts for chat models directly
            outputs = self.pipeline(
                messages, 
                max_new_tokens=self.max_tokens,
                temperature=max(0.1, self.temperature), # ensure > 0
                do_sample=True,
                return_full_text=False
            )
            response_text = outputs[0]["generated_text"]
        except Exception:
            # Fallback for models not supporting list-of-dicts input directly
            prompt_text = f"{VERIFICATION_SYSTEM_PROMPT}\n\n{user_prompt}\n\nResponse:"
            outputs = self.pipeline(
                prompt_text,
                max_new_tokens=self.max_tokens,
                temperature=max(0.1, self.temperature),
                do_sample=True,
                return_full_text=False
            )
            response_text = outputs[0]["generated_text"]

        # Parse JSON
        try:
            # Try to find JSON object in the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
            return VerificationReport(
                query=query,
                doc_id=doc_metadata.get("doc_id", "unknown"),
                summary=data.get("summary", "No summary generated"),
                findings=data.get("findings", []),
                answer=data.get("answer", "No answer generated"),
                confidence=float(data.get("confidence", 0.5)),
                limitations=data.get("limitations", None),
                recommendations=data.get("recommendations", None),
                retrieved_chunks=chunks,
                generation_metadata={"model": self.model_name, "type": "local"}
            )
        except Exception as e:
            logger.error(f"Failed to parse local model output: {e}\nOutput: {response_text}")
            # Return a fallback report with the raw text
            return VerificationReport(
                query=query,
                doc_id=doc_metadata.get("doc_id", "unknown"),
                summary="JSON Parsing Failed",
                findings=[],
                answer=f"Model output (unstructured): {response_text[:500]}...",
                confidence=0.0,
                limitations=f"Failed to generate structured JSON: {str(e)}",
                retrieved_chunks=chunks,
                generation_metadata={"model": self.model_name, "raw_output": response_text}
            )


class MockGenerator(BaseGenerator):
    """
    Mock generator for testing without API/model
    
    Generates a simple report based on template
    """
    
    def generate(
        self,
        query: str,
        chunks: List,
        doc_metadata: Optional[Dict] = None,
    ) -> VerificationReport:
        """Generate mock report"""
        
        doc_metadata = doc_metadata or {}
        
        # Extract chunk IDs for citations
        chunk_ids = [chunk.chunk_id for chunk in chunks[:3]]
        
        # Generate mock findings
        findings = []
        for i, chunk in enumerate(chunks[:3]):
            findings.append({
                "statement": f"Relevant content found in chunk {i+1}",
                "evidence": [chunk.chunk_id],
                "confidence": 0.8 - (i * 0.1),
            })
        
        # Create report
        report = VerificationReport(
            query=query,
            doc_id=doc_metadata.get("doc_id", "unknown"),
            summary=f"Analysis of query: '{query[:50]}...'",
            findings=findings,
            answer=f"Based on {len(chunks)} retrieved chunks, "
                   f"the submission addresses the query with evidence "
                   f"from {', '.join(chunk_ids[:2])}.",
            confidence=0.75,
            limitations="This is a mock report for testing purposes.",
            recommendations="Review the cited chunks for full context.",
            retrieved_chunks=chunks,
            generation_metadata={
                "model": "mock",
                "generator": "MockGenerator",
            },
        )
        
        return report


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def get_generator(
    generator_type: str = GENERATOR_TYPE,
    model_name: str = GENERATOR_MODEL_NAME,
    **kwargs
) -> BaseGenerator:
    """
    Factory function to create appropriate generator
    
    Args:
        generator_type: "groq", "openai", "local", or "mock"
        model_name: Model identifier
        **kwargs: Additional generator-specific parameters
        
    Returns:
        Generator instance
        
    Example:
        >>> gen = get_generator("groq", "mixtral-8x7b-32768")
        >>> report = gen.generate(query, chunks, doc_metadata)
    """
    if generator_type == "groq":
        return GroqGenerator(model_name=model_name, **kwargs)
    elif generator_type == "openai" or generator_type == "api":
        # Keep OpenAI support as fallback
        logger.warning("OpenAI generator not fully implemented, consider using Groq")
        raise NotImplementedError("OpenAI generator - use Groq instead")
    elif generator_type == "local":
        return LocalGenerator(model_name=model_name, **kwargs)
    elif generator_type == "mock":
        return MockGenerator(**kwargs)
    else:
        raise ValueError(
            f"Unknown generator type: {generator_type}. "
            f"Use 'groq', 'openai', 'local', or 'mock'"
        )


def generate_report(
    query: str,
    chunks: List,
    doc_metadata: Optional[Dict] = None,
    generator: Optional[BaseGenerator] = None,
) -> VerificationReport:
    """
    Convenience function to generate a report
    
    Args:
        query: Teacher's question
        chunks: Retrieved chunks
        doc_metadata: Document metadata
        generator: Generator instance (created if None)
        
    Returns:
        VerificationReport
        
    Example:
        >>> from retrieval import retrieve_for_query
        >>> chunks = retrieve_for_query("What is the thesis?", index, embedder)
        >>> report = generate_report(
        ...     "What is the thesis?",
        ...     chunks,
        ...     {"doc_id": "essay_001", "filename": "student_essay.pdf"}
        ... )
        >>> print(report.to_markdown())
    """
    if generator is None:
        generator = get_generator()
    
    return generator.generate(query, chunks, doc_metadata)


# ==============================================================================
# VALIDATION & TESTING
# ==============================================================================

def validate_generation():
    """
    Self-test function to verify generation module
    
    TEST CASES:
    1. Mock generator (no API needed)
    2. Prompt formatting
    3. Report serialization (JSON, markdown)
    4. Citation extraction
    """
    print("=" * 70)
    print("GENERATION MODULE VALIDATION")
    print("=" * 70)
    
    # Test 1: Mock generator
    print("\n[TEST 1] Mock generator")
    
    # Create mock chunks
    from dataclasses import dataclass
    
    @dataclass
    class MockChunk:
        chunk_id: str
        text: str
        score: float
        rank: int
        metadata: dict
        retrieval_stage: str = "mock"
        
        def to_dict(self):
            return {
                "chunk_id": self.chunk_id,
                "text": self.text,
                "score": self.score,
                "rank": self.rank,
                "metadata": self.metadata,
            }
    
    chunks = [
        MockChunk(
            "chunk_001",
            "Machine learning is a subset of AI that focuses on data.",
            0.92,
            0,
            {"doc_id": "essay_001", "page_num": 1}
        ),
        MockChunk(
            "chunk_002",
            "Neural networks are inspired by biological neurons.",
            0.85,
            1,
            {"doc_id": "essay_001", "page_num": 2}
        ),
    ]
    
    query = "What is machine learning?"
    doc_metadata = {
        "doc_id": "essay_001",
        "filename": "student_essay.pdf",
        "page_count": 5,
    }
    
    generator = MockGenerator()
    report = generator.generate(query, chunks, doc_metadata)
    
    print("✓ Generated report with mock generator")
    print(f"  Summary: {report.summary[:80]}...")
    print(f"  Confidence: {report.confidence}")
    print(f"  Findings: {len(report.findings)}")
    
    # Test 2: Prompt formatting
    print("\n[TEST 2] Prompt formatting")
    chunks_text = format_chunks_for_prompt(chunks)
    print(f"✓ Formatted {len(chunks)} chunks for prompt")
    print(f"  Length: {len(chunks_text)} chars")
    assert "chunk_001" in chunks_text, "Should include chunk ID"
    assert "Machine learning" in chunks_text, "Should include chunk text"
    
    # Test 3: Serialization
    print("\n[TEST 3] Report serialization")
    
    # JSON
    json_str = report.to_json()
    print(f"✓ Serialized to JSON ({len(json_str)} chars)")
    json_data = json.loads(json_str)
    assert json_data["query"] == query, "Query should match"
    assert "timestamp" in json_data, "Should include timestamp"
    
    # Markdown
    md_str = report.to_markdown()
    print(f"✓ Serialized to Markdown ({len(md_str)} chars)")
    assert "# Verification Report" in md_str, "Should have title"
    assert query in md_str, "Should include query"
    
    # Dict
    dict_data = report.to_dict()
    print(f"✓ Serialized to dict ({len(dict_data)} keys)")
    
    # Test 4: Citation extraction
    print("\n[TEST 4] Citation validation")
    # Check that findings reference chunk IDs
    for finding in report.findings:
        evidence = finding.get("evidence", [])
        print(f"  Finding: {finding['statement'][:60]}...")
        print(f"    Evidence: {evidence}")
        assert len(evidence) > 0, "Finding should have evidence"
    print("✓ All findings have citations")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nGENERATION SETTINGS:")
    print(f"- Generator type: {GENERATOR_TYPE}")
    print(f"- Model: {GENERATOR_MODEL_NAME}")
    print(f"- Temperature: {GENERATION_TEMPERATURE}")
    print(f"- Max tokens: {GENERATION_MAX_TOKENS}")
    print(f"- Top-p: {GENERATION_TOP_P}")
    print("\nUSAGE:")
    print("  # For testing:")
    print("  generator = MockGenerator()")
    print("  ")
    print("  # For production with OpenAI:")
    print("  export OPENAI_API_KEY=your_key")
    print("  generator = OpenAIGenerator('gpt-3.5-turbo')")
    print("  ")
    print("  # Generate report:")
    print("  report = generator.generate(query, chunks, doc_metadata)")
    print("  print(report.to_markdown())")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    validate_generation()
