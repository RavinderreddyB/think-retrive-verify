

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from tools import search_tool, wiki_tool, save_tool
import os
import json
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# ====================================================
# ENV + MODEL INIT
# ====================================================
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

repo_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.1,  # Lower for more deterministic answers
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    max_new_tokens=256,  # Shorter for concise answers
    provider="auto",
)

chat_llm = ChatHuggingFace(llm=llm)

# Load semantic model for hybrid retrieval
print("Loading semantic embeddings model...")
try:
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Semantic model loaded")
except:
    print("⚠️ Semantic model failed, using BM25 only")
    semantic_model = None

# ====================================================
# STRUCTURED OUTPUTS
# ====================================================
class ExpansionItem(BaseModel):
    id: int
    text: str
    type: str
    description: str

class ExpansionResponse(BaseModel):
    expansions: List[ExpansionItem]

class SkepticQuery(BaseModel):
    id: int
    original_claim: str
    skeptic_query: str
    query_type: str
    reasoning: str

class SkepticResponse(BaseModel):
    skeptic_queries: List[SkepticQuery]

class SkepticReport(BaseModel):
    original_expansion: str
    skeptic_queries_generated: List[str]
    warnings_found: List[str]
    contradictions_found: List[str]
    safety_concerns: List[str]
    is_safe: bool
    confidence: float
    recommendation: str

class CriticResponse(BaseModel):
    expansion: str
    bm25_score: float
    semantic_score: float  # Changed from tfidf_score
    relevance_score: float
    coverage_score: float
    composite_score: float
    ranking: int

class FinalAnswer(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

# ====================================================
# QUERY DECOMPOSER (NEW - IMPROVEMENT #2)
# ====================================================
class QueryDecomposer:
    """Decompose multi-hop queries into sub-questions"""
    
    def decompose(self, query: str) -> List[str]:
        """Break complex queries into simpler sub-queries"""
        query_lower = query.lower()
        
        if not self._is_multihop(query_lower):
            return [query]
        
        sub_queries = []
        
        # Pattern: "Which came first, X or Y?"
        if 'which' in query_lower and ' or ' in query_lower:
            parts = query.split(' or ')
            if len(parts) == 2:
                entity1 = parts[0].replace('Which', '').replace('which', '').strip()
                entity2 = parts[1].strip().rstrip('?')
                sub_queries.append(f"When was {entity1}?")
                sub_queries.append(f"When was {entity2}?")
        
        # Pattern: "What year was X born who did Y?"
        elif 'who' in query_lower and any(w in query_lower for w in ['what', 'when', 'where']):
            entity_match = re.search(r'([\w\s]+) who ', query)
            if entity_match:
                entity = entity_match.group(1).strip()
                sub_queries.append(f"Who is {entity}?")
        
        # Always include original query
        if query not in sub_queries:
            sub_queries.append(query)
        
        return sub_queries[:3]
    
    def _is_multihop(self, query: str) -> bool:
        """Detect if query requires multi-hop reasoning"""
        return any([
            len(re.findall(r'\b(who|what|when|where)\b', query)) > 1,
            ' who ' in query,
            ' or ' in query and 'which' in query,
        ])

# ====================================================
# EXPANSION AGENT
# ====================================================
exp_parser = PydanticOutputParser(pydantic_object=ExpansionResponse)

expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Query Expansion Agent. Generate focused, concise search queries. ONLY respond with valid JSON."),
    ("human", 
     "Query: {query}\n\n"
     "Generate 5 focused expansions (each 2-8 words):\n"
     "1. Entity/keyword extraction\n"
     "2. Reformulated question\n"
     "3. Specific aspect\n"
     "4. Temporal/contextual\n"
     "5. Alternative phrasing\n\n"
     "Output ONLY valid JSON:\n{format_instructions}\n\nOutput:")
]).partial(format_instructions=exp_parser.get_format_instructions())

# ====================================================
# SKEPTIC AGENT
# ====================================================
skeptic_parser = PydanticOutputParser(pydantic_object=SkepticResponse)

skeptic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a SKEPTIC AGENT. Generate 2-3 counter-questions checking for safety/contradictions. ONLY respond with valid JSON."),
    ("human",
     "Expansion: {expansion}\n\n"
     "Generate 2-3 skeptic queries checking for:\n"
     "1. Safety/toxicity\n"
     "2. Contradictions\n"
     "3. Side effects\n\n"
     "Output ONLY valid JSON:\n{format_instructions}\n\nOutput:")
]).partial(format_instructions=skeptic_parser.get_format_instructions())

class ActiveSkepticAgent:
    def __init__(self, llm_caller):
        self.llm_caller = llm_caller
        self.safety_keywords = {
            'toxic', 'poison', 'harmful', 'dangerous', 'death', 'lethal',
            'overdose', 'side effects', 'adverse', 'risk', 'hazard'
        }
    
    def analyze_expansion(self, expansion: str) -> SkepticReport:
        """Generate skeptic queries and check safety"""
        
        skeptic_queries = self._generate_skeptic_queries(expansion)
        safety_concerns = []
        contradictions = []
        warnings = []
        
        for query_obj in skeptic_queries[:2]:  # Limit to 2 to save time
            try:
                search_result = search_tool.run(query_obj.skeptic_query)
                docs = [search_result[:400]]
                
                analysis = self._analyze_safety(docs)
                safety_concerns.extend(analysis['concerns'])
                contradictions.extend(analysis['contradictions'])
                warnings.extend(analysis['warnings'])
            except:
                pass
        
        total_issues = len(safety_concerns) + len(contradictions) + len(warnings)
        is_safe = total_issues == 0
        confidence = max(0.0, 1.0 - (total_issues * 0.15))
        
        if not is_safe:
            recommendation = f"⚠️ Found {total_issues} concerns"
        else:
            recommendation = "✅ Safe"
        
        return SkepticReport(
            original_expansion=expansion,
            skeptic_queries_generated=[q.skeptic_query for q in skeptic_queries],
            warnings_found=warnings[:3],
            contradictions_found=contradictions[:3],
            safety_concerns=safety_concerns[:3],
            is_safe=is_safe,
            confidence=confidence,
            recommendation=recommendation
        )
    
    def _generate_skeptic_queries(self, expansion: str) -> List[SkepticQuery]:
        try:
            skeptic_raw = self.llm_caller(skeptic_prompt.format(expansion=expansion))
            skeptic_response = safe_parse(skeptic_parser, skeptic_raw, lambda: SkepticResponse(skeptic_queries=[]))
            return skeptic_response.skeptic_queries[:2] if skeptic_response.skeptic_queries else self._fallback_queries(expansion)
        except:
            return self._fallback_queries(expansion)
    
    def _fallback_queries(self, expansion: str) -> List[SkepticQuery]:
        return [
            SkepticQuery(id=1, original_claim=expansion, skeptic_query=f"Is {expansion} safe?", query_type="safety", reasoning="Safety check"),
            SkepticQuery(id=2, original_claim=expansion, skeptic_query=f"Side effects of {expansion}?", query_type="side_effects", reasoning="Check effects")
        ]
    
    def _analyze_safety(self, docs: List[str]) -> Dict:
        combined = " ".join(docs).lower()
        safety_found = [kw for kw in self.safety_keywords if kw in combined]
        
        concerns = []
        contradictions = []
        warnings = []
        
        for doc in docs:
            for sent in sent_tokenize(doc)[:5]:
                sent_lower = sent.lower()
                if any(kw in sent_lower for kw in ['toxic', 'poison', 'harmful', 'dangerous']):
                    concerns.append(sent[:150])
                if any(kw in sent_lower for kw in ['however', 'contrary', 'disputed']):
                    contradictions.append(sent[:150])
                if any(kw in sent_lower for kw in ['warning', 'caution', 'avoid']):
                    warnings.append(sent[:150])
        
        return {
            'concerns': list(set(concerns))[:2],
            'contradictions': list(set(contradictions))[:2],
            'warnings': list(set(warnings))[:2]
        }

# ====================================================
# HYBRID CRITIC AGENT (IMPROVED - BM25 + SEMANTIC)
# ====================================================
class HybridCriticAgent:
    """Hybrid BM25 + Semantic Embeddings"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2))
        self.semantic_model = semantic_model
    
    def evaluate_expansions(self, query: str, expansions_with_docs: List[Tuple[str, List[str]]]) -> List[CriticResponse]:
        """Hybrid ranking: BM25 + Semantic embeddings"""
        
        critic_results = []
        all_docs_text = [" ".join(docs) for _, docs in expansions_with_docs]
        
        # BM25 scoring
        tokenized_docs = [word_tokenize(doc.lower()) for doc in all_docs_text]
        bm25 = BM25Okapi(tokenized_docs) if any(tokenized_docs) else None
        
        # Semantic embeddings (if available)
        if self.semantic_model and all_docs_text:
            try:
                query_emb = self.semantic_model.encode(query)
                doc_embs = self.semantic_model.encode(all_docs_text)
            except:
                query_emb = None
                doc_embs = None
        else:
            query_emb = None
            doc_embs = None
        
        for idx, (expansion, docs) in enumerate(expansions_with_docs):
            # BM25 score
            if bm25:
                expansion_tokens = word_tokenize(expansion.lower())
                bm25_score = bm25.get_scores(expansion_tokens)[idx]
                bm25_normalized = min(bm25_score / 10, 1.0)
            else:
                bm25_normalized = 0.5
            
            # Semantic score
            if query_emb is not None and doc_embs is not None:
                expansion_emb = self.semantic_model.encode(expansion)
                semantic_score = float(cosine_similarity([expansion_emb], [doc_embs[idx]])[0][0])
                relevance_score = float(cosine_similarity([expansion_emb], [query_emb])[0][0])
            else:
                semantic_score = 0.5
                relevance_score = 0.5
            
            # Coverage
            doc_text = " ".join(docs)
            coverage_score = min(len(doc_text) / 500, 1.0)
            
            # Hybrid composite: Favor semantic + BM25
            composite_score = (
                0.35 * bm25_normalized +       # Lexical
                0.35 * semantic_score +        # Semantic (NEW)
                0.20 * relevance_score +       # Query relevance
                0.10 * coverage_score          # Document length
            )
            
            critic_results.append(CriticResponse(
                expansion=expansion,
                bm25_score=float(bm25_normalized),
                semantic_score=float(semantic_score),
                relevance_score=float(relevance_score),
                coverage_score=float(coverage_score),
                composite_score=float(composite_score),
                ranking=0
            ))
        
        critic_results.sort(key=lambda x: x.composite_score, reverse=True)
        for rank, result in enumerate(critic_results, 1):
            result.ranking = rank
        
        return critic_results

# ====================================================
# SELECTOR AGENT
# ====================================================
class AlgorithmicSelectorAgent:
    @staticmethod
    def select_best_expansions(
        critic_results: List[CriticResponse],
        skeptic_reports: List[SkepticReport],
        top_k: int = 3
    ) -> List[str]:
        """Select top-k safe expansions"""
        
        scores = []
        for critic, skeptic in zip(critic_results, skeptic_reports):
            base_score = critic.composite_score
            safety_penalty = 0.1 if not skeptic.is_safe else 1.0
            confidence_boost = 1.0 + (0.2 * skeptic.confidence)
            final_score = base_score * safety_penalty * confidence_boost
            scores.append((critic.expansion, final_score, skeptic.is_safe))
        
        scores.sort(key=lambda x: (x[2], x[1]), reverse=True)
        selected = [exp for exp, _, is_safe in scores[:top_k] if is_safe]
        
        if not selected:
            selected = [scores[0][0]] if scores else []
        
        return selected

# ====================================================
# ANSWER EXTRACTION (IMPROVED - MORE AGGRESSIVE)
# ====================================================
def extract_short_answer(full_answer: str, query: str) -> str:
    """Extract concise answer - AGGRESSIVE for HotpotQA"""
    
    query_lower = query.lower()
    
    # Yes/no
    if full_answer.lower().strip().startswith(('yes', 'no')):
        return full_answer.split()[0].lower()
    
    # Which questions - extract just the entity
    if query_lower.startswith('which'):
        # Look for quoted text first
        quoted = re.findall(r'"([^"]*)"', full_answer)
        if quoted:
            return quoted[0]
        
        # Look for capitalized proper nouns
        words = full_answer.split()
        # Find first sequence of capitalized words
        cap_seq = []
        for word in words:
            if word and word[0].isupper() and len(word) > 1:
                cap_seq.append(word)
                if len(cap_seq) >= 4:  # Max 4 words
                    break
            elif cap_seq:
                break
        
        if cap_seq:
            return ' '.join(cap_seq).rstrip('.,!?;:')
        
        # Fallback: first 3-5 words
        return ' '.join(words[:5]).rstrip('.,!?;:')
    
    # When questions - extract year/date
    if query_lower.startswith('when'):
        years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', full_answer)
        if years:
            return years[0]
    
    # Who questions - extract name
    if query_lower.startswith('who'):
        words = full_answer.split()
        cap_words = [w for w in words[:10] if w and len(w) > 1 and w[0].isupper()]
        if cap_words:
            return ' '.join(cap_words[:3]).rstrip('.,!?;:')
    
    # Default: first sentence, max 8 words
    sentences = sent_tokenize(full_answer)
    if sentences:
        first_sent = sentences[0]
        # Remove common prefixes
        for prefix in ['The answer is', 'It is', 'It was', 'This is']:
            if first_sent.startswith(prefix):
                first_sent = first_sent[len(prefix):].strip()
        words = first_sent.split()[:8]
        return ' '.join(words).rstrip('.,!?;:')
    
    return full_answer[:50].rstrip('.,!?;:')

# ====================================================
# ANSWER GENERATOR (IMPROVED PROMPT)
# ====================================================
answer_parser = PydanticOutputParser(pydantic_object=FinalAnswer)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at extracting precise, SHORT answers."),
    ("human", 
     "Examples:\n"
     "Q: Which magazine was started first Arthur's Magazine or First for Women?\n"
     "A: Arthur's Magazine\n\n"
     "Q: What is the capital of France?\n"
     "A: Paris\n\n"
     "RULES:\n"
     "- Answer in 1-5 words ONLY\n"
     "- NO explanations\n"
     "- Just the direct answer\n\n"
     "Query: {query}\n"
     "Context: {docs}\n\n"
     "Output JSON:\n{format_instructions}\n\nAnswer:")
]).partial(format_instructions=answer_parser.get_format_instructions())

# ====================================================
# VERIFIER
# ====================================================
class AlgorithmicVerifier:
    @staticmethod
    def verify_answer(answer: str, docs: List[str]) -> Dict:
        """Quick verification"""
        answer_words = set(word_tokenize(answer.lower()))
        doc_text = " ".join(docs).lower()
        doc_words = set(word_tokenize(doc_text))
        
        overlap = answer_words & doc_words
        coverage = len(overlap) / max(len(answer_words), 1)
        
        return {
            "verification_rate": coverage,
            "is_trustworthy": coverage > 0.6,
            "assessment": f"{coverage*100:.0f}% verified"
        }

# ====================================================
# HELPER FUNCTIONS
# ====================================================
def extract_json_from_text(text: str) -> Optional[dict]:
    if not text:
        return None
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    return None

def call_llm(prompt):
    response = chat_llm.invoke(prompt)
    if isinstance(response, list) and len(response) > 0:
        return response[-1].content
    if hasattr(response, "content"):
        return response.content
    return str(response)

def safe_parse(parser, raw_text: str, default_factory):
    try:
        return parser.parse(raw_text)
    except:
        json_obj = extract_json_from_text(raw_text)
        if json_obj:
            try:
                return parser.pydantic_object.model_validate(json_obj)
            except:
                pass
        return default_factory()

def default_expansion():
    return ExpansionResponse(expansions=[
        ExpansionItem(id=1, text="general information", type="semantic", description="broad"),
        ExpansionItem(id=2, text="definition", type="keyword", description="meaning"),
    ])

def default_answer(query):
    return FinalAnswer(answer="Unknown", sources=[], confidence=0.1)

# ====================================================
# MAIN PIPELINE (OPTIMIZED WITH PARALLELIZATION)
# ====================================================
def multi_agent_pipeline(query: str):
    """
    OPTIMIZED PIPELINE with:
    - Query decomposition
    - Hybrid retrieval
    - Parallel processing
    - Aggressive answer extraction
    """
    print(f"\n🔍 Query: {query}\n")
    start_time = time.time()
    api_call_count = 0
    
    # Initialize agents
    decomposer = QueryDecomposer()
    skeptic_agent = ActiveSkepticAgent(call_llm)
    critic_agent = HybridCriticAgent()
    selector_agent = AlgorithmicSelectorAgent()
    verifier = AlgorithmicVerifier()
    executor = ThreadPoolExecutor(max_workers=3)
    
    # STEP 1: Query Decomposition (NEW)
    sub_queries = decomposer.decompose(query)
    if len(sub_queries) > 1:
        print(f"📋 Decomposed into {len(sub_queries)} sub-queries")
    
    # STEP 2: Expansion (1 API call per sub-query, but we limit to main query)
    print("📝 Generating expansions...")
    expansions_raw = call_llm(expansion_prompt.format(query=sub_queries[-1]))  # Use most specific
    api_call_count += 1
    expansions = safe_parse(exp_parser, expansions_raw, default_expansion)
    print(f"✅ {len(expansions.expansions)} expansions")
    
    # STEP 3: Parallel Retrieval
    print("\n🔎 Parallel retrieval...")
    retrieval_futures = {}
    for exp in expansions.expansions[:5]:
        future = executor.submit(lambda e: (search_tool.run(e)[:600], wiki_tool.run(e)[:600]), exp.text)
        retrieval_futures[future] = exp.text
    
    expansions_with_docs = []
    for future in as_completed(retrieval_futures, timeout=10):
        exp_text = retrieval_futures[future]
        try:
            search_result, wiki_result = future.result()
            expansions_with_docs.append((exp_text, [search_result, wiki_result]))
        except:
            expansions_with_docs.append((exp_text, ["No results"]))
    
    # STEP 4: Skeptic Analysis (top 2 only, parallel)
    print("\n🤔 Skeptic check...")
    skeptic_futures = []
    for exp_text, docs in expansions_with_docs[:2]:
        future = executor.submit(skeptic_agent.analyze_expansion, exp_text)
        skeptic_futures.append(future)
    
    skeptic_reports = []
    for future in as_completed(skeptic_futures, timeout=12):
        try:
            skeptic_reports.append(future.result())
            api_call_count += 1
        except:
            pass
    
    # Pad skeptic reports
    while len(skeptic_reports) < len(expansions_with_docs):
        skeptic_reports.append(SkepticReport(
            original_expansion="", skeptic_queries_generated=[], warnings_found=[],
            contradictions_found=[], safety_concerns=[], is_safe=True,
            confidence=0.5, recommendation="Not analyzed"
        ))
    
    # STEP 5: Hybrid Critic
    print("\n📊 Hybrid ranking (BM25 + Semantic)...")
    critic_results = critic_agent.evaluate_expansions(query, expansions_with_docs)
    
    # STEP 6: Selector
    print("🎯 Selecting best expansions...")
    selected_expansions = selector_agent.select_best_expansions(critic_results, skeptic_reports, top_k=3)
    
    # STEP 7: Answer Generation (1 API call)
    print("\n💡 Generating answer...")
    all_docs = []
    for exp_text in selected_expansions[:2]:  # Top 2 only
        try:
            all_docs.append(search_tool.run(exp_text)[:500])
        except:
            pass
    
    docs_str = "\n\n".join(all_docs)[:1500]
    
    answer_raw = call_llm(answer_prompt.format(query=query, docs=docs_str))
    api_call_count += 1
    final_answer = safe_parse(answer_parser, answer_raw, lambda: default_answer(query))
    
    # STEP 8: Aggressive Answer Extraction
    short_answer = extract_short_answer(final_answer.answer, query)
    final_answer.answer = short_answer
    
    # STEP 9: Quick Verification
    verification = verifier.verify_answer(final_answer.answer, all_docs)
    
    elapsed = time.time() - start_time
    
    # Save
    save_tool.run(final_answer.model_dump_json())
    
    print(f"\n✅ Complete in {elapsed:.1f}s ({api_call_count} API calls)")
    
    executor.shutdown(wait=False)
    
    return {
        "query": query,
        "expansions": [e.model_dump() for e in expansions.expansions],
        "skeptic_reports": [r.model_dump() for r in skeptic_reports],
        "critic": [r.model_dump() for r in critic_results],
        "selected_expansions": selected_expansions,
        "final_answer": final_answer.model_dump(),
        "verification": verification,
        "api_calls": api_call_count,
        "latency": elapsed,
        "approach": "optimized_hybrid_parallel"
    }

# ====================================================
# RUN
# ====================================================
if __name__ == "__main__":
    query = input("What can I help you research? ")
    result = multi_agent_pipeline(query)
    
    print("\n" + "="*60)
    print("ANSWER:", result["final_answer"]["answer"])
    print(f"Confidence: {result['final_answer']['confidence']:.2f}")
    print(f"Verified: {result['verification']['assessment']}")
    print(f"Time: {result['latency']:.1f}s | API: {result['api_calls']}")
    print("="*60)