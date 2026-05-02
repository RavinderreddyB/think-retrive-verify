
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
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download NLTK data (one-time)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

# ====================================================
# ENV + MODEL INIT
# ====================================================
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

repo_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.3,  # Lower for more deterministic answers
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    max_new_tokens=512,  # Shorter for concise answers
    provider="auto",
)

chat_llm = ChatHuggingFace(llm=llm)

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

class SkepticReport(BaseModel):
    expansion: str
    is_factual: bool
    confidence: float
    contradictions: List[str]
    warnings: List[str]
    verified_claims: List[str]

class CriticResponse(BaseModel):
    expansion: str
    bm25_score: float
    tfidf_score: float
    relevance_score: float
    coverage_score: float
    composite_score: float
    ranking: int

class FinalAnswer(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

# ====================================================
# EXPANSION AGENT (LLM with Few-Shot)
# ====================================================
exp_parser = PydanticOutputParser(pydantic_object=ExpansionResponse)

expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert Query Expansion Agent. Generate diverse search queries. "
     "IMPORTANT: Keep expansions focused and specific. ONLY respond with valid JSON."),
    ("human", 
     "Examples of good expansions:\n"
     "Query: 'Which magazine was started first?'\n"
     "Expansions: ['Arthur\\'s Magazine founding date', 'First for Women publication history', "
     "'magazine founding dates comparison']\n\n"
     "Query: 'What is the capital of France?'\n"
     "Expansions: ['France capital city', 'Paris capital of France', 'French government location']\n\n"
     "Now generate 5 focused expansions for this query:\n"
     "User query: {query}\n\n"
     "Generate:\n"
     "1. Entity/keyword extraction\n"
     "2. Reformulated question\n"
     "3. Specific aspect query\n"
     "4. Comparative/contextual query\n"
     "5. Definition/explanation query\n\n"
     "Output ONLY valid JSON:\n{format_instructions}\n\n"
     "Output:")
]).partial(format_instructions=exp_parser.get_format_instructions())

# ====================================================
# SKEPTIC AGENT (Algorithmic)
# ====================================================
class AlgorithmicSkepticAgent:
    """Advanced fact-checking and contradiction detection"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 
                               'nowhere', 'none', 'cannot', 'cant', "can't", 'won\'t',
                               'wouldn\'t', 'shouldn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 
                               'weren\'t', 'hasn\'t', 'haven\'t', 'hadn\'t', 'doesn\'t', 
                               'don\'t', 'didn\'t'}
        self.uncertainty_words = {'maybe', 'perhaps', 'possibly', 'might', 'could', 
                                 'allegedly', 'reportedly', 'supposedly', 'apparently',
                                 'seems', 'appears', 'likely', 'probably'}
        self.strong_claim_words = {'always', 'never', 'all', 'none', 'every', 'absolutely',
                                  'definitely', 'certainly', 'undoubtedly', 'impossible'}
    
    def analyze_expansion(self, expansion: str, docs: List[str]) -> SkepticReport:
        """Perform skeptical analysis on expansion and retrieved documents"""
        expansion_lower = expansion.lower()
        combined_docs = " ".join(docs).lower()
        
        contradictions = self._detect_contradictions(expansion_lower, combined_docs)
        is_factual, confidence = self._check_factual_support(expansion, docs)
        warnings = self._detect_warnings(expansion_lower, combined_docs)
        verified_claims = self._extract_verified_claims(expansion, docs)
        
        return SkepticReport(
            expansion=expansion,
            is_factual=is_factual,
            confidence=confidence,
            contradictions=contradictions,
            warnings=warnings,
            verified_claims=verified_claims
        )
    
    def _detect_contradictions(self, expansion: str, docs: str) -> List[str]:
        """Detect contradictions between expansion and docs"""
        contradictions = []
        exp_has_negation = any(word in expansion for word in self.negation_words)
        doc_sentences = sent_tokenize(docs)
        
        if exp_has_negation:
            exp_words = set(word_tokenize(expansion)) - self.stop_words - self.negation_words
            for sent in doc_sentences[:10]:
                sent_words = set(word_tokenize(sent.lower())) - self.stop_words
                overlap = exp_words & sent_words
                if len(overlap) >= 2:
                    sent_has_negation = any(word in sent.lower() for word in self.negation_words)
                    if not sent_has_negation:
                        contradictions.append(f"Potential contradiction: {sent[:100]}")
        
        return contradictions
    
    def _check_factual_support(self, expansion: str, docs: List[str]) -> Tuple[bool, float]:
        """Check if expansion is supported by factual evidence"""
        if not docs or all(len(doc.strip()) < 20 for doc in docs):
            return False, 0.1
        
        exp_words = set(word_tokenize(expansion.lower())) - self.stop_words
        total_coverage = 0
        
        for doc in docs:
            doc_words = set(word_tokenize(doc.lower())) - self.stop_words
            if len(exp_words) > 0:
                coverage = len(exp_words & doc_words) / len(exp_words)
                total_coverage += coverage
        
        avg_coverage = total_coverage / len(docs)
        is_factual = avg_coverage > 0.3
        confidence = min(avg_coverage * 2, 1.0)
        
        return is_factual, confidence
    
    def _detect_warnings(self, expansion: str, docs: str) -> List[str]:
        """Detect potential issues in expansion"""
        warnings = []
        
        uncertainty_found = [word for word in self.uncertainty_words if word in expansion]
        if uncertainty_found:
            warnings.append(f"Uncertainty markers: {', '.join(uncertainty_found)}")
        
        strong_claims = [word for word in self.strong_claim_words if word in expansion]
        if strong_claims and len(docs) < 100:
            warnings.append(f"Strong claim with limited evidence: {', '.join(strong_claims)}")
        
        exp_words = word_tokenize(expansion)
        if len(exp_words) <= 2:
            warnings.append("Query too vague")
        
        return warnings
    
    def _extract_verified_claims(self, expansion: str, docs: List[str]) -> List[str]:
        """Extract claims verified by documents"""
        verified = []
        doc_text = " ".join(docs)
        doc_sentences = sent_tokenize(doc_text)
        exp_words = set(word_tokenize(expansion.lower())) - self.stop_words
        
        for sent in doc_sentences[:15]:
            sent_words = set(word_tokenize(sent.lower())) - self.stop_words
            overlap = exp_words & sent_words
            if len(overlap) >= 2 and not sent.strip().endswith('?'):
                verified.append(sent.strip()[:150])
        
        return verified[:3]

# ====================================================
# CRITIC AGENT (BM25 + TF-IDF)
# ====================================================
class BM25CriticAgent:
    """Uses BM25 and TF-IDF for document ranking"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def evaluate_expansions(self, query: str, expansions_with_docs: List[Tuple[str, List[str]]]) -> List[CriticResponse]:
        """Evaluate all expansions using BM25 and TF-IDF"""
        critic_results = []
        all_docs_text = [" ".join(docs) for _, docs in expansions_with_docs]
        
        # TF-IDF
        if all_docs_text:
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_docs_text)
                query_vector = self.tfidf_vectorizer.transform([query])
            except:
                tfidf_matrix = None
        else:
            tfidf_matrix = None
        
        # BM25
        tokenized_docs = [word_tokenize(doc.lower()) for doc in all_docs_text]
        if tokenized_docs and any(tokenized_docs):
            bm25 = BM25Okapi(tokenized_docs)
        else:
            bm25 = None
        
        # Evaluate each expansion
        for idx, (expansion, docs) in enumerate(expansions_with_docs):
            if bm25:
                expansion_tokens = word_tokenize(expansion.lower())
                bm25_score = bm25.get_scores(expansion_tokens)[idx]
                bm25_normalized = min(bm25_score / 10, 1.0)
            else:
                bm25_normalized = 0.5
            
            if tfidf_matrix is not None:
                expansion_vector = self.tfidf_vectorizer.transform([expansion])
                tfidf_score = cosine_similarity(expansion_vector, tfidf_matrix[idx:idx+1])[0][0]
                relevance_score = cosine_similarity(expansion_vector, query_vector)[0][0]
            else:
                tfidf_score = 0.5
                relevance_score = 0.5
            
            doc_text = " ".join(docs)
            coverage_score = min(len(doc_text) / 500, 1.0)
            
            composite_score = (
                0.30 * bm25_normalized +
                0.30 * tfidf_score +
                0.25 * relevance_score +
                0.15 * coverage_score
            )
            
            critic_results.append(CriticResponse(
                expansion=expansion,
                bm25_score=float(bm25_normalized),
                tfidf_score=float(tfidf_score),
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
# SELECTOR AGENT (Algorithmic)
# ====================================================
class AlgorithmicSelectorAgent:
    """Select best expansions based on critic and skeptic"""
    
    @staticmethod
    def select_best_expansions(
        critic_results: List[CriticResponse],
        skeptic_reports: List[SkepticReport],
        top_k: int = 3
    ) -> List[str]:
        """Select top-k expansions with fact-checking"""
        scores = []
        
        for critic, skeptic in zip(critic_results, skeptic_reports):
            base_score = critic.composite_score
            factual_boost = 1.0 + (0.3 * skeptic.confidence if skeptic.is_factual else 0)
            contradiction_penalty = 1.0 - (0.2 * len(skeptic.contradictions))
            contradiction_penalty = max(contradiction_penalty, 0.5)
            warning_penalty = 1.0 - (0.1 * len(skeptic.warnings))
            warning_penalty = max(warning_penalty, 0.7)
            
            final_score = base_score * factual_boost * contradiction_penalty * warning_penalty
            scores.append((critic.expansion, final_score, skeptic.is_factual))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        for expansion, score, is_factual in scores:
            if is_factual and len(selected) < top_k:
                selected.append(expansion)
        
        if len(selected) < top_k:
            for expansion, score, is_factual in scores:
                if expansion not in selected and len(selected) < top_k:
                    selected.append(expansion)
        
        return selected

# ====================================================
# ANSWER EXTRACTION (NEW - for better EM)
# ====================================================
def extract_short_answer(full_answer: str, query: str) -> str:
    """Extract concise answer from longer response for better EM"""
    
    query_lower = query.lower()
    
    # For yes/no questions
    if full_answer.lower().strip().startswith(('yes', 'no')):
        return full_answer.split()[0].lower()
    
    # For "which" questions - extract entity
    if query_lower.startswith('which'):
        sentences = sent_tokenize(full_answer)
        if sentences:
            first_sent = sentences[0]
            # Look for quoted text or capitalized phrases
            quoted = re.findall(r'"([^"]*)"', first_sent)
            if quoted:
                return quoted[0]
            # Otherwise first few words
            words = first_sent.split()[:8]
            return ' '.join(words).rstrip('.,!?;:')
    
    # For "when" questions - extract year/date
    if query_lower.startswith('when'):
        # Look for years
        years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', full_answer)
        if years:
            return years[0]
    
    # For "who" questions - extract name (capitalized words)
    if query_lower.startswith('who'):
        words = full_answer.split()
        capitalized = [w for w in words if w[0].isupper() and len(w) > 1]
        if capitalized:
            return ' '.join(capitalized[:3])
    
    # Default: first sentence, max 10 words
    sentences = sent_tokenize(full_answer)
    if sentences:
        first_sent = sentences[0]
        words = first_sent.split()[:10]
        return ' '.join(words).rstrip('.,!?;:')
    
    return full_answer

# ====================================================
# ANSWER GENERATOR (LLM with Few-Shot)
# ====================================================
answer_parser = PydanticOutputParser(pydantic_object=FinalAnswer)

# Create few-shot examples as separate messages (no curly brace conflicts)
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert at extracting precise answers from documents. "
     "Study these examples carefully:"),
    ("human", 
     "Query: Which magazine was started first Arthur's Magazine or First for Women?\n"
     "Documents: Arthur's Magazine (1844-1846) was published in Philadelphia. "
     "First for Women was launched in 1989.\n\n"
     "Provide answer as JSON with fields: answer, sources, confidence"),
    ("assistant", 
     "Here is my response:\n"
     "answer: Arthur's Magazine\n"
     "sources: Arthur's Magazine 1844-1846\n"
     "confidence: 0.95"),
    ("human",
     "Query: What is the capital of France?\n"
     "Documents: Paris is the capital and largest city of France.\n\n"
     "Provide answer as JSON with fields: answer, sources, confidence"),
    ("assistant",
     "Here is my response:\n"
     "answer: Paris\n"
     "sources: Paris capital of France\n"
     "confidence: 0.98"),
    ("human", 
     "Now answer this query following the same pattern.\n\n"
     "CRITICAL INSTRUCTIONS:\n"
     "- Give ONLY the direct answer, not explanations\n"
     "- For 'which/what/who': Just the name/entity (2-5 words)\n"
     "- For 'when': Just the date/year\n"
     "- For 'yes/no': Just 'yes' or 'no'\n"
     "- Extract exact information from documents\n"
     "- Keep answers under 10 words\n\n"
     "Query: {query}\n\n"
     "Documents:\n{docs}\n\n"
     "Output ONLY valid JSON following this format:\n{format_instructions}\n\n"
     "Your response:")
]).partial(format_instructions=answer_parser.get_format_instructions())

# ====================================================
# VERIFIER (Algorithmic)
# ====================================================
class AlgorithmicVerifier:
    """Cross-check answer against source documents"""
    
    @staticmethod
    def verify_answer(answer: str, docs: List[str]) -> Dict:
        """Verify answer claims against documents"""
        answer_sentences = sent_tokenize(answer)
        verified_count = 0
        unverified_count = 0
        
        doc_text = " ".join(docs).lower()
        doc_words = set(word_tokenize(doc_text))
        stop_words = set(stopwords.words('english'))
        
        for sent in answer_sentences:
            sent_words = set(word_tokenize(sent.lower())) - stop_words
            overlap = sent_words & doc_words
            coverage = len(overlap) / max(len(sent_words), 1)
            
            if coverage > 0.5:
                verified_count += 1
            else:
                unverified_count += 1
        
        verification_rate = verified_count / max(len(answer_sentences), 1)
        
        return {
            "verification_rate": verification_rate,
            "verified_claims": verified_count,
            "unverified_claims": unverified_count,
            "is_trustworthy": verification_rate > 0.7,
            "assessment": f"{verification_rate*100:.1f}% of claims verified in source documents"
        }

# ====================================================
# HELPER FUNCTIONS
# ====================================================
def extract_json_from_text(text: str) -> Optional[dict]:
    if not text:
        return None
    
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
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
        result = parser.parse(raw_text)
        print("✅ Parsed successfully")
        return result
    except (OutputParserException, Exception) as e:
        print(f"⚠️ Parsing failed, trying JSON extraction...")
        json_obj = extract_json_from_text(raw_text)
        if json_obj:
            try:
                result = parser.pydantic_object.model_validate(json_obj)
                print("✅ Parsed via JSON extraction")
                return result
            except Exception as e2:
                print(f"⚠️ JSON extraction failed: {e2}")
        print("🔄 Using fallback")
        return default_factory()

def default_expansion():
    return ExpansionResponse(expansions=[
        ExpansionItem(id=1, text="general information", type="semantic", description="broad search"),
        ExpansionItem(id=2, text="definition", type="keyword", description="meaning"),
        ExpansionItem(id=3, text="explanation", type="explanation", description="how it works"),
    ])

def default_answer(query):
    return FinalAnswer(
        answer=f"Unable to find reliable answer for: {query}",
        sources=[],
        confidence=0.1
    )

# ====================================================
# FINAL OPTIMIZED PIPELINE
# ====================================================
def multi_agent_pipeline(query: str):
    """
    FINAL VERSION with improved answer extraction
    ~2 LLM API calls, maximum accuracy on HotpotQA
    """
    print(f"\n🔍 Processing query: {query}\n")
    api_call_count = 0
    
    skeptic_agent = AlgorithmicSkepticAgent()
    critic_agent = BM25CriticAgent()
    selector_agent = AlgorithmicSelectorAgent()
    verifier = AlgorithmicVerifier()
    
    # STEP 1: LLM Expansion (1 API call)
    print("📝 Generating expansions (LLM with few-shot)...")
    expansions_raw = call_llm(expansion_prompt.format(query=query))
    api_call_count += 1
    expansions = safe_parse(exp_parser, expansions_raw, default_expansion)
    print(f"✅ Generated {len(expansions.expansions)} expansions")
    
    # STEP 2: Retrieve documents
    print("\n🔎 Retrieving documents...")
    expansions_with_docs = []
    
    for exp in expansions.expansions[:5]:
        try:
            search_result = search_tool.run(exp.text)
            wiki_result = wiki_tool.run(exp.text)
            docs = [search_result[:600], wiki_result[:600]]
        except Exception as e:
            print(f"  ⚠️ Search failed: {e}")
            docs = ["No results"]
        
        expansions_with_docs.append((exp.text, docs))
    
    # STEP 3: Skeptic Analysis (0 API calls)
    print("\n🤔 Skeptic analysis (fact-checking)...")
    skeptic_reports = []
    
    for expansion, docs in expansions_with_docs:
        report = skeptic_agent.analyze_expansion(expansion, docs)
        skeptic_reports.append(report)
        
        if not report.is_factual:
            print(f"  ⚠️ Low confidence: {expansion[:50]}... ({report.confidence:.2f})")
    
    print(f"✅ Analyzed {len(skeptic_reports)} expansions")
    
    # STEP 4: BM25 Critic (0 API calls)
    print("\n📊 Critic evaluation (BM25 + TF-IDF)...")
    critic_results = critic_agent.evaluate_expansions(query, expansions_with_docs)
    print(f"✅ Ranked {len(critic_results)} expansions")
    
    # STEP 5: Selector (0 API calls)
    print("\n🎯 Selecting best verified expansions...")
    selected_expansions = selector_agent.select_best_expansions(
        critic_results, skeptic_reports, top_k=3
    )
    print(f"✅ Selected {len(selected_expansions)} verified expansions")
    
    # STEP 6: Answer Generator (1 API call)
    print("\n💡 Generating final answer (LLM with few-shot)...")
    
    all_docs = []
    for exp_text in selected_expansions:
        try:
            all_docs.append(search_tool.run(exp_text)[:600])
            all_docs.append(wiki_tool.run(exp_text)[:600])
        except Exception as e:
            print(f"  ⚠️ Failed to retrieve: {e}")
    
    docs_str = "\n\n---\n\n".join(all_docs)[:2500]
    
    answer_raw = call_llm(answer_prompt.format(query=query, docs=docs_str))
    api_call_count += 1
    final_answer = safe_parse(answer_parser, answer_raw, lambda: default_answer(query))
    
    # STEP 6.5: Extract short answer (NEW - improves EM)
    short_answer = extract_short_answer(final_answer.answer, query)
    final_answer.answer = short_answer
    
    # STEP 7: Algorithmic Verification (0 API calls)
    print("\n✓ Verifying answer...")
    verification_result = verifier.verify_answer(final_answer.answer, all_docs)
    
    print(f"  {verification_result['assessment']}")
    
    # Save output
    save_message = save_tool.run(final_answer.model_dump_json())
    print(f"\n{save_message}")
    
    print(f"\n📞 Total LLM API calls: {api_call_count}")
    
    return {
        "query": query,
        "expansions": [e.model_dump() for e in expansions.expansions],
        "skeptic_reports": [r.model_dump() for r in skeptic_reports],
        "critic": [r.model_dump() for r in critic_results],
        "selected_expansions": selected_expansions,
        "final_answer": final_answer.model_dump(),
        "verification": verification_result,
        "save_status": save_message,
        "api_calls": api_call_count,
        "approach": "final_optimized_improved"
    }

# ====================================================
# RUN
# ====================================================
if __name__ == "__main__":
    query = input("What can I help you research? ")
    result = multi_agent_pipeline(query)
    
    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print("="*60)
    print(result["final_answer"]["answer"])
    print(f"\nConfidence: {result['final_answer']['confidence']:.2f}")
    print(f"Verification: {result['verification']['assessment']}")
    print("\n" + "="*60)
    print(f"📞 LLM API Calls: {result['api_calls']}")
    print(f"✅ Improvements: Few-shot prompting + Answer extraction")
    print("="*60)