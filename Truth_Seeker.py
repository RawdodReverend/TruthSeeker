import os, glob, shutil, hashlib, re, json, webbrowser, tempfile
import chromadb
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re

# --- CONFIG ---
LM_STUDIO_API = "http://192.168.1.171:1234/v1"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
CHAT_MODEL = "lmstudio-community/gemma-3-27b-it"
PERSIST_PATH = "./chroma_db"
MODEL_LIMITS = {
    "text-embedding-all-minilm-l6-v2-embedding": 512,
    "text-embedding-nomic-embed-text-v1.5": 8192,
}
MAX_TOKENS = MODEL_LIMITS.get(EMBED_MODEL, 8192)
CHUNK_TOKENS = 400  # Larger chunks for big datasets
OVERLAP_TOKENS = 80

@dataclass
class SearchResult:
    document: str
    metadata: dict
    distance: float
    relevance_score: float = 0.0
    source_query: str = ""
    iteration: int = 0

@dataclass
class ResearchNode:
    query: str
    results: List[SearchResult]
    iteration: int
    parent_queries: List[str] = field(default_factory=list)
    children_queries: List[str] = field(default_factory=list)
    findings: str = ""
    importance: float = 0.0

@dataclass 
class ResearchReport:
    original_query: str
    total_iterations: int
    total_searches: int
    research_tree: List[ResearchNode]
    final_synthesis: str
    critical_findings: List[str]
    timestamp: str

class RecursiveRAGAgent:
    def __init__(self, num_shards=5):
        self.embed_client = OpenAI(api_key="not-needed", base_url=LM_STUDIO_API)
        self.chat_client = OpenAI(api_key="not-needed", base_url=LM_STUDIO_API)
        self.chroma_client = chromadb.PersistentClient(path=PERSIST_PATH)
        self.num_shards = num_shards
        self.shards = []
        for i in range(self.num_shards):
            shard_name = f"shard_{i:02d}"
            collection = self.chroma_client.get_or_create_collection(shard_name)
            self.shards.append(collection)
        self.research_history = []
        self.query_genealogy = {}
        self.conversation_memory = []  # Track conversation for follow-ups
        self.last_research_report = None  # Store last research for follow-ups

    def _get_shard_for_id(self, doc_id: str) -> int:
        """Deterministically assign a document to a shard based on its ID."""
        return hash(doc_id) % self.num_shards

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    def smart_chunk_text(self, text, size=CHUNK_TOKENS*4, overlap=OVERLAP_TOKENS*4):
        """Smart chunking that preserves document structure."""
        text = self.preprocess_text(text)
        # Try to chunk by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) > size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep some overlap from end of previous chunk
                overlap_text = ' '.join(current_chunk.split()[-overlap//4:])
                current_chunk = overlap_text + " " + para
            else:
                current_chunk += ("\n" if current_chunk else "") + para
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        # If paragraphs are too big, fall back to sentence chunking
        if len(chunks) <= 1 and len(text) > size:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > size and current_chunk:
                    chunks.append(current_chunk.strip())
                    overlap_words = ' '.join(current_chunk.split()[-overlap//4:])
                    current_chunk = overlap_words + " " + sentence
                else:
                    current_chunk += (" " if current_chunk else "") + sentence
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        return chunks if chunks else [text]

    def search_with_query(self, query: str, n_results: int = 15, iteration: int = 0) -> List[SearchResult]:
        """Execute search across ALL shards and merge results."""
        try:
            q_emb = self.embed_client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
            
            # Query all shards in parallel
            all_results = []
            for i, collection in enumerate(self.shards):
                try:
                    results = collection.query(
                        query_embeddings=[q_emb],
                        n_results=n_results,  # Get top N from each shard
                        include=["documents", "metadatas", "distances"]
                    )
                    if results["documents"] and results["documents"][0]:
                        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                            all_results.append(SearchResult(
                                document=doc,
                                metadata=meta or {},
                                distance=dist,
                                source_query=query,
                                iteration=iteration
                            ))
                except Exception as e:
                    print(f"⚠️ Search error in shard_{i:02d}: {e}")
                    continue

            # Deduplicate by document preview (first 100 chars)
            seen = set()
            unique_results = []
            for result in all_results:
                key = result.document[:100]
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)

            # Sort by distance (lower is better) and return top n_results
            unique_results.sort(key=lambda x: x.distance)
            return unique_results[:n_results]

        except Exception as e:
            print(f"⚠️ Search error for '{query}': {e}")
            return []

    def evaluate_and_generate_followups(self, original_query: str, current_query: str, results: List[SearchResult]) -> Tuple[List[SearchResult], List[str], str]:
        """Score relevance and generate follow-up queries."""
        if not results:
            return results, [], ""
        docs_text = ""
        for i, result in enumerate(results):
            preview = result.document[:300] + "..." if len(result.document) > 300 else result.document
            docs_text += f"\nDocument {i+1}: {preview}\n"
        analysis_prompt = f"""RESEARCH ANALYSIS
Original Question: "{original_query}"
Current Search: "{current_query}"
Found Documents:
{docs_text}
TASKS:
1. RELEVANCE: Rate each document's relevance to the original question (1-10)
2. FOLLOW-UPS: Generate 3-4 specific follow-up search queries that would find complementary information
3. KEY INSIGHT: What's the most important insight from these documents?
Think like a thorough researcher - what related topics, different perspectives, or specific details should be explored next?
Response format:
SCORES: [8, 3, 9, 5, ...]
QUERIES:
- specific follow-up query 1
- specific follow-up query 2
- specific follow-up query 3
INSIGHT: most important finding from this search"""
        try:
            response = self.chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=500
            )
            response_text = response.choices[0].message.content
            # Parse response
            scores = []
            follow_ups = []
            insight = ""
            # Extract scores
            if "SCORES:" in response_text:
                scores_line = response_text.split("SCORES:")[1].split("QUERIES:")[0].strip()
                nums = re.findall(r"\d+", scores_line)
                scores = [int(n) for n in nums] if nums else [5] * len(results)
            # Extract follow-up queries
            if "QUERIES:" in response_text:
                queries_section = response_text.split("QUERIES:")[1]
                if "INSIGHT:" in queries_section:
                    queries_section = queries_section.split("INSIGHT:")[0]
                follow_ups = [q.strip().lstrip('- ') for q in queries_section.split('\n') if q.strip() and q.strip().startswith('-')]
            # Extract insight
            if "INSIGHT:" in response_text:
                insight = response_text.split("INSIGHT:")[1].strip()
            # Apply scores
            for i, score in enumerate(scores[:len(results)]):
                results[i].relevance_score = float(score)
            return sorted(results, key=lambda x: x.relevance_score, reverse=True), follow_ups, insight
        except Exception as e:
            print(f"⚠️ Analysis error: {e}")
            # Fallback scoring
            for result in results:
                result.relevance_score = max(0, 10 - (result.distance * 10))
            return results, [], "Analysis failed"

    def recursive_research(self, original_query: str, max_iterations: int = 5, min_confidence: float = 7.0) -> ResearchReport:
        """Main recursive research engine with larger data handling."""
        print(f"🔄 RECURSIVE RESEARCH: '{original_query}'")
        print(f"📊 Max iterations: {max_iterations}, Target confidence: {min_confidence}/10")
        research_tree = []
        all_evidence = []
        processed_queries = set()
        query_queue = [original_query]
        iteration = 0
        while iteration < max_iterations and query_queue:
            iteration += 1
            current_queries = query_queue[:4]  # Process more queries per iteration
            query_queue = query_queue[4:]
            print(f"\n🔄 ITERATION {iteration}/{max_iterations}")
            print(f"📋 Processing: {', '.join([f'\"{q}\"' for q in current_queries])}")
            iteration_nodes = []
            for query in current_queries:
                if query.lower() in processed_queries:
                    continue
                processed_queries.add(query.lower())
                print(f"  🔍 '{query}'")
                # Execute search with more results for big datasets
                results = self.search_with_query(query, n_results=20, iteration=iteration)
                # Analyze and get follow-ups
                scored_results, follow_ups, key_insight = self.evaluate_and_generate_followups(
                    original_query, query, results
                )
                # Keep more results for big datasets
                quality_results = [r for r in scored_results if r.relevance_score >= 4.0]  # Lower threshold
                all_evidence.extend(quality_results)
                # Create research node
                node = ResearchNode(
                    query=query,
                    results=quality_results,
                    iteration=iteration,
                    findings=key_insight,
                    importance=max([r.relevance_score for r in quality_results], default=0)
                )
                # Track query relationships
                if query != original_query:
                    node.parent_queries = self.query_genealogy.get(query, [])
                node.children_queries = follow_ups
                for child_q in follow_ups:
                    self.query_genealogy[child_q] = self.query_genealogy.get(child_q, []) + [query]
                iteration_nodes.append(node)
                print(f"    ✅ {len(quality_results)} relevant results (best: {node.importance:.1f}/10)")
                if key_insight:
                    print(f"    💡 {key_insight[:100]}...")
                print(f"    🎯 {len(follow_ups)} follow-ups generated")
                # Add promising follow-ups to queue
                promising_followups = [fq for fq in follow_ups if fq.lower() not in processed_queries]
                query_queue.extend(promising_followups[:3])  # More follow-ups for big datasets
            research_tree.extend(iteration_nodes)
            # Progress assessment
            total_quality = len([r for r in all_evidence if r.relevance_score >= min_confidence])
            avg_quality = sum(r.relevance_score for r in all_evidence) / len(all_evidence) if all_evidence else 0
            print(f"  📈 Progress: {total_quality} high-confidence, avg: {avg_quality:.1f}/10, {len(all_evidence)} total")
            print(f"  📋 Queue: {len(query_queue)} queries for next iteration")
            # Adjusted stopping criteria for big datasets
            if total_quality >= 20 and avg_quality >= min_confidence:
                print(f"  🎯 Excellence threshold reached - research complete")
                break
        print(f"\n🏁 Research complete: {iteration} iterations, {len(processed_queries)} searches, {len(all_evidence)} evidence pieces")
        # Deduplicate and keep more results for big datasets
        seen_docs = set()
        unique_evidence = []
        for result in all_evidence:
            doc_key = result.document[:100]
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                unique_evidence.append(result)
        final_evidence = sorted(unique_evidence, key=lambda x: (x.relevance_score, -x.iteration), reverse=True)
        # Generate synthesis with more evidence
        synthesis = self.generate_comprehensive_synthesis(original_query, research_tree, final_evidence[:30])
        # Extract critical findings
        critical_findings = self.extract_critical_findings(research_tree, final_evidence)
        report = ResearchReport(
            original_query=original_query,
            total_iterations=iteration,
            total_searches=len(processed_queries),
            research_tree=research_tree,
            final_synthesis=synthesis,
            critical_findings=critical_findings,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        # Store for follow-up questions
        self.last_research_report = report
        return report

    def generate_comprehensive_synthesis(self, original_query: str, research_tree: List[ResearchNode], evidence: List[SearchResult]) -> str:
        """Generate final research synthesis from all findings."""
        # Organize evidence by iteration
        evidence_by_iteration = {}
        for result in evidence:
            iter_num = result.iteration
            if iter_num not in evidence_by_iteration:
                evidence_by_iteration[iter_num] = []
            evidence_by_iteration[iter_num].append(result)
        # Build research progression narrative
        research_progression = []
        for node in research_tree:
            if node.findings:
                research_progression.append(f"Iteration {node.iteration} - '{node.query}' → {node.findings}")
        # Compile evidence sections
        context_sections = []
        context_sections.append("RESEARCH PROGRESSION:\n" + '\n'.join(research_progression))
        for iter_num in sorted(evidence_by_iteration.keys()):
            iter_evidence = evidence_by_iteration[iter_num][:6]
            iter_content = []
            for i, result in enumerate(iter_evidence):
                filename = result.metadata.get('filename', 'unknown')
                score = result.relevance_score
                preview = result.document[:400] + "..." if len(result.document) > 400 else result.document
                iter_content.append(f"[Source: {filename} | Relevance: {score:.1f}/10]\n{preview}")
            context_sections.append(f"ITERATION {iter_num} FINDINGS:\n" + '\n---\n'.join(iter_content))
        full_context = '\n' + '='*60 + '\n'.join(context_sections)
        synthesis_prompt = f"""COMPREHENSIVE RESEARCH SYNTHESIS
Original Research Question: "{original_query}"
Research Method: Recursive investigation with {len(research_tree)} query branches across {max([n.iteration for n in research_tree])} iterations
{full_context}
SYNTHESIS REQUIREMENTS:
1. Provide a thorough, well-organized answer that integrates ALL findings
2. Structure with clear sections and headings
3. Cite specific sources with relevance scores
4. Highlight key discoveries from the recursive research process
5. Note how findings evolved across iterations
6. Address any contradictions or information gaps
7. Explain the research journey and what each iteration revealed
8. Make this a comprehensive research report
Generate the complete synthesis:"""
        try:
            response = self.chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.1,
                max_tokens=2500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Synthesis generation failed: {e}"

    def extract_critical_findings(self, research_tree: List[ResearchNode], evidence: List[SearchResult]) -> List[str]:
        """Extract the most important discoveries."""
        critical_findings = []
        # From high-importance research nodes
        for node in research_tree:
            if node.importance >= 8.0 and node.findings:
                critical_findings.append(f"[Iteration {node.iteration}] {node.findings}")
        # From highest-scoring evidence
        top_evidence = [r for r in evidence if r.relevance_score >= 8.5][:8]
        for result in top_evidence:
            filename = result.metadata.get('filename', 'unknown')
            preview = result.document[:250] + "..." if len(result.document) > 250 else result.document
            critical_findings.append(f"[{filename}] {preview}")
        return critical_findings[:12]

    def generate_investigation_flowchart(self, report: ResearchReport) -> str:
        """Generate Maltego-style investigation flowchart."""
        flowchart_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Investigation: {report.original_query}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
            overflow-x: auto;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            text-align: center;
            backdrop-filter: blur(10px);
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #FFD700, #FF6B6B);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(5px);
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
        }}
        .flow-diagram {{
            display: flex;
            flex-direction: column;
            gap: 40px;
            margin: 30px 0;
        }}
        .iteration-layer {{
            position: relative;
            background: rgba(0,0,0,0.2);
            border-radius: 15px;
            padding: 25px;
            border-left: 6px solid;
        }}
        .iter-1 {{ border-left-color: #FF6B6B; }}
        .iter-2 {{ border-left-color: #4ECDC4; }}
        .iter-3 {{ border-left-color: #45B7D1; }}
        .iter-4 {{ border-left-color: #96CEB4; }}
        .iter-5 {{ border-left-color: #FECA57; }}
        .iteration-header {{
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }}
        .iteration-badge {{
            background: linear-gradient(45deg, #667eea, #764ba2);
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-right: 15px;
        }}
        .query-nodes {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .query-node {{
            background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
            position: relative;
            transition: all 0.3s ease;
        }}
        .query-node:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .query-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }}
        .query-text {{
            font-weight: bold;
            font-size: 1.1em;
            flex: 1;
            margin-right: 10px;
        }}
        .importance-badge {{
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .imp-high {{ background: #4CAF50; }}
        .imp-medium {{ background: #FF9800; }}
        .imp-low {{ background: #757575; }}
        .insight-box {{
            background: rgba(255,215,0,0.15);
            border-left: 4px solid #FFD700;
            padding: 12px;
            margin: 10px 0;
            border-radius: 5px;
            font-style: italic;
        }}
        .evidence-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 12px;
            margin-top: 15px;
        }}
        .evidence-card {{
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .evidence-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .source-tag {{
            font-size: 0.85em;
            color: #B0BEC5;
        }}
        .relevance-score {{
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .score-excellent {{ background: #4CAF50; }}
        .score-good {{ background: #8BC34A; }}
        .score-fair {{ background: #FF9800; }}
        .score-poor {{ background: #757575; }}
        .evidence-preview {{
            font-size: 0.9em;
            line-height: 1.4;
            max-height: 80px;
            overflow: hidden;
        }}
        .follow-ups {{
            margin-top: 15px;
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }}
        .follow-up-list {{
            list-style: none;
            margin-top: 8px;
        }}
        .follow-up-list li {{
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .flow-arrow {{
            text-align: center;
            font-size: 2.5em;
            margin: 20px 0;
            animation: bounce 2s infinite;
        }}
        @keyframes bounce {{
            0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
            40% {{ transform: translateY(-10px); }}
            60% {{ transform: translateY(-5px); }}
        }}
        .final-synthesis {{
            background: linear-gradient(145deg, rgba(0,0,0,0.4), rgba(0,0,0,0.2));
            border-radius: 15px;
            padding: 30px;
            margin-top: 40px;
            border: 2px solid #4CAF50;
        }}
        .synthesis-content {{
            background: rgba(255,255,255,0.05);
            padding: 25px;
            border-radius: 10px;
            white-space: pre-wrap;
            line-height: 1.6;
            font-size: 1.05em;
        }}
        .critical-findings {{
            background: rgba(255,0,0,0.1);
            border: 2px solid #FF6B6B;
            border-radius: 15px;
            padding: 25px;
            margin: 30px 0;
        }}
        .critical-list {{
            list-style: none;
            margin-top: 15px;
        }}
        .critical-list li {{
            padding: 10px;
            margin: 8px 0;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            border-left: 4px solid #FF6B6B;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🕵️ RECURSIVE RESEARCH INVESTIGATION</h1>
            <h2>"{report.original_query}"</h2>
            <p>📅 Investigation completed: {report.timestamp}</p>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-number">{report.total_iterations}</div>
                    <div>Iterations</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{report.total_searches}</div>
                    <div>Total Searches</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{len(report.research_tree)}</div>
                    <div>Query Branches</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{len([r for node in report.research_tree for r in node.results])}</div>
                    <div>Evidence Pieces</div>
                </div>
            </div>
        </div>"""
        # Add critical findings
        if report.critical_findings:
            flowchart_html += f"""
        <div class="critical-findings">
            <h3>🚨 CRITICAL DISCOVERIES</h3>
            <ul class="critical-list">"""
            for finding in report.critical_findings[:8]:
                flowchart_html += f"<li>{finding}</li>"
            flowchart_html += "</ul></div>"
        # Research flow diagram
        flowchart_html += '<div class="flow-diagram">'
        # Group by iteration
        iterations = {}
        for node in report.research_tree:
            if node.iteration not in iterations:
                iterations[node.iteration] = []
            iterations[node.iteration].append(node)
        for iter_num in sorted(iterations.keys()):
            nodes = iterations[iter_num]
            flowchart_html += f'''
        <div class="iteration-layer iter-{iter_num}">
            <div class="iteration-header">
                <div class="iteration-badge">Iteration {iter_num}</div>
                <div>{len(nodes)} research branches</div>
            </div>
            <div class="query-nodes">'''
            for node in nodes:
                # Determine importance class
                imp_class = "imp-high" if node.importance >= 7 else "imp-medium" if node.importance >= 5 else "imp-low"
                flowchart_html += f'''
                <div class="query-node">
                    <div class="query-header">
                        <div class="query-text">🔍 "{node.query}"</div>
                        <div class="importance-badge {imp_class}">{node.importance:.1f}/10</div>
                    </div>
                    {f'<div class="insight-box">💡 <strong>Key Insight:</strong> {node.findings}</div>' if node.findings else ''}
                    <div class="evidence-grid">'''
                for result in node.results[:8]:
                    filename = result.metadata.get('filename', 'unknown')
                    score = result.relevance_score
                    score_class = "score-excellent" if score >= 8 else "score-good" if score >= 6 else "score-fair" if score >= 4 else "score-poor"
                    preview = result.document[:200] + "..." if len(result.document) > 200 else result.document
                    flowchart_html += f'''
                        <div class="evidence-card">
                            <div class="evidence-header">
                                <span class="source-tag">📄 {filename}</span>
                                <span class="relevance-score {score_class}">{score:.1f}</span>
                            </div>
                            <div class="evidence-preview">{preview}</div>
                        </div>'''
                flowchart_html += '</div>'  # Close evidence-grid
                # Show follow-up queries
                if node.children_queries:
                    flowchart_html += f'''
                    <div class="follow-ups">
                        <strong>🎯 Generated Follow-ups:</strong>
                        <ul class="follow-up-list">'''
                    for child_q in node.children_queries[:4]:
                        flowchart_html += f'<li>"{child_q}"</li>'
                    flowchart_html += '''</ul>
                    </div>'''
                flowchart_html += '</div>'  # Close query-node
            flowchart_html += '</div>'  # Close query-nodes
            # Add flow arrow between iterations
            if iter_num < max(iterations.keys()):
                flowchart_html += '<div class="flow-arrow">⬇️</div>'
            flowchart_html += '</div>'  # Close iteration-layer
        # Final synthesis section
        flowchart_html += f'''
        </div>
        <div class="final-synthesis">
            <h2>📋 COMPREHENSIVE RESEARCH SYNTHESIS</h2>
            <div class="synthesis-content">{report.final_synthesis}</div>
        </div>
        <div style="text-align: center; margin-top: 30px; opacity: 0.7;">
            <p>🔬 Investigation completed with recursive research methodology</p>
            <p>Generated by Recursive RAG Agent | {report.timestamp}</p>
        </div>
    </div>
</body>
</html>'''
        return flowchart_html

    def save_and_open_flowchart(self, report: ResearchReport) -> str:
        """Save investigation flowchart and open in browser."""
        try:
            flowchart_html = self.generate_investigation_flowchart(report)
            # Create filename based on query
            safe_query = re.sub(r'[^\w\s-]', '', report.original_query).strip()
            safe_query = re.sub(r'[-\s]+', '-', safe_query)[:30]
            # Save to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(flowchart_html)
                temp_path = f.name
            # Try to open in browser
            webbrowser.open(f'file://{temp_path}')
            return f"📊 Investigation flowchart opened in browser: {temp_path}"
        except Exception as e:
            return f"⚠️ Flowchart generation failed: {e}"

    def handle_followup_question(self, question: str) -> str:
        """Handle follow-up questions about previous research."""
        if not self.last_research_report:
            return "No previous research to follow up on. Please ask a research question first."
        print(f"🔗 Follow-up question about: '{self.last_research_report.original_query}'")
        # Check if they want additional search
        needs_search = "/search" in question.lower()
        clean_question = question.replace("/search", "").strip()
        if needs_search:
            print("🔍 Additional search requested - connecting more dots...")
            # Generate search queries based on the follow-up
            search_prompt = f"""Previous research topic: "{self.last_research_report.original_query}"
Previous findings summary: {self.last_research_report.critical_findings[:3]}
Follow-up question: "{clean_question}"
Generate 4-5 specific search queries that would help answer this follow-up question by finding additional relevant information or connections.
Return only the search queries, one per line:"""
            try:
                response = self.chat_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": search_prompt}],
                    temperature=0.3,
                    max_tokens=200
                )
                additional_queries = [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()]
                # Execute additional searches
                new_evidence = []
                for aq in additional_queries:
                    print(f"  🔍 '{aq}'")
                    results = self.search_with_query(aq, n_results=12)
                    scored_results, _, _ = self.evaluate_and_generate_followups(
                        self.last_research_report.original_query, aq, results
                    )
                    quality_results = [r for r in scored_results if r.relevance_score >= 5.0]
                    new_evidence.extend(quality_results)
                print(f"  ✅ Found {len(new_evidence)} additional evidence pieces")
                # Combine with previous evidence
                all_evidence = []
                for node in self.last_research_report.research_tree:
                    all_evidence.extend(node.results)
                all_evidence.extend(new_evidence)
            except Exception as e:
                print(f"⚠️ Additional search failed: {e}")
                # Use existing evidence
                all_evidence = []
                for node in self.last_research_report.research_tree:
                    all_evidence.extend(node.results)
        else:
            # Use existing research evidence
            all_evidence = []
            for node in self.last_research_report.research_tree:
                all_evidence.extend(node.results)
        # Deduplicate and sort
        seen = set()
        unique_evidence = []
        for result in all_evidence:
            key = result.document[:100]
            if key not in seen:
                seen.add(key)
                unique_evidence.append(result)
        top_evidence = sorted(unique_evidence, key=lambda x: x.relevance_score, reverse=True)[:25]
        # Generate follow-up answer
        previous_context = f"PREVIOUS RESEARCH: {self.last_research_report.original_query}\n"
        previous_context += f"PREVIOUS FINDINGS: {self.last_research_report.final_synthesis[:1000]}...\n"
        evidence_context = ""
        for i, result in enumerate(top_evidence):
            filename = result.metadata.get('filename', 'unknown')
            score = result.relevance_score
            evidence_context += f"[Source {i+1}: {filename} | Score: {score:.1f}]\n{result.document}\n---\n"
        followup_prompt = f"""{previous_context}RELEVANT EVIDENCE:
{evidence_context}
FOLLOW-UP QUESTION: "{clean_question}"
Based on the previous research and the relevant evidence, provide a detailed answer to this follow-up question. Reference specific sources and connect the dots between the previous findings and this new question.
{'Note: Additional search was performed to gather more relevant information.' if needs_search else ''}
Answer:"""
        try:
            response = self.chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": followup_prompt}],
                temperature=0.15,
                max_tokens=1500
            )
            answer = response.choices[0].message.content
            # Store this interaction
            self.conversation_memory.append({
                "question": clean_question,
                "answer": answer,
                "used_search": needs_search,
                "evidence_count": len(top_evidence)
            })
            search_note = f"\n🔍 Connected {len(new_evidence) if needs_search else len(top_evidence)} evidence pieces" + (f" (+{len(new_evidence)} new)" if needs_search else "")
            return answer + search_note
        except Exception as e:
            return f"Follow-up analysis failed: {e}"

    def classify_query_intent(self, query: str) -> str:
        """Enhanced classification that detects follow-ups."""
        query_lower = query.lower()
        # Follow-up question indicators
        followup_keywords = [
            'what about', 'how about', 'what if', 'can you', 'tell me more',
            'expand on', 'explain', 'clarify', 'elaborate', 'details',
            'why', 'how', 'when', 'where', 'who', 'which',
            'that', 'it', 'they', 'those', 'this finding',
            'the research', 'your analysis', 'previous'
        ]
        # Check if this might be a follow-up
        if self.last_research_report and any(kw in query_lower for kw in followup_keywords):
            return "followup_question"
        # Original classification logic
        analysis_keywords = [
            'analyze', 'review', 'check', 'examine', 'audit',
            'shady', 'suspicious', 'wrong', 'issues', 'problems',
            'what do you think', 'your opinion', 'assessment'
        ]
        context_keywords = [
            'this', 'these', 'the document', 'the file', 'above',
            'what i', 'what we', 'my document', 'this content'
        ]
        recursive_keywords = [
            'deep dive', 'thorough', 'comprehensive', 'investigate',
            'research thoroughly', 'dig deep', 'full investigation',
            'recursive', 'exhaustive'
        ]
        if any(kw in query_lower for kw in context_keywords) and any(kw in query_lower for kw in analysis_keywords):
            return "document_analysis"
        elif any(kw in query_lower for kw in recursive_keywords):
            return "recursive_research"  
        else:
            return "information_search"

    def analyze_documents_for_issues(self, query: str) -> str:
        """Analyze all documents for issues/problems."""
        print("🔍 Document Analysis Mode")
        all_docs = self.get_all_documents()
        if not all_docs:
            return "No documents available to analyze."
        # Group by filename
        files_content = {}
        for result in all_docs:
            filename = result.metadata.get('filename', 'unknown')
            if filename not in files_content:
                files_content[filename] = []
            files_content[filename].append(result.document)
        print(f"📄 Analyzing {len(files_content)} files for: '{query}'")
        file_analyses = []
        for filename, chunks in files_content.items():
            print(f"  📖 {filename}")
            # Reconstruct full content
            full_content = '\n'.join(chunks)
            # General document analysis prompt
            analysis_prompt = f"""DOCUMENT ANALYSIS REQUEST
File: {filename}
Analysis Query: "{query}"
Document Content:
{full_content}
Please analyze this document in the context of the user's question. Look for:
- Anything that matches what the user is asking about
- Potential issues, problems, or concerns
- Notable patterns or anomalies  
- Important information relevant to their query
- Overall assessment of the document
Provide a thorough analysis addressing their specific question. Be detailed and specific about what you find."""
            try:
                response = self.chat_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.1,
                    max_tokens=1000
                )
                analysis = response.choices[0].message.content
                file_analyses.append(f"## 📄 Analysis: {filename}\n{analysis}")
            except Exception as e:
                file_analyses.append(f"## 📄 {filename}\n⚠️ Analysis error: {e}")
        # Compile final report
        report = f"""# 📋 DOCUMENT ANALYSIS REPORT
**Query:** "{query}"
**Files Analyzed:** {len(files_content)}
**Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
## 📖 INDIVIDUAL FILE ANALYSES
{chr(10).join(file_analyses)}
## 🎯 OVERALL ASSESSMENT"""
        # Generate overall assessment
        summary_prompt = f"""Based on this comprehensive document analysis:
{report}
Provide an overall assessment that:
1. Summarizes the main findings across all documents
2. Addresses the original question: "{query}"
3. Highlights the most important discoveries
4. Provides conclusions and recommendations
Be thorough and insightful."""
        try:
            response = self.chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.1,
                max_tokens=800
            )
            assessment = response.choices[0].message.content
            return f"{report}\n{assessment}"
        except Exception as e:
            return f"{report}\n⚠️ Could not generate overall assessment: {e}"

    def standard_research(self, query: str) -> str:
        """Standard multi-query research mode with larger result sets."""
        print(f"🔬 Research: '{query}'")
        # Generate research queries
        query_gen_prompt = f"""Generate 6-8 different search queries to thoroughly research: "{query}"
Think like a researcher - consider different angles, related concepts, synonyms, specific aspects, broader/narrower scopes, technical terms, and alternative phrasings.
Return only the search queries, one per line:"""
        try:
            response = self.chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": query_gen_prompt}],
                temperature=0.4,
                max_tokens=300
            )
            research_queries = [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()]
        except:
            research_queries = [query, f"what is {query}", f"information about {query}", f"{query} details"]
        print(f"📝 Researching from {len(research_queries)} angles")
        # Execute all searches with larger result sets
        all_evidence = []
        for rq in research_queries:
            print(f"  🔍 '{rq}'")
            results = self.search_with_query(rq, n_results=12)  # More results per query
            all_evidence.extend(results)
        if not all_evidence:
            return "No relevant information found in the document collection."
        # Deduplicate and rank - keep more results
        seen = set()
        unique = []
        for result in all_evidence:
            key = result.document[:100]
            if key not in seen:
                seen.add(key)
                unique.append(result)
        unique.sort(key=lambda x: x.distance)
        top_results = unique[:15]  # More results for synthesis
        # Synthesize with larger context
        context_parts = []
        for i, result in enumerate(top_results, 1):
            filename = result.metadata.get('filename', 'unknown')
            context_parts.append(f"[Source {i}: {filename}]\n{result.document}")
        context = "\n---\n".join(context_parts)
        synthesis_prompt = f"""Research Question: "{query}"
Evidence from document collection ({len(top_results)} sources):
{context}
Based on this evidence, provide a comprehensive answer that:
1. Directly addresses the research question with depth and detail
2. Synthesizes information from multiple sources
3. Cites which sources support key points
4. Explores different aspects and perspectives found
5. Notes any patterns, contradictions, or gaps
6. Provides thorough coverage of the topic
Generate a detailed research-based response:"""
        try:
            response = self.chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.15,
                max_tokens=2000  # Longer responses
            )
            answer = response.choices[0].message.content
            # Store research context for follow-ups
            self.conversation_memory.append({
                "query": query,
                "answer": answer,
                "evidence_used": len(top_results)
            })
            return answer + f"\n📊 Research: {len(research_queries)} queries, {len(top_results)} sources analyzed"
        except Exception as e:
            return f"Synthesis error: {e}"

    def smart_chat(self, query: str, mode: str = "auto", max_iterations: int = 3) -> str:
        """Enhanced chat router with follow-up support."""
        # Store conversation context
        self.conversation_memory.append({"user_query": query, "timestamp": datetime.now().isoformat()})
        if mode == "recursive":
            print(f"🔄 RECURSIVE MODE: {max_iterations} iterations max")
            report = self.recursive_research(query, max_iterations)
            flowchart_msg = self.save_and_open_flowchart(report)
            return f"{report.final_synthesis}\n{flowchart_msg}\n📊 Deep Research: {report.total_iterations} iterations, {report.total_searches} searches"
        elif mode == "auto":
            intent = self.classify_query_intent(query)
            if intent == "followup_question":
                return self.handle_followup_question(query)
            elif intent == "document_analysis":
                return self.analyze_documents_for_issues(query)
            elif intent == "recursive_research":
                return self.smart_chat(query, mode="recursive", max_iterations=5)
            else:
                return self.standard_research(query)
        elif mode == "simple":
            return self.simple_search(query)
        else:  # mode == "research"
            return self.standard_research(query)

    def simple_search(self, query: str) -> str:
        """Basic single-search mode."""
        results = self.search_with_query(query, n_results=5)
        if not results:
            return "No relevant documents found."
        context = "\n".join([r.document for r in results[:3]])
        messages = [
            {"role": "system", "content": "Answer based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\nQuestion: {query}"}
        ]
        try:
            response = self.chat_client.chat.completions.create(
                model=CHAT_MODEL, messages=messages, temperature=0.2, max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def get_all_documents(self) -> List[SearchResult]:
        """Retrieve all documents from all shards."""
        all_results = []
        for i, collection in enumerate(self.shards):
            try:
                all_data = collection.get(include=["documents", "metadatas"])
                for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
                    all_results.append(SearchResult(
                        document=doc,
                        metadata=meta or {},
                        distance=0.0,
                        relevance_score=10.0
                    ))
            except Exception as e:
                print(f"⚠️ Error retrieving documents from shard_{i:02d}: {e}")
                continue
        return all_results

    def ingest_txt_files(self, folder="./txt", processed="./processed"):
        """Ingest text files into the collection with sharding."""
        os.makedirs(processed, exist_ok=True)
        txt_files = glob.glob(os.path.join(folder, "*.txt"))
        print(f"📥 Found {len(txt_files)} files to ingest...")
        
        # Collect all chunks first
        all_chunks, all_ids, all_metas = [], [], []
        
        for filepath in tqdm(txt_files, desc="Processing", unit="file"):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if not text.strip():
                    continue
                chunks = self.smart_chunk_text(text)
                filename = os.path.basename(filepath)
                for i, chunk in enumerate(chunks):
                    uid = str(uuid.uuid4())  # unique ID
                    metadata = {
                        "filename": filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "char_count": len(chunk),
                        "file_path": filepath
                    }
                    all_chunks.append(chunk)
                    all_ids.append(uid)
                    all_metas.append(metadata)
            except Exception as e:
                print(f"⚠️ Error: {e}")
                continue
        
        # Distribute to shards
        shard_batches = {i: {"texts": [], "ids": [], "metas": []} for i in range(self.num_shards)}
        
        for chunk, uid, meta in zip(all_chunks, all_ids, all_metas):
            shard_idx = self._get_shard_for_id(uid)
            shard_batches[shard_idx]["texts"].append(chunk)
            shard_batches[shard_idx]["ids"].append(uid)
            shard_batches[shard_idx]["metas"].append(meta)
        
        # Process each shard
        for shard_idx, batch in tqdm(shard_batches.items(), desc="Upserting to shards"):
            texts = batch["texts"]
            ids = batch["ids"]
            metas = batch["metas"]
            
            if not texts:
                continue
                
            for i in range(0, len(texts), MAX_BATCH_SIZE):
                batch_texts = texts[i:i+MAX_BATCH_SIZE]
                batch_ids = ids[i:i+MAX_BATCH_SIZE]
                batch_metas = metas[i:i+MAX_BATCH_SIZE]
                
                try:
                    resp = self.embed_client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
                    embs = [d.embedding for d in resp.data]
                    self.shards[shard_idx].upsert(
                        documents=batch_texts,
                        embeddings=embs,
                        ids=batch_ids,
                        metadatas=batch_metas
                    )
                except Exception as e:
                    print(f"⚠️ Embedding batch failed for shard_{shard_idx:02d} at {i}: {e}")
        
        # Move processed files
        for f in txt_files:
            dest = os.path.join(processed, os.path.basename(f))
            try:
                shutil.move(f, dest)
            except:
                pass
        
        print(f"✅ Processed {len(txt_files)} files")

    def quick_search(self, query: str, n_results: int = 5):
        """Debug search to see raw results across all shards."""
        try:
            q_emb = self.embed_client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
            
            print(f"\n🔍 Debug Search: '{query}'")
            print("="*50)
            
            # Query all shards
            all_results = []
            for i, collection in enumerate(self.shards):
                try:
                    results = collection.query(
                        query_embeddings=[q_emb], 
                        n_results=n_results, 
                        include=["documents", "metadatas", "distances"]
                    )
                    if results["documents"] and results["documents"][0]:
                        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                            all_results.append((doc, meta, dist, i))  # Include shard index
                except Exception as e:
                    print(f"⚠️ Search error in shard_{i:02d}: {e}")
                    continue
            
            # Sort by distance and show top results
            all_results.sort(key=lambda x: x[2])  # Sort by distance
            top_results = all_results[:n_results]
            
            if top_results:
                for i, (doc, meta, dist, shard_idx) in enumerate(top_results):
                    filename = meta.get("filename", "unknown") if meta else "unknown"
                    chunk_idx = meta.get("chunk_index", 0) if meta else 0
                    print(f"\nResult {i+1}: {filename} (chunk {chunk_idx+1}) [Shard: {shard_idx}]")
                    print(f"Distance: {dist:.3f}")
                    print(f"Preview: {doc[:200]}...")
                    print("-" * 30)
            else:
                print("❌ No results found.")
                
        except Exception as e:
            print(f"⚠️ Search error: {e}")

    def get_stats(self):
        """Display collection statistics across all shards."""
        try:
            total_chunks = 0
            all_files = set()
            total_chars = 0
            
            for i, collection in enumerate(self.shards):
                try:
                    all_data = collection.get(include=["metadatas"])
                    count = len(all_data["ids"])
                    total_chunks += count
                    print(f"📊 Shard_{i:02d}: {count:,} chunks")
                    
                    for meta in all_data["metadatas"]:
                        if meta:
                            if "filename" in meta:
                                all_files.add(meta["filename"])
                            if "char_count" in meta:
                                total_chars += meta.get("char_count", 0)
                except Exception as e:
                    print(f"⚠️ Stats error for shard_{i:02d}: {e}")
                    continue
            
            print(f"\n📈 TOTAL: {total_chunks:,} chunks across {len(self.shards)} shards")
            print(f"📁 Unique Files: {len(all_files)}")
            print(f"📝 Total Characters: {total_chars:,}")
            if total_chunks > 0:
                print(f"📊 Avg Chars/Chunk: {total_chars // total_chunks}")
                
        except Exception as e:
            print(f"⚠️ Stats error: {e}")

# Global agent instance
agent = RecursiveRAGAgent(num_shards=10)  # Default 10 shards

def ingest_txt_files(folder="./txt", processed="./processed"):
    """Convenience function for ingestion."""
    return agent.ingest_txt_files(folder, processed)

def rag_chat(query: str, mode: str = "auto", max_iterations: int = 3) -> str:
    """Main chat interface."""
    return agent.smart_chat(query, mode, max_iterations)

if __name__ == "__main__":
    print("🧠 RECURSIVE RAG RESEARCH AGENT")
    print("=" * 60)
    print("📚 General-purpose document research system")
    print("🔄 Works with ANY text documents")
    # Ingest files
    try:
        agent.ingest_txt_files("./txt")
        agent.get_stats()
    except Exception as e:
        print(f"❌ Ingestion error: {e}")
    print("\n🤖 RESEARCH MODES:")
    print("  🔄 recursive: <query>     - Deep recursive research + visual flowchart")
    print("  🔍 research: <query>      - Multi-angle research")  
    print("  ⚡ simple: <query>        - Single search")
    print("  🔍 search: <query>        - Debug search results")
    print("  📊 stats                  - Collection statistics")
    print("  ❌ quit                   - Exit")
    print("=" * 60)
    print("💡 Auto-detects document analysis vs. information research")
    print("💡 Ask follow-up questions naturally after any research")
    print("💡 Use '/search' in follow-ups to find additional connections")
    print("💡 Optimized for large datasets with bigger chunks & more results")
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            if not user_input or user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower().startswith('recursive:'):
                query = user_input[10:].strip()
                iterations = 5  # Default for recursive
                print(f"🔄 RECURSIVE MODE: {iterations} max iterations")
                answer = rag_chat(query, mode="recursive", max_iterations=iterations)
                print(f"\n🤖 Researcher: {answer}")
            elif user_input.lower().startswith('research:'):
                query = user_input[9:].strip()
                print("🔬 Multi-angle research mode")
                answer = rag_chat(query, mode="research")
                print(f"\n🤖 Researcher: {answer}")
            elif user_input.lower().startswith('simple:'):
                query = user_input[7:].strip()
                print("⚡ Simple search mode")
                answer = rag_chat(query, mode="simple")
                print(f"\n🤖 Agent: {answer}")
            elif user_input.lower().startswith('search:'):
                query = user_input[7:].strip()
                agent.quick_search(query)
            elif user_input.lower() == 'stats':
                agent.get_stats()
            else:
                # Auto-detect mode based on query
                print("🧠 Auto-detecting research approach...")
                answer = rag_chat(user_input, mode="auto")
                print(f"\n🤖 Agent: {answer}")
        except KeyboardInterrupt:
            print("\n👋 Research session ended.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
    print("📋 Session complete. Goodbye!")