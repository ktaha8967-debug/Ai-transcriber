import re
import numpy as np

def split_into_sentences(text):
    # Splits on English (.), Urdu (۔), Hindi (|), and common sentence endings
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|۔|\|)\s*')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def clean_text(text):
    # Basic cleaning: remove punctuation, lowercase
    text = re.sub(r'[^\w\s\u0600-\u06FF\u0900-\u097F]', '', text) # Keep Arabic/Urdu and Devanagari/Hindi chars
    return text.lower()

def sentence_similarity(sent1, sent2):
    words1 = set(clean_text(sent1).split())
    words2 = set(clean_text(sent2).split())
    
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    # Cosine-like overlap
    return len(intersection) / (np.log(len(words1) + 1) + np.log(len(words2) + 1) + 1e-5)

def summarize_text(text, num_sentences=3):
    sentences = split_into_sentences(text)
    if len(sentences) <= num_sentences:
        return sentences, sentences # Return everything if short
        
    # Build similarity matrix
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = sentence_similarity(sentences[i], sentences[j])
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim
            
    # PageRank power iteration
    damping = 0.85
    scores = np.ones(n) / n
    for _ in range(30): # 30 iterations is usually enough
        new_scores = np.zeros(n)
        for i in range(n):
            sum_links = 0
            for j in range(n):
                if similarity_matrix[j][i] > 0:
                    # Sum of (score(j) * similarity(j, i) / sum of similarity of j to all other nodes)
                    total_out = np.sum(similarity_matrix[j])
                    if total_out > 0:
                        sum_links += scores[j] * similarity_matrix[j][i] / total_out
            new_scores[i] = (1 - damping) / n + damping * sum_links
        scores = new_scores

    # Rank and select top sentences
    ranked_indices = np.argsort(scores)[::-1]
    top_indices = sorted(ranked_indices[:num_sentences])
    
    summary = [sentences[idx] for idx in top_indices]
    
    # Bullet points could be the top scoring sentences (sorted by score)
    bullet_indices = ranked_indices[:min(num_sentences + 2, len(sentences))]
    bullet_points = [sentences[idx] for idx in bullet_indices]
    
    return summary, bullet_points
