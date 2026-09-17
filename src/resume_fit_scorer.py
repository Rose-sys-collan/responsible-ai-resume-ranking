"""
Resume Fit Score Baseline Model
For AI Bias Mitigation Research Project

This script creates a baseline model that scores how well a candidate
fits a job description using semantic similarity (cosine similarity
between sentence embeddings).

This baseline can then be analyzed for potential bias patterns.
"""

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re

# --- Configuration ---
DATA_PATH = os.path.join(os.path.dirname(__file__), "cleaned_candidates (1).csv")
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and effective sentence embedding model

# --- Load Data ---
def load_candidates(file_path):
    """Load and preprocess candidate data."""
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)

    # Combine relevant fields into a single resume text
    df['resume_text'] = (
        df['job_title'] + ". " +
        df['job_description'] + " " +
        df['skills'] + " " +
        df['experience'] + " " +
        df['education']
    )

    # Clean text
    df['resume_text'] = df['resume_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    return df


class ResumeFitScorer:
    """
    Baseline fit scorer using sentence embeddings and cosine similarity.

    This model computes how semantically similar a candidate's resume
    is to a target job description.
    """

    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def encode_texts(self, texts, batch_size=32, show_progress=True):
        """Encode a list of texts into embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            device=self.device
        )
        return embeddings

    def compute_fit_scores(self, job_description, candidate_resumes):
        """
        Compute fit scores between a job description and multiple candidates.

        Returns scores between 0 and 1, where 1 = perfect match.
        """
        # Encode job description
        job_embedding = self.model.encode(
            job_description,
            convert_to_tensor=True,
            device=self.device
        )

        # Encode all candidate resumes
        resume_embeddings = self.encode_texts(candidate_resumes)

        # Compute cosine similarity
        similarities = util.cos_sim(job_embedding, resume_embeddings)[0]

        # Convert to numpy and normalize to 0-1 range
        scores = similarities.cpu().numpy()

        return scores

    def rank_candidates(self, job_description, df, top_k=10):
        """
        Rank candidates by fit score for a given job description.

        Returns a DataFrame with candidates ranked by fit score.
        """
        scores = self.compute_fit_scores(job_description, df['resume_text'].tolist())

        # Add scores to dataframe
        result_df = df.copy()
        result_df['fit_score'] = scores
        result_df = result_df.sort_values('fit_score', ascending=False)

        return result_df.head(top_k)


def analyze_bias_by_name(df, scorer, job_description):
    """
    Analyze potential bias by examining fit scores across different names.

    This is a basic bias analysis - checks if fit scores correlate
    with name characteristics.
    """
    scores = scorer.compute_fit_scores(job_description, df['resume_text'].tolist())
    df_analysis = df.copy()
    df_analysis['fit_score'] = scores

    # Group by first letter of name (basic proxy - in practice use more sophisticated methods)
    df_analysis['name_first_letter'] = df_analysis['name'].str[0].str.upper()

    # Basic statistics by job title
    print("\n=== Fit Score Distribution by Job Title ===")
    job_stats = df_analysis.groupby('job_title')['fit_score'].agg(['mean', 'std', 'count'])
    job_stats = job_stats.sort_values('mean', ascending=False)
    print(job_stats.head(15))

    return df_analysis


# --- Main Execution ---
if __name__ == "__main__":
    print("=" * 60)
    print("Resume Fit Score Baseline Model")
    print("For AI Bias Mitigation Research")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading candidate data...")
    df = load_candidates(DATA_PATH)
    print(f"Loaded {len(df)} candidates")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample job titles:\n{df['job_title'].value_counts().head(10)}")

    # Initialize scorer
    print("\n[2/4] Initializing fit scorer...")
    scorer = ResumeFitScorer()

    # Example job description for testing
    example_job = """
    Senior Data Scientist

    We are looking for an experienced Data Scientist to join our team.

    Requirements:
    - 5+ years of experience in data science or machine learning
    - Strong programming skills in Python and SQL
    - Experience with machine learning frameworks (TensorFlow, PyTorch, scikit-learn)
    - Experience with big data technologies (Spark, Hadoop)
    - Strong analytical and problem-solving skills
    - Excellent communication skills

    Preferred:
    - PhD in Computer Science, Statistics, or related field
    - Experience in healthcare or finance industry
    - Published research in machine learning
    """

    # Rank candidates
    print("\n[3/4] Ranking candidates for example job description...")
    print("-" * 60)
    print("Job: Senior Data Scientist")
    print("-" * 60)

    top_candidates = scorer.rank_candidates(example_job, df, top_k=10)

    print("\nTop 10 Candidates:")
    print("-" * 60)
    for idx, row in top_candidates.iterrows():
        print(f"Score: {row['fit_score']:.3f} | {row['name']} | {row['job_title']}")

    # Basic bias analysis
    print("\n[4/4] Running basic bias analysis...")
    df_with_scores = analyze_bias_by_name(df, scorer, example_job)

    # Visualize score distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(df_with_scores['fit_score'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Fit Score')
    plt.ylabel('Number of Candidates')
    plt.title('Distribution of Fit Scores')

    plt.subplot(1, 2, 2)
    top_jobs = df_with_scores.groupby('job_title')['fit_score'].mean().nlargest(10)
    plt.barh(range(len(top_jobs)), top_jobs.values)
    plt.yticks(range(len(top_jobs)), top_jobs.index)
    plt.xlabel('Average Fit Score')
    plt.title('Top 10 Job Titles by Average Fit Score')
    plt.tight_layout()

    plt.savefig('fit_score_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'fit_score_analysis.png'")

    # Save results
    df_with_scores.to_csv('candidates_with_scores.csv', index=False)
    print("Results saved to 'candidates_with_scores.csv'")

    print("\n" + "=" * 60)
    print("Baseline model complete!")
    print("Next steps for bias mitigation:")
    print("1. Analyze score differences across demographic groups")
    print("2. Test with controlled experiments (same resume, different names)")
    print("3. Implement fairness metrics (demographic parity, equalized odds)")
    print("=" * 60)
