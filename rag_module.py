
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def load_rag_data(csv_path='loan_approval_dataset.csv'):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['loan_status'] = df['loan_status'].str.strip()
    return df


def create_statistical_documents(df):
    approved = df[df['loan_status'] == 'Approved']
    rejected = df[df['loan_status'] == 'Rejected']

    stat_documents = [
        f"The average CIBIL score of approved applicants is {approved['cibil_score'].mean():.0f}, compared to {rejected['cibil_score'].mean():.0f} for rejected applicants.",
        f"Approved applicants have an average annual income of {approved['income_annum'].mean():.0f}, while rejected applicants average {rejected['income_annum'].mean():.0f}.",
        f"High loan amounts are more likely to be rejected.",
        f"Low credit score increases rejection probability."
    ]

    return stat_documents


def create_case_documents(df):
    def row_to_text(row):
        return (
            f"Applicant with CIBIL {row['cibil_score']}, income {row['income_annum']}, "
            f"loan {row['loan_amount']}, term {row['loan_term']} months, status {row['loan_status']}."
        )

    approved = df[df['loan_status'] == 'Approved']
    rejected = df[df['loan_status'] == 'Rejected']

    approved_sample = approved.sample(min(250, len(approved)), random_state=42)
    rejected_sample = rejected.sample(min(250, len(rejected)), random_state=42)

    sampled_df = pd.concat([approved_sample, rejected_sample]).reset_index(drop=True)

    case_documents = [row_to_text(row) for _, row in sampled_df.iterrows()]
    return case_documents


def build_knowledge_base(stat_documents, case_documents):
    return stat_documents + case_documents


def build_embeddings(knowledge_base):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(knowledge_base).astype('float32')
    return model, embeddings


def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_evidence(query, rag_model, index, knowledge_base, top_k=3):
    query_vec = rag_model.encode([query]).astype('float32')
    distances, indices = index.search(query_vec, top_k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        results.append({
            "rank": rank,
            "document": knowledge_base[idx],
            "distance": float(dist)
        })

    return results


def applicant_to_query(row):
    return (
        f"Applicant with CIBIL score {row.get('cibil_score')}, "
        f"income {row.get('income_annum')}, "
        f"loan amount {row.get('loan_amount')}, "
        f"loan term {row.get('loan_term')} months."
    )
