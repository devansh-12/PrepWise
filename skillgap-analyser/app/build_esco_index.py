# import os
# import json
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# # === Paths ===
# csv_path = "data/esco/skills_en.csv"          # adjust if filename differs
# index_path = "vector_store/esco_faiss.index"
# meta_path = "vector_store/esco_meta.json"

# os.makedirs("vector_store", exist_ok=True)

# # === Load ESCO CSV ===
# print("📥 Loading ESCO skills...")
# df = pd.read_csv(csv_path, dtype=str).fillna("")

# # Some ESCO datasets call columns differently, try to normalize
# label_col = "preferredLabel" if "preferredLabel" in df.columns else "preferred_label"
# desc_col = "description" if "description" in df.columns else None
# alt_col  = "altLabels" if "altLabels" in df.columns else "alt_labels"
# uri_col  = "uri" if "uri" in df.columns else "id"

# print(f"✅ Found columns: {list(df.columns)}")

# # === Embedding model ===
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# vectors = []
# meta = {}

# print("⚙️ Encoding ESCO skills...")
# for idx, row in df.iterrows():
#     text_parts = [row.get(label_col, "")]
#     if desc_col: text_parts.append(row.get(desc_col, ""))
#     if alt_col:  text_parts.append(row.get(alt_col, ""))
#     text = " ".join([t for t in text_parts if isinstance(t, str)])

#     emb = model.encode(text, normalize_embeddings=True)
#     vectors.append(emb)

#     # 🔑 Store metadata with row index as key (string!)
#     meta[str(idx)] = {
#         "preferredLabel": row.get(label_col, ""),
#         "uri": row.get(uri_col, ""),
#         "altLabels": row.get(alt_col, "")
#     }

# # === Build FAISS index ===
# emb_matrix = np.vstack(vectors).astype("float32")
# index = faiss.IndexFlatIP(emb_matrix.shape[1])  # cosine similarity (dot product)
# index.add(emb_matrix)

# # === Save ===
# faiss.write_index(index, index_path)
# with open(meta_path, "w", encoding="utf-8") as f:
#     json.dump(meta, f, indent=2)

# print(f"\n✅ FAISS index saved to {index_path}")
# print(f"✅ Metadata saved to {meta_path}")
# print(f"📦 Indexed {len(vectors)} ESCO skills")




# import os
# import json
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# # === Paths ===
# skills_path = "data/esco/skills_en.csv"           # core skills
# hierarchy_path = "data/esco/skillsHierarchy_en.csv"   # maps skills -> groups
# groups_path = "data/esco/skillGroups_en.csv"          # group labels

# index_path = "vector_store/esco_faiss.index"
# meta_path = "vector_store/esco_meta.json"

# os.makedirs("vector_store", exist_ok=True)

# # === Load ESCO datasets ===
# print("📥 Loading ESCO datasets...")

# skills_df = pd.read_csv(skills_path, dtype=str).fillna("")
# hierarchy_df = pd.read_csv(hierarchy_path, dtype=str).fillna("")
# groups_df = pd.read_csv(groups_path, dtype=str).fillna("")

# # Normalize column names
# skills_label_col = "preferredLabel" if "preferredLabel" in skills_df.columns else "preferred_label"
# skills_desc_col = "description" if "description" in skills_df.columns else None
# skills_alt_col  = "altLabels" if "altLabels" in skills_df.columns else "alt_labels"
# skills_uri_col  = "uri" if "uri" in skills_df.columns else "id"

# # Build group lookup
# group_lookup = {row["id"]: row["preferredLabel"] for _, row in groups_df.iterrows() if "id" in row and "preferredLabel" in row}

# # Build mapping skill_id -> [group_names]
# skill_to_groups = {}
# for _, row in hierarchy_df.iterrows():
#     sid = row.get("skill_id") or row.get("skillId") or row.get("id")
#     gid = row.get("group_id") or row.get("groupId") or row.get("broaderSkillGroup")
#     if sid and gid and gid in group_lookup:
#         skill_to_groups.setdefault(sid, []).append(group_lookup[gid])

# print(f"✅ Skills: {len(skills_df)}, Groups: {len(groups_df)}, Hierarchy links: {len(hierarchy_df)}")

# # === Embedding model ===
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# vectors = []
# meta = {}

# print("⚙️ Encoding ESCO skills...")
# for idx, row in skills_df.iterrows():
#     skill_id = row.get(skills_uri_col, str(idx))  # unique id
#     text_parts = [row.get(skills_label_col, "")]
#     if skills_desc_col: text_parts.append(row.get(skills_desc_col, ""))
#     if skills_alt_col:  text_parts.append(row.get(skills_alt_col, ""))
#     text = " ".join([t for t in text_parts if isinstance(t, str)])

#     emb = model.encode(text, normalize_embeddings=True)
#     vectors.append(emb)

#     # Attach taxonomy categories if available
#     categories = skill_to_groups.get(skill_id, [])

#     meta[str(idx)] = {
#         "preferredLabel": row.get(skills_label_col, ""),
#         "uri": row.get(skills_uri_col, ""),
#         "altLabels": row.get(skills_alt_col, ""),
#         "categories": categories
#     }

# # === Build FAISS index ===
# emb_matrix = np.vstack(vectors).astype("float32")
# index = faiss.IndexFlatIP(emb_matrix.shape[1])  # cosine similarity
# index.add(emb_matrix)

# # === Save ===
# faiss.write_index(index, index_path)
# with open(meta_path, "w", encoding="utf-8") as f:
#     json.dump(meta, f, indent=2, ensure_ascii=False)

# print(f"\n✅ FAISS index saved to {index_path}")
# print(f"✅ Metadata saved to {meta_path}")
# print(f"📦 Indexed {len(vectors)} ESCO skills with taxonomy categories")


# 




# import os
# import json
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# def build_esco_index():
#     """
#     Processes ESCO CSV files to create a FAISS vector index and a corresponding
#     JSON metadata file for skill search and categorization.
#     """
#     # --- 1. Define File Paths ---
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(script_dir)

#     skills_path = os.path.join(project_root, "data","esco", "skills_en.csv")
#     hierarchy_path = os.path.join(project_root, "data","esco","skillsHierarchy_en.csv")
#     relations_path = os.path.join(project_root, "data","esco","broaderRelationsSkillPillar_en.csv")
#     vector_store_dir = os.path.join(project_root, "vector_store")
#     meta_path = os.path.join(vector_store_dir, "esco_meta.json")
#     index_path = os.path.join(vector_store_dir, "esco_faiss.index")

#     # --- 2. Load ESCO Datasets ---
#     os.makedirs(vector_store_dir, exist_ok=True)
#     print("📥 Loading ESCO datasets...")

#     try:
#         skills_df = pd.read_csv(skills_path, dtype=str).fillna("")
#         hierarchy_df = pd.read_csv(hierarchy_path, dtype=str).fillna("")
#         relations_df = pd.read_csv(relations_path, dtype=str).fillna("")
#     except FileNotFoundError as e:
#         print(f"❌ ERROR: Could not find data file. {e}")
#         print("Please ensure 'data/skills_en.csv', 'data/skillsHierarchy_en.csv', and 'data/broaderRelationsSkillPillar_en.csv' exist.")
#         return

#     # --- 3. Build Group URI to Category Mapping ---
#     print("⚙️ Building hierarchy mapping from group URIs to categories...")
#     group_uri_to_categories = {}
#     for _, row in hierarchy_df.iterrows():
#         cats = {
#             "level_0": row.get("Level 0 preferred term", "").strip(),
#             "level_1": row.get("Level 1 preferred term", "").strip(),
#             "level_2": row.get("Level 2 preferred term", "").strip(),
#             "level_3": row.get("Level 3 preferred term", "").strip()
#         }
#         cats = {k: v for k, v in cats.items() if v}

#         for level_uri in ["Level 0 URI", "Level 1 URI", "Level 2 URI", "Level 3 URI"]:
#             uri = row.get(level_uri, "").strip()
#             if uri:
#                 group_uri_to_categories[uri] = cats
#     print(f"✅ Built hierarchy mapping for {len(group_uri_to_categories)} group URIs.")

#     # --- 4. Build Skill URI to Parent Group URI Mapping ---
#     print("⚙️ Building mapping from skills to their parent groups...")
#     skill_to_group_map = {}
#     for _, row in relations_df.iterrows():
#         skill_uri = row.get("conceptUri", "").strip()
#         group_uri = row.get("broaderUri", "").strip()
#         if skill_uri and group_uri:
#             if skill_uri not in skill_to_group_map:
#                 skill_to_group_map[skill_uri] = group_uri
#     print(f"✅ Mapped {len(skill_to_group_map)} skills to a parent group.")

#     # --- 5. Encode Skills and Generate Metadata ---
#     print("⚙️ Loading CareerBERT model (this may take a moment)...")
#     model = SentenceTransformer("lwolfrum2/careerbert-jg")  # ✅ CareerBERT instead of MiniLM
    
#     vectors, meta = [], {}
#     total_skills = len(skills_df)
#     print(f"⚙️ Encoding {total_skills} ESCO skills and building metadata...")

#     for idx, row in skills_df.iterrows():
#         skill_uri = row.get("conceptUri", "").strip()
        
#         parent_group_uri = skill_to_group_map.get(skill_uri)
#         categories = group_uri_to_categories.get(parent_group_uri, {})

#         text_to_embed = " ".join([
#             row.get("preferredLabel", ""),
#             row.get("description", ""),
#             row.get("altLabels", "")
#         ])

#         embedding = model.encode(text_to_embed, normalize_embeddings=True)
#         vectors.append(embedding)

#         meta[str(idx)] = {
#             "preferredLabel": row.get("preferredLabel", "").strip(),
#             "uri": skill_uri,
#             "description": row.get("description", "").strip(),
#             "categories": categories
#         }
        
#         if (idx + 1) % 1000 == 0:
#             print(f"   ...processed {idx + 1} of {total_skills} skills.")

#     # --- 6. Build and Save FAISS Index ---
#     print("\n⚙️ Building and saving FAISS index...")
#     embedding_matrix = np.vstack(vectors).astype("float32")
#     index = faiss.IndexFlatIP(embedding_matrix.shape[1])
#     index.add(embedding_matrix)
#     faiss.write_index(index, index_path)
#     print(f"✅ FAISS index saved to {index_path}")

#     # --- 7. Save Metadata ---
#     print("⚙️ Saving metadata...")
#     with open(meta_path, "w", encoding="utf-8") as f:
#         json.dump(meta, f, indent=2, ensure_ascii=False)
#     print(f"✅ Metadata saved to {meta_path}")
#     print("\n🎉 Build process complete!")

# if __name__ == "__main__":
#     build_esco_index()

import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_esco_index():
    """
    Processes ESCO CSV files to create a FAISS vector index and a corresponding
    JSON metadata file for skill search and categorization.
    """
    # --- 1. Define File Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    skills_path = os.path.join(project_root, "data","esco", "skills_en.csv")
    hierarchy_path = os.path.join(project_root, "data","esco","skillsHierarchy_en.csv")
    relations_path = os.path.join(project_root, "data","esco","broaderRelationsSkillPillar_en.csv")
    vector_store_dir = os.path.join(project_root, "vector_store")
    meta_path = os.path.join(vector_store_dir, "esco_meta.json")
    index_path = os.path.join(vector_store_dir, "esco_faiss.index")

    # --- 2. Load ESCO Datasets ---
    os.makedirs(vector_store_dir, exist_ok=True)
    print("📥 Loading ESCO datasets...")

    try:
        skills_df = pd.read_csv(skills_path, dtype=str).fillna("")
        hierarchy_df = pd.read_csv(hierarchy_path, dtype=str).fillna("")
        relations_df = pd.read_csv(relations_path, dtype=str).fillna("")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find data file. {e}")
        print("Please ensure 'data/skills_en.csv', 'data/skillsHierarchy_en.csv', and 'data/broaderRelationsSkillPillar_en.csv' exist.")
        return

    # --- 3. Build Group URI to Category Mapping ---
    print("⚙️ Building hierarchy mapping from group URIs to categories...")
    group_uri_to_categories = {}
    for _, row in hierarchy_df.iterrows():
        cats = {
            "level_0": row.get("Level 0 preferred term", "").strip(),
            "level_1": row.get("Level 1 preferred term", "").strip(),
            "level_2": row.get("Level 2 preferred term", "").strip(),
            "level_3": row.get("Level 3 preferred term", "").strip()
        }
        cats = {k: v for k, v in cats.items() if v}

        for level_uri in ["Level 0 URI", "Level 1 URI", "Level 2 URI", "Level 3 URI"]:
            uri = row.get(level_uri, "").strip()
            if uri:
                group_uri_to_categories[uri] = cats
    print(f"✅ Built hierarchy mapping for {len(group_uri_to_categories)} group URIs.")

    # --- 4. Build Skill URI to Parent Group URI Mapping ---
    print("⚙️ Building mapping from skills to their parent groups...")
    skill_to_group_map = {}
    for _, row in relations_df.iterrows():
        skill_uri = row.get("conceptUri", "").strip()
        group_uri = row.get("broaderUri", "").strip()
        if skill_uri and group_uri:
            if skill_uri not in skill_to_group_map:
                skill_to_group_map[skill_uri] = group_uri
    print(f"✅ Mapped {len(skill_to_group_map)} skills to a parent group.")

    # --- 5. Encode Skills and Generate Metadata ---
    print("⚙️ Loading CareerBERT model (this may take a moment)...")
    model = SentenceTransformer("lwolfrum2/careerbert-jg")  # ✅ CareerBERT instead of MiniLM
    
    vectors, meta = [], {}
    total_skills = len(skills_df)
    print(f"⚙️ Encoding {total_skills} ESCO skills and building metadata...")

    for idx, row in skills_df.iterrows():
        skill_uri = row.get("conceptUri", "").strip()
        
        parent_group_uri = skill_to_group_map.get(skill_uri)
        categories = group_uri_to_categories.get(parent_group_uri, {})

        text_to_embed = " ".join([
            row.get("preferredLabel", ""),
            row.get("description", ""),
            row.get("altLabels", "")
        ])

        embedding = model.encode(text_to_embed, normalize_embeddings=True)
        vectors.append(embedding)

        meta[str(idx)] = {
            "preferredLabel": row.get("preferredLabel", "").strip(),
            "uri": skill_uri,
            "description": row.get("description", "").strip(),
            "altLabels": row.get("altLabels", "").strip(),   # ✅ Added here
            "categories": categories
        }
        
        if (idx + 1) % 1000 == 0:
            print(f"   ...processed {idx + 1} of {total_skills} skills.")

    # --- 6. Build and Save FAISS Index ---
    print("\n⚙️ Building and saving FAISS index...")
    embedding_matrix = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    faiss.write_index(index, index_path)
    print(f"✅ FAISS index saved to {index_path}")

    # --- 7. Save Metadata ---
    print("⚙️ Saving metadata...")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"✅ Metadata saved to {meta_path}")
    print("\n🎉 Build process complete!")

if __name__ == "__main__":
    build_esco_index()
