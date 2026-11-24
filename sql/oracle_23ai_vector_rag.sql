-- Oracle Database 23ai example: store PR text and its 384-dimensional embedding.
-- Prerequisite: ALL_MINILM_L6_V2 ONNX embedding model is available in the schema.

CREATE TABLE pr_code_review_vectors (
    pr_id          VARCHAR2(100) PRIMARY KEY,
    pr_title       VARCHAR2(1000),
    code_diff      CLOB NOT NULL,
    pr_embedding   VECTOR(384, FLOAT32)
);

-- Populate the vector column using the database's imported ONNX model.
UPDATE pr_code_review_vectors
SET pr_embedding = VECTOR_EMBEDDING(
    ALL_MINILM_L6_V2 USING code_diff AS data
);

CREATE VECTOR INDEX idx_pr_hnsw
ON pr_code_review_vectors (pr_embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Bind :query_vector to a VECTOR(384, FLOAT32) embedding of the current PR.
SELECT
    pr_id,
    pr_title,
    VECTOR_DISTANCE(pr_embedding, :query_vector, COSINE) AS cosine_distance
FROM pr_code_review_vectors
WHERE pr_embedding IS NOT NULL
ORDER BY cosine_distance
FETCH APPROX FIRST 5 ROWS ONLY;
