SELECT id, quy_an FROM user_embeddings;

SELECT id, quy_an FROM user_embeddings

DROP TABLE user_embeddings;

--Calculating L2 distance from first embedding of quy_an to another of quy_an
WITH qa_first_embedding AS (
	SELECT quy_an FROM user_embeddings WHERE id = 1
)
SELECT u.id, u.quy_an <-> qa.quy_an AS distance, u.quy_an
FROM user_embeddings u, qa_first_embedding qa
WHERE u.id != 1
ORDER BY distance;

--Max L2 distance
WITH qa_first_embedding AS (
	SELECT quy_an FROM user_embeddings WHERE id = 1
)
SELECT MAX(u.quy_an <-> qa.quy_an) AS max_distance
FROM user_embeddings u, qa_first_embedding qa
WHERE u.id != 1

--Calculating L2 distance from first embedding of quy_an to another of quy_an
--DECREASING
WITH qa_first_embedding AS (
	SELECT quy_an FROM user_embeddings WHERE id = 1
)
SELECT u.id, u.quy_an <-> qa.quy_an AS distance, u.quy_an
FROM user_embeddings u, qa_first_embedding qa
WHERE u.id != 1
ORDER BY distance DESC;

--Calculating L2 distance from first embedding of quy_an to another of minh_hai
WITH qa_first_embedding AS (
	SELECT quy_an FROM user_embeddings WHERE id = 1
)
SELECT u.id, u.minh_hai <-> qa.quy_an AS distance, u.minh_hai
FROM user_embeddings u, qa_first_embedding qa
ORDER BY distance DESC;


--Calculating L2 distance from first embedding of quy_an to another of trungviet
WITH qa_first_embedding AS (
	SELECT quy_an FROM user_embeddings WHERE id = 1
)
SELECT u.id, u.trungviet <-> qa.quy_an AS distance, u.trungviet
FROM user_embeddings u, qa_first_embedding qa
ORDER BY distance DESC;

--Calculating L1 distance form first quy_an to another of embedding quy_an
WITH qa_first_embedding AS (
	SELECT quy_an FROM user_embeddings WHERE id = 1
)
SELECT u.id, u.quy_an <+> qa.quy_an AS distance, u.quy_an
FROM user_embeddings u, qa_first_embedding qa
WHERE u.id != 1 ORDER BY distance ASC; 

--Calculating L1 distance form first quy_an to another of embedding trungviet
WITH qa_first_embedding AS (
	SELECT quy_an FROM user_embeddings WHERE id = 1
)
SELECT u.id, u.trungviet <+> qa.quy_an AS distance, u.trungviet
FROM user_embeddings u, qa_first_embedding qa
WHERE u.id != 1 ORDER BY distance ASC; 

--Calculating L1 distance form first quy_an to another of embedding quy_an
WITH qa_first_embedding AS (
	SELECT minh_hai FROM user_embeddings WHERE id = 1
)
SELECT u.id,  (1 - (u.minh_hai <=> qa.minh_hai)) AS distance, u.minh_hai
FROM user_embeddings u, qa_first_embedding qa
WHERE u.id != 1 
ORDER BY distance ASC;
LIMIT 1; 

--Calculating L1 distance form first quy_an to another of embedding trungviet
WITH qa_first_embedding AS (
	SELECT quy_an FROM user_embeddings WHERE id = 1
)
SELECT u.id, u.trungviet <=> qa.quy_an AS distance, u.trungviet
FROM user_embeddings u, qa_first_embedding qa
WHERE u.id != 1 ORDER BY distance desc; 