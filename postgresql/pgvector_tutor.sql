--Create a vector column with 3 dimensions
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));

--Insert vectors
INSERT INTO items(embedding) VALUES ('[1,2,3]'),('[4,5,6]');

--Get the nearest neighbor by L2 distance
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;

--Insert new vectors
INSERT INTO items (embedding) VALUES ('[7,8,9]'), ('[2,3,4]'), ('[5,0,2]');

--Get the nearest neighbor by L2 distance
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]';

--See all values
SELECT * FROM items;

--Calculating L2 distance from first rows embeddings to other rows embeddings
WITH first_embedding AS (
	SELECT embedding FROM items WHERE id = 1
)
SELECT i.id, i.embedding, i.embedding <-> fe.embedding AS distance
FROM items i, first_embedding fe
WHERE i.id !=1
ORDER BY distance;

--Calculating L2 distance from second row to another embeddings row
WITH snd_embedding AS (
	SELECT embedding FROM items WHERE id = 2
)
SELECT i.id, i.embedding, i.embedding <-> s.embedding AS distance
FROM items i, snd_embedding s
WHERE i.id != 2
ORDER BY distance;

--Calculating L1 distance
SELECT id, embedding, '[3,1,2]' <+> embedding AS L1_distance FROM items
ORDER BY L1_distance;

--Calculating L1 distance to other row
WITH first_embedding AS (
	SELECT embedding FROM items WHERE id = 1
)
SELECT i.id, i.embedding, i.embedding <+> fe.embedding AS L1_distance
FROM items i, first_embedding fe
WHERE i.id != 1
ORDER BY L1_distance;