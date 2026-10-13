// 09 · Evidence down to the sites: "Sites mislabelled?" was ruled out by a
// blind listening audit (pre-registration §7). Which audited sites did not
// clearly hold a flap? Verdicts sit on the SAMPLES edge: they belong to that
// audit, not to the site.
// Needs the corpus layer and the local review log (kg/build_corpus.py).

// graph: the finding, its evidence, and the sites the listener doubted
MATCH p = (:Finding {key: 'find-sites'})-[:EVIDENCED_BY]->(a:Artifact {key: 'art-audit-sample'})
          -[e:SAMPLES]->(s:Site)-[:IN_TOKEN]->(:Token)-[:SPOKEN_BY]->(:Speaker)
WHERE e.flap <> 'yes'
RETURN p;

// table: per question and class, as audit-20260928.txt reports them
MATCH (:Artifact {key: 'art-audit-sample'})-[e:SAMPLES]->(s:Site)
UNWIND ['flap', 'word', 'clip'] AS question
WITH question, s.underlying AS class, e[question] AS verdict
RETURN question, '/' + class + '/' AS class,
       sum(CASE verdict WHEN 'yes' THEN 1 ELSE 0 END) AS yes,
       sum(CASE verdict WHEN 'no' THEN 1 ELSE 0 END) AS no,
       sum(CASE verdict WHEN 'cant_tell' THEN 1 ELSE 0 END) AS cant_tell
ORDER BY question, class DESC;
