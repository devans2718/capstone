// 03 · The anchor query, multi-hop: "Why did the pilot fail?"
// Run → failed Outcome → Criterion, the Findings that explain each failure,
// what each Finding rests on, and the papers behind each criterion.

// graph
MATCH p = (:Run {run_id: 'pilot-20260926'})<-[:FROM_RUN]-(o:Outcome {passed: false})
          -[:EVALUATES]->(c:Criterion)
MATCH q = (f:Finding)-[:EXPLAINS]->(o)
OPTIONAL MATCH e = (f)-[:EVIDENCED_BY]->(:Artifact|Amendment)
OPTIONAL MATCH j = (:Paper)-[:JUSTIFIES]->(c)
RETURN p, q, e, j;

// table
MATCH (:Run {run_id: 'pilot-20260926'})<-[:FROM_RUN]-(o:Outcome {passed: false})
      -[:EVALUATES]->(c:Criterion)
MATCH (f:Finding)-[:EXPLAINS]->(o)
RETURN c.number AS criterion, o.detail AS result,
       collect(f.candidate_cause + ' — ' + f.status) AS explanations
ORDER BY criterion;
