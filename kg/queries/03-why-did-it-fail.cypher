// 03 · The anchor query, multi-hop: "Why did the pilot fail?"
// Run → failed Outcome → Criterion, the Findings that explain each failure,
// and what each Finding rests on. (No paper justifies a failed criterion:
// criteria 1, 3 and 4 are registered judgment calls.)

// graph
MATCH p = (:Run {run_id: 'pilot-20260926'})<-[:FROM_RUN]-(o:Outcome {passed: false})
          -[:EVALUATES]->(c:Criterion)
MATCH q = (f:Finding)-[:EXPLAINS]->(o)
OPTIONAL MATCH e = (f)-[:EVIDENCED_BY]->(:Artifact|Amendment)
RETURN p, q, e;

// table
MATCH (:Run {run_id: 'pilot-20260926'})<-[:FROM_RUN]-(o:Outcome {passed: false})
      -[:EVALUATES]->(c:Criterion)
MATCH (f:Finding)-[:EXPLAINS]->(o)
RETURN c.number AS criterion, o.detail AS result,
       collect(f.candidate_cause + ' — ' + f.status) AS explanations
ORDER BY criterion;
