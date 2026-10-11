// 02 · Traversing relationships: which script produced each of the run's files,
// and at which run_probes.py stage.

// graph
MATCH p = (:Run {run_id: 'pilot-20260926'})<-[:FROM_RUN]-(:Artifact)-[:GENERATED_BY]->(:Code)
RETURN p;

// table
MATCH (:Run {run_id: 'pilot-20260926'})<-[:FROM_RUN]-(a:Artifact)-[g:GENERATED_BY]->(c:Code)
RETURN a.name AS artifact, c.path AS generated_by, g.stage AS stage, a.commit AS committed_in
ORDER BY generated_by, artifact;
