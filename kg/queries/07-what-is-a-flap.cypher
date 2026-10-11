// 07 · "What's a flap?": the concept path a reader without linguistics follows
// from the experiment and its task.

// graph
MATCH p = (:Experiment {name: 'English flapping'})-[:INVOLVES]->(:Concept)
OPTIONAL MATCH q = (:Task {key: 'task-voicing'})-[:INVOLVES]->(:Concept)
OPTIONAL MATCH r = (:Concept)-[:BROADER|INVOLVES]->(:Concept)
RETURN p, q, r;

// table
MATCH (:Experiment {name: 'English flapping'})-[:INVOLVES]->(c:Concept)
OPTIONAL MATCH (c)-[:BROADER|INVOLVES]->(d:Concept)
RETURN c.term AS term, c.definition AS definition, collect(d.term) AS related
ORDER BY term;
