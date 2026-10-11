// 01 · Finding entities with MATCH
// The four criteria the pilot was gated on, as registered in revision 3.

// table
MATCH (:Registration {revision: 3})-[:SETS_CRITERION]->(c:Criterion)
RETURN c.number AS criterion, c.statement AS statement, c.threshold AS threshold
ORDER BY criterion;
