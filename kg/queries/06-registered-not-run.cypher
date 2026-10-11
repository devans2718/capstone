// 06 · Registered but not run: what revision 3 SPECIFIES that the pilot
// never USED or REALIZED.

// table
MATCH (:Registration {revision: 3})-[s:SPECIFIES]->(x)
WHERE NOT EXISTS { (:Run)-[:USED|REALIZES]->(x) }
RETURN labels(x)[0] AS kind, coalesce(x.name, x.key) AS item, s.role AS role,
       x.note AS note;
