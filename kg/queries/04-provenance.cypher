// 04 · Provenance: how criterion 2 got its final form.
// Its revisions, the registrations that set them, the paper section that
// justified the change, and the dated amendments to revision 3.

// graph
MATCH lineage = (c3:Criterion {key: 'crit-rev3-2'})-[:SUPERSEDES*]->(:Criterion {revision: 1})
MATCH regs = (:Registration)-[:SETS_CRITERION]->(c:Criterion {number: 2})
OPTIONAL MATCH why = (:Paper)-[:JUSTIFIES]->(c3)
OPTIONAL MATCH amend = (:Amendment)-[:AMENDS]->(:Registration {revision: 3})
RETURN lineage, regs, why, amend;

// table
MATCH (r:Registration)-[:SETS_CRITERION]->(c:Criterion {number: 2})
OPTIONAL MATCH (p:Paper)-[j:JUSTIFIES]->(c)
RETURN r.revision AS revision, r.commit AS commit, r.date AS date,
       c.threshold AS threshold, c.note AS note,
       p.short + ' ' + j.section AS justified_by
ORDER BY revision;
