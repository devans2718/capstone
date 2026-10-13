// 08 · The graph recomputes a registered number: how many `joint` test sites
// have their lemma in their probe's training set anyway? A joint probe trains
// on the sites sharing neither the test site's speaker fold nor its type fold.
// Registered (lemma-overlap-joint-20260926.tsv): 2,612 same label,
// 191 other label only, 7,586 unseen; 10,389 eligible sites.
// Needs the corpus layer (kg/build_corpus.py), which is built locally.

// graph: the lemma spanning the most type keys, its tokens and their speakers
MATCH (l:Lemma)<-[:HAS_LEMMA]-(:Token)-[:HAS_KEY]->(k:TypeKey)
WITH l, count(DISTINCT k) AS n_keys ORDER BY n_keys DESC, l.name LIMIT 1
MATCH p = (l)<-[:HAS_LEMMA]-(t:Token)-[:HAS_KEY]->(:TypeKey)
MATCH q = (t)-[:SPOKEN_BY]->(:Speaker)
RETURN p, q LIMIT 40;

// table
MATCH (s:Site {eligible: true})-[:IN_TOKEN]->(:Token)-[:HAS_LEMMA]->(l:Lemma)
OPTIONAL MATCH (l)<-[:HAS_LEMMA]-(:Token)<-[:IN_TOKEN]-(o:Site {eligible: true})
  WHERE o.fold_speaker_disjoint <> s.fold_speaker_disjoint
    AND o.fold_word_disjoint <> s.fold_word_disjoint
WITH s, collect(DISTINCT o.underlying) AS train_labels
WITH CASE WHEN s.underlying IN train_labels THEN 'same label'
          WHEN size(train_labels) > 0 THEN 'other label only'
          ELSE 'unseen' END AS lemma_in_training, count(*) AS sites
RETURN lemma_in_training, sites,
       round(100.0 * sites / 10389, 1) AS percent
ORDER BY sites;
