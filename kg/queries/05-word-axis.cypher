// 05 · Research query: how much survives as the lexical and speaker routes
// are closed? WavJEPA's best layer against MFCC, per split condition
// (mean pooling, full training sets).

// graph
MATCH p = (c:Curve {pooling: 'mean', subsampled: false})-[:UNDER_CONDITION]->(:Condition)
MATCH q = (c)-[:FOR_REPRESENTATION]->(:Representation)
RETURN p, q;

// table
MATCH (w:Curve {representation: 'wavjepa', pooling: 'mean', subsampled: false})
      -[:UNDER_CONDITION]->(k:Condition)<-[:UNDER_CONDITION]-
      (m:Curve {representation: 'mfcc', pooling: 'mean', subsampled: false})
RETURN k.name AS condition, k.held_out AS held_out,
       w.best_layer AS best_layer, round(w.best_balanced_acc, 3) AS wavjepa,
       round(m.best_balanced_acc, 3) AS mfcc,
       round(w.best_balanced_acc - m.best_balanced_acc, 3) AS gap,
       w.lookup_balanced_acc AS word_lookup
ORDER BY wavjepa DESC;
