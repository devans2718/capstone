# GraphRAG ablation: without and with `communities`

48 runs, 3 per question × condition, claude-opus-5-5; total $1.28. A = the six reviewed retrievals; B = seven, `communities` included. `outcome` and `same outcome` are the scoring rule's labels only. `daniel` is Daniel's hand-check verdicts, over the runs he checked (empty until he has); Claude's check is in the hand-check sheet and the agreement line, not in the counts.

| question | A routing | A keys | A numbers | A outcome | A daniel | A same retrieval | A same outcome | B routing | B keys | B numbers | B outcome | B daniel | B same retrieval | B same outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q1 | 3/3 | 12/12 | – | 3 correct |  | 3/3 | 3/3 | 3/3 | 9/9 | – | 3 correct |  | 3/3 | 3/3 |
| q2 | 3/3 | 18/18 | 45/45 | 3 correct |  | 3/3 | 3/3 | 3/3 | 19/19 | 53/53 | 3 correct |  | 3/3 | 3/3 |
| q3 | 3/3 | 30/30 | 41/41 | 3 correct |  | 3/3 | 3/3 | 3/3 | 34/34 | 46/46 | 3 correct |  | 3/3 | 3/3 |
| q4 | 3/3 | 30/30 | 63/63 | 3 correct |  | 3/3 | 3/3 | 3/3 | 36/36 | 66/66 | 3 correct |  | 3/3 | 3/3 |
| q5 | 3/3 | 56/56 | 15/15 | 3 partial |  | 3/3 | 3/3 | 3/3 | 45/45 | 15/15 | 3 partial |  | 3/3 | 3/3 |
| q6 | 3/3 | 21/21 | 58/58 | 3 correct |  | 3/3 | 3/3 | 3/3 | 24/24 | 55/55 | 3 correct |  | 3/3 | 3/3 |
| q7 | n/a (concept 2, why_failed 1) | 4/4 | 1/1 | 1 wrong, 2 correct refusal |  | 2/3 | 2/3 | 3/3 | 28/28 | 58/58 | 3 correct |  | 3/3 | 3/3 |
| q8 | n/a (why_failed 3) | 15/15 | 10/10 | 3 wrong |  | 3/3 | 3/3 | 3/3 | 28/28 | 49/49 | 3 correct |  | 3/3 | 3/3 |

Questions: q1 What's a flap?; q2 How did criterion 2 get its final form?; q3 Why did the pilot fail, and what's planned to test it?; q4 How does WavJEPA's best layer compare with MFCC under each split condition?; q5 What does registration revision 3 specify that the pilot never used?; q6 What did the span grid find?; q7 Do the flaps' phonetic environments group into communities, and do those communities line up with underlying /t/ versus /d/?; q8 What do the encoders' representations group the flaps by?

Agreement on the hand-checked runs: automatic vs Claude's check: 14/20.
