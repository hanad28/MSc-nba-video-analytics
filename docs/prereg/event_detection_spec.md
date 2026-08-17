# Event detection: labelling schema and scoring specification

Locked 12 August 2026, before any labelling or implementation. Stage 6,
`basketball/events/event_detector.py`.

---

## Part 1 — Ground truth schema

### File

`data/annotations/event_gt.csv`, tracked in git alongside the possession and team
ground truth.

### Columns

| Column | Type | Meaning |
|---|---|---|
| `clip` | str | `clip_1`, `clip_2`, `clip_3` |
| `event_id` | int | Sequential within a clip, from 1, in chronological order |
| `event_type` | str | From the vocabulary below |
| `start_frame` | int | The frame the ball leaves the originating player's control |
| `end_frame` | int | The frame the ball comes under the receiving player's control |
| `from_track_id` | str | Track ID of the originating player, or `unknown` |
| `to_track_id` | str | Track ID of the receiving player, or `unknown` |
| `from_team` | str | `1`, `2` or `unclear` |
| `to_team` | str | `1`, `2` or `unclear` |
| `notes` | str | Free text; anything that made the call difficult |

`from_track_id` and `to_track_id` are strings so `unknown` and a numeric ID share one
column, following the possession ground truth's `holder` convention rather than mixing
integers with sentinels.

For an event with no receiving player (a shot that scores, a ball leaving play),
`to_track_id` is `unknown` and `end_frame` is the frame the outcome resolves.

### Vocabulary

Label what happened, not what the detector could represent.

| `event_type` | Definition |
|---|---|
| `pass` | Deliberate transfer between teammates, received by the intended teammate |
| `interception` | Opponent takes the ball while it is in transit between teammates |
| `steal` | Opponent takes the ball directly from a player in possession, not in transit |
| `rebound` | Recovery of a missed shot, by either team |
| `block` | Shot deflected by a defender |
| `loose_ball` | Ball uncontrolled after a deflection or lost handle, then recovered |
| `turnover` | Possession lost without an opponent taking it directly (bad handle, out of play) |
| `shot` | Attempt on the basket, made or missed |
| `inbound` | Ball returned to play from out of bounds |

If an event genuinely fits none of these, label it `other` and describe it in `notes`.
Do not stretch a definition to fit.

### Mapping to the detector's two classes

The detector can only emit `pass` and `interception`. This mapping is fixed now, before
any results exist:

| Ground truth type | Detector should emit |
|---|---|
| `pass` | `pass` |
| `interception` | `interception` |
| `steal` | `interception` — a possession change to the opposing team |
| `rebound` (to a teammate of the shooter) | `pass` — the detector cannot distinguish it |
| `rebound` (to the opposing team) | `interception` — likewise |
| `block`, `loose_ball`, `turnover`, `shot`, `inbound` | **unrepresentable** |

The unrepresentable set is the measurement that matters. It quantifies the ceiling
imposed by a two-class scheme over possession changes, rather than asserting it. Report
the count of unrepresentable events alongside precision and recall, never folded into
them.

Rebounds map by team because the detector has no way to know a shot occurred. This is
deliberate and is itself a finding: the scheme silently relabels rebounds as passes or
interceptions, and the count of those cases belongs in the discussion.

### Labelling rules

1. **Watch the raw clip, not the annotated output.** The annotated video shows the
   pipeline's possession highlight, and labelling against it would write the system's
   errors into the ground truth built to score them. Use `data/raw/clip_N.mp4`.
2. **Track IDs come from the annotated output, consulted only to read an ID** once the
   event and its frames are already decided from the raw clip. Note in `notes` if a
   participant carries no visible track.
3. **`start_frame` is release, `end_frame` is reception**, judged as precisely as
   scrubbing allows. Where they are the same frame, record the same number twice.
4. **When the type is uncertain between two categories**, pick the better fit and say so
   in `notes`. Do not invent a hybrid.
5. **Label every possession change**, including ones the detector cannot represent.
   Their absence would make the unrepresentable count unmeasurable.

### Expected scale

The 18 July narrative implies roughly seven events across 534 frames: clip_1 a dribble
and a block, clip_2 four passes and a dunk, clip_3 a lost handle, an interception, two
passes and a dunk. That narrative is a memory of a visual check and is **not** ground
truth. Expect the labelled count to differ, and treat any large divergence from it as
worth investigating rather than as an error in the labelling.

---

## Part 2 — Scoring specification

### Matching

A predicted event matches a true event when all of the following hold:

1. The predicted frame falls within `[start_frame - T, end_frame + T]`.
2. The predicted type equals the true type's mapped detector class.
3. The match criterion below is satisfied.

`T` is the tolerance in frames. Primary results at `T = 10`; the same table also
reported at `T = 5` and `T = 20` so the reader sees the sensitivity rather than
trusting one value. Ten frames is a third of a second at 30 fps and is justified
mechanically, not chosen for convenience: `hold_threshold = 5` means the receiver's
streak must accumulate before possession confirms, and the measured misattribution on
clip_2 frames 118 to 132 began four frames before the ball was released.

**Match criteria, both reported:**

- **Primary — type and team.** The predicted event's team must equal the true event's
  mapped team (the possessing team for a pass, the taking team for an interception).
- **Secondary — type, team and participants.** Both track IDs must additionally match.

The primary criterion is the headline because track fragmentation means one physical
player can hold several track IDs, so strict participant matching would penalise the
tracker rather than the event logic. The gap between the two figures measures exactly
that identity error, which is why both appear.

### Assignment

One-to-one, greedy by temporal proximity: for each true event in chronological order,
match the nearest unmatched qualifying prediction. A prediction already matched cannot
match again. Unmatched predictions are false positives; unmatched true events are false
negatives.

### Metrics

Per clip, and pooled at the event level (never as a mean of per-clip rates):

- True positives, false positives, false negatives — **as raw counts, always**
- Precision, recall, F1 — secondary, never shown without the counts beside them
- Unclassifiable transitions emitted (the third output class)
- Unrepresentable true events (from the mapping above)

**No percentage appears without its n.** With roughly seven events total, one missed
pass on clip_2 moves recall by 25 points. No confidence intervals are reported: at this
sample size they would be theatre.

### The two-way decomposition

The detector runs twice over the same clips, differing only in its possession input:

| Run | Possession input | Team input | What it measures |
|---|---|---|---|
| **Predicted** | pipeline `possession.pkl` | predicted | End-to-end performance |
| **Oracle-possession** | `possession_gt_per_frame.csv` | predicted | The ceiling this event logic can reach given perfect possession |

The gap between them is possession's inherited contribution to event error. Without it
the phase cannot separate its own errors from an upstream stage already measured at 41
and 53 per cent false-positive rates on clips 2 and 3.

Team assignment stays predicted in both runs. Substituting team ground truth is not
possible: it covers only 31 per cent of named-holder frames and its covered subset is
strongly team-imbalanced.

The oracle run needs the ground-truth holder list in the same shape the detector
expects, `list[int]` with `-1` for nobody. `unclear` frames become `-1`, and that
substitution is stated in the results rather than hidden, since it makes the oracle
slightly pessimistic on the 46 unclear frames.

---

## Part 3 — Pre-registered predictions (E1–E5)

Locked before implementation and before labelling. Refuted predictions are documented
as refuted, not revised.

### E1 — The clip_2 bounce pass produces a false interception, detected early

The transition sequence across clip_2 frames 118 to 132 (track 10 team 1, track 1
team 2, track 7 team 1) will cause the detector to emit an interception followed by a
pass, where ground truth records one pass. The interception will be timestamped before
the true `start_frame` of that pass.

*Basis:* measured directly on 12 August. The possession list contains this sequence;
the detector reads transitions in that list.

*Refuted if:* no interception is emitted in that window, or it falls at or after the
true release frame.

*Confidence:* high. This is close to a mechanical consequence rather than a forecast.

### E2 — The clip_2 give-and-go is unrecoverable in principle

At least one true pass in clip_2 frames 92 to 134 will be missed under the predicted
run, and recovered under the oracle-possession run.

*Basis:* the 18 July investigation found `find_holder` correctly identifies player 10 at
frames 93, 94, 106 and 108, never for the five consecutive frames confirmation
requires, so possession reads `-1` across the window. The event logic cannot recover a
transition the possession list does not contain.

*Refuted if:* the pass is detected under the predicted run, or is missed under both.
The second case would be the more interesting refutation, since it would mean the
failure is not purely inherited.

*Confidence:* high on the miss, medium on the oracle recovery.

### E3 — Most event error is inherited, not introduced

Across all three clips, the oracle-possession run will record at least twice as many
true positives as the predicted run.

*Basis:* possession's false-positive rate is 41 to 53 per cent on two clips, and every
phantom possession is a candidate spurious transition.

*Refuted if:* the ratio is below 2. That would mean the event logic contributes more of
its own error than the upstream stage does, which would redirect the phase's conclusion
entirely.

*Confidence:* medium.

### E4 — False positives outnumber true positives on the predicted run

On clips 2 and 3 combined, the predicted run will emit more false-positive events than
true positives.

*Basis:* every possession false positive is two transitions, in and out, so phantom
possessions should generate events at roughly twice their own rate.

*Refuted if:* false positives are fewer than true positives. Given a true-event count in
single digits, this prediction is fragile and may resolve on a difference of one or two
events; report the counts and note the fragility rather than treating the outcome as
decisive.

*Confidence:* medium-low.

### E5 — A material fraction of real events is unrepresentable

At least 20 per cent of labelled true events across the three clips will fall in the
unrepresentable set: blocks, loose balls, turnovers, shots or inbounds.

*Basis:* clip_1's narrative contains a block and no passes at all; clip_3's contains a
lost handle before the interception. Neither maps to the two-class scheme.

*Refuted if:* below 20 per cent, which would mean the two-class scheme covers this
footage better than expected and the ceiling argument is weaker than anticipated.

*Confidence:* medium.

---

**Amendment, 12 August 2026, before labelling completed and before any results
exist.** `interception` as originally defined ("opponent takes the ball while it is
in transit between teammates") excluded a contested airborne ball following a lost
handle. Widened to: *an opponent gains a ball that is in transit or otherwise not
under any player's control, where the gain is contested or occurs against an
opponent's active attempt to recover it.* `loose_ball` is correspondingly narrowed
to uncontested recoveries and recoveries by the losing player's own team.

Rationale: the original wording made clip_3's turnover unrepresentable purely on a
technicality of how possession was lost, when the basketball event — an opponent
taking the ball off a mistake — is the thing the detector exists to catch. Recorded
as an amendment rather than a silent edit.

---

## Order of work

1. Lock this document (commit it).
2. Label `event_gt.csv` from the raw clips.
3. Design and implement `EventDetector`, spec first.
4. Score both runs, all three tolerances, both match criteria.
5. Read the results against E1–E5, scoring each as supported, refuted or
   indeterminate against the stated thresholds and nothing else.
6. Inspect the frames behind anything surprising before treating it as a finding.
7. Write up.

Steps 2 and 3 are independent and can happen in either order, but both precede step 4.
Implementation must not begin before this document is committed.