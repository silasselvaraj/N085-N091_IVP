# Research idea menu

A brainstorm of possible research directions, each grounded in an IVP concept this project
actually implements (see the corresponding numbered script) rather than a generic reading
list. This is a menu to pick from and expand, not a finished paper — each entry sketches the
question, why it's interesting, and one concrete next step.

## 1. CIELAB colour-space clustering for coat colour (stage 8)

**Question:** does clustering in CIELAB space, then merging clusters by nearest named
colour, actually produce more stable colour labels across lighting conditions than
clustering in RGB or HSV would?

**Why it's interesting:** a solid black dog under mixed lighting can k-means-split into 3+
clusters by shade alone (shadow/mid-tone/highlight) — this project's own fix is to merge
clusters that name to the same vocabulary term *after* clustering, rather than trying to
prevent the over-split in the first place. Is there a colour space, or a pre-clustering
normalisation, where the over-split doesn't happen at all?

**Next step:** re-run stage 8 on the same photo under 3-4 synthetically relit versions
(different white points, different shadow contrast) and measure how often the final merged
region count and names stay stable, in CIELAB vs. HSV vs. raw RGB clustering.

## 2. Energy-based out-of-distribution detection vs. max-softmax (stage 12)

**Question:** for a 120-way fine-grained classifier, how much does an energy score
(`-T * logsumexp(logits/T)`) actually separate in-distribution dogs from out-of-distribution
inputs (other animals, objects, blurry noise) better than the simpler "1 - max softmax
probability" baseline?

**Why it's interesting:** this project chose energy-based scoring and measured an OOD AUROC
of 0.982 on its (admittedly thin, 14-image) held-out set — but never published a head-to-head
number against max-softmax on the *same* set, which is the natural ablation.

**Next step:** compute both scores over the same OOD set stage 12 already has access to
(`fixtures/real/labels.json`'s non-dog images) and compare ROC curves directly.

## 3. Temperature-scaling calibration (Guo et al. 2017) (stage 12)

**Question:** does a single scalar temperature, fit once on validation logits, stay
well-calibrated across genuinely different input distributions — studio-lit dataset photos
vs. real phone snapshots vs. this showcase's own background-suppressed crops?

**Why it's interesting:** temperature scaling is popular because it's cheap and doesn't
change the model's ranking of classes, only the sharpness of the probabilities — but it's a
single number fit on one distribution, and this project's own honest gap between validation
ECE (0.023) and test ECE (0.029) already shows some drift.

**Next step:** measure ECE separately on Stanford Dogs test-split crops vs. a small
hand-collected real-photo set (PROJECT_STATE.md §3.1 already flags this second set as
"doesn't exist yet" — collecting even 30-40 photos would let this question be answered for
real).

## 4. Keypoint-based morphometric measurement and its view/posture failure modes (stages 6, 7)

**Question:** the real project measured its view classifier at 14.3% and its posture
classifier at 25.0% accuracy on a 24-image hand-labelled set, traced to genuine overlap
between true view classes on every geometric signal tried (lateral spread, box aspect
ratio, mask eccentricity) — is there a signal that *doesn't* overlap, or is view/posture from
a single 2D image with no depth ground truth fundamentally under-determined for this task?

**Why it's interesting:** this is a case where more tuning provably wasn't the answer (three
independent signals were tried and all failed the same way) — which makes it a genuinely
open question rather than an engineering backlog item.

**Next step:** the most promising untried direction per PROJECT_STATE.md §3.2 is a small
learned classifier over the keypoints directly (rather than a hand-built geometric rule),
but it needs a class-balanced labelled set an order of magnitude larger than the current 24
images to avoid overfitting 5 views × 6 postures. Sizing that collection effort concretely
would itself be a useful first output.

## 5. Transfer learning strategy: frozen linear probe vs. two-stage fine-tune (Part 6, stage 12's checkpoint)

**Question:** this project's baseline was a frozen DINOv2 ViT-S/14 linear probe (83.9% val
top-1), beaten by a fully fine-tuned ConvNeXt-Tiny (86.8%) — how much of that 2.9-point gap
is the backbone architecture, and how much is simply that fine-tuning updates more
parameters on a 14,211-image training set that a frozen probe can't use?

**Why it's interesting:** the project never ran the missing cell of that 2×2 (a fine-tuned
DINOv2, or a frozen-backbone ConvNeXt-Tiny linear probe) — so architecture and
training-strategy are confounded in the one comparison that was actually made.

**Next step:** run the missing cell(s) and report whether the gap holds, narrows, or
reverses — a fine-tuned DINOv2 in particular might close most of the distance, which would
say the backbone matters less than the training strategy here.

## 6. Segmentation mask-quality scoring independent of the model's own IoU estimate (stage 3)

**Question:** SAM2 reports its own IoU estimate per mask hypothesis — this project
deliberately doesn't trust that number and instead measures box-fill, solidity, boundary
smoothness and edge contact independently. How well does the model's own IoU estimate
actually correlate with this independent quality score, across a range of real photos?

**Why it's interesting:** "a model can be confident and wrong" is this project's recurring
design principle (it's also why ViTPose+ small was rejected in Part 4 despite 0.85-1.00
reported confidence on broken landmarks) — this is the one place a direct, cheap
correlation study would either support or complicate that principle with real numbers.

**Next step:** log both scores across the ~20,000 preprocessed Part 5 dataset images (the
segmentation step already ran once for every one of them) and plot the independent quality
score against SAM2's own IoU estimate.

## 7. The mixed-breed/open-set problem in fine-grained classification generally (stage 12)

**Question:** this project's mixed-breed heuristic — "if the top prediction isn't dominant
and the plausible candidates span multiple breed groups, flag as possibly mixed" — needs a
breed→group mapping that Part 7 never built, so it's a no-op in production today (this
showcase's script 12 uses a dozen-breed illustrative stand-in to demonstrate the mechanism
only). Once a real mapping exists, how often does this heuristic actually agree with a
genuinely mixed-breed dog vs. simply flagging any photo the classifier finds ambiguous for
other reasons (poor lighting, unusual pose, a purebred dog that resembles two breeds)?

**Why it's interesting:** "mixed breed" is arguably what a meaningful fraction of real
uploaded photos actually are, and a heuristic that can't distinguish "genuinely mixed" from
"just uncertain" risks being uninformative in exactly the cases users care about most.

**Next step:** once Part 7's breed→group database exists, validate this heuristic against
a small set of dogs with owner-reported mixed ancestry (even self-reported, imperfect
ground truth would be more signal than the current zero).
