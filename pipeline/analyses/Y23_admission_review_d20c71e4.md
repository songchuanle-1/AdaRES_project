# Y23 Admission Review

Owned tick d20c71e418354a6b818d9f37ab72fc7b.
Scientific status: eligible for falsifiable training, not proven efficacy.

## Diagnosis
Actual W2 training semantic alignment includes language padding in its
query-to-token denominator. Train64/256expressions,63paddedscans: all63 have
native padding gradients. Padding-only projected-token replacement changes
native loss by mean.0001416538624; candidate changes0. Validtokens untouched.
This is annotation support correction, not confidence calibration/threshold/
temperature/weight/valid-feature scanning. No claimed task benefit from size.

## Controls
32all-valid cases: candidate/parent loss and gradient difference0.
Appending padding leaves loss/gradient invariant within float32 reduction
noise4.77e-7/5.96e-7. Empty/nonprefix language support failsclosed.
Candidate padding gradient/counterfactual0 across train64, valid/query
gradients finite/nonzero. Every inherited inference/model module is unchanged.

## Native Backward
ReleasedSPFormer initialization,2native paired batches: parent padding-gradient
norms.9633/1.4002, candidate0; interventiongradientnorm1.4925/1.6428.
Valid/query gradients nonzero,400decodergradients finite/nonzero.
tools.train native2optimizersteps completed. Model/AdamW/PolyLR checkpoint
restore exactly before continuation; nextstep lossdiff0/modelmax2.29925e-5/
optmax1.74183e-8 retained. Not whole-dataloader or bitwise replay proof.

## Resources
164629637parameters for both/0added;128queries/1selected unchanged.
Largest frozen train pair1055503points/7618SP:
candidate/parentmedianstep1.120161/1.116741s, reserved14.076172/14.078125GiB.
Other large387654points/3259SP:.492189/.498768s,5.636719GiBboth. NoOOM.
Serial samecheckpoint train8+largest9inferences,3measurementsafter2warmups:
memory identical eachscene,largestreserved1.7578125GiB;
largestmedian.177614/.181404s. Timings are not stablep95 or trainedParetoproof.

## Independence And Constraints
Physical sevenfile package/config copied from W2. ASTcompiled, no
cross-variant production imports; ipdn/dec/backbone/sample_model unchanged,
only loss semantic support and initcomment changed. Config exactW2.
Remote testbundle hashes equal; commit identity still must be verified.
All seven goal constraints and three bans retained in motivation/design.
server-2GPU0 only prospective launch, GPU1Y18 untouched; no otherproject/
scheduler operations. Fixed30epochs/seed1999/releasedSPFormer.

## Repeat Risk
Initial train8 prediction test: candidate/parent selected/binary/sample flips0,
but candidate selfrepeat scene0002 had16879binary flips/maxprob.999991.
Followup exactcacheddecoderinputs: mask/indicator differences0 andsample
mismatches0;8scene newnative repeat bothmodels0flips. Then frozen anomaly
scene16repeats/model: W2 flipped15/16(afterfirstbaseline),same16879points,
4FPSseedmismatches/226sampleindexmismatches/2selectionflips;Y23thisaudit0.
Thus inherited end-to-end repeat instability is observed, not a candidate
inference-source change. UnderlyingFPS/backbone cause remains unknown.
No claim candidate is globally stable or inference repeat gate universallyPASS.
All original failedreadouts preserved; posttraining independentrepeats required.

## Disposition
Six gates reviewed as executable: diagnosis, matchedcontrols, nativebackward,
resources, physicalindependence, goalconstraints. Admit for native fixed30
training only after scoped testedcommit/hash receipt and canonical dispatcher.
No trainedefficacy, universalrepeatability or literaturenovelty claim.
Frozen fullvalidation scene-clustered W2/B1/subgroupharm/cost plan in
pipeline/designs/exp_design_Y23_d20c71e4.md. No validation used to tune proposal.
