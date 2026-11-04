# Joint MC actor–critic phase v100 — 2026-10-10

User authorizes more tasks, iterative actor/critic updates, Q-head:slow-writer10:1
and eight-task batches as before. Continue v96 final actor, initially6300actorupdates;
critic70k/writer12k. Keep prior v96/v98 and offline frozen snapshots unchanged.

500 ADDITIONAL fresh IIDfriction U[.005,.015] tasks in exclusivejointnamespace,
250clean calibratednear /250broadspeedU[.247,.435], shuffled4+4 groups8 (last2+2).
Six identicalclean officialbaseprogramtrials per task, same initialphysicalstate,
controllerresetpertrial, allprior observation/actionhistory retainedwithintask.
Newseeds201xxxx distinctfromv96. Calibration outsidehistory, stop onfailednear
qualification withoutfrictionreplacement. Allmodelweights fixed duringa collection
group. Sixtrials use actor, notinitialbase-only. Noiseaccsigma.05 truncated±.15,
combinedacc bound±.5m/s², residualVbound±.1m/s; midpointintegrationdt.1; native
absoluteYenvelopepriority, offsetuncapped; Q andTTT actioneffective residualV/.5.

After eachgroup100criticupdates, everysecondcriticsteponeactorupdate, everytenth
criticstepalsooneslowwriterupdate. Totals6300Q/3150actor/630writer updates.
Eachcriticandactorbatch8DISTINCTtasks4near4broad uniformlyfromALLnewphasecollected
trainingtasks; onejudgedtrial/phase/query per task; actoronlyactivepushqueries,
criticallavailablephasequeries including sliding. Neveruse development/reserved
labels in optimization. Censoredtrials supplyhistory butnoMCtarget. No oldphase
trainingreplay pooled, though network/optimizerinitialization is preserved.

SingleQ, MCrecordedterminalreturnG/5, gamma1, Huberbeta1. This remains mixed
historicalbehavior-continuation fitting: MC replay doesnotrelabel oldtrials as
currentpolicyQ and exploratorystochasticcontinuations differfromdetactor.
NoTD/twincritics/targetnet/RUDDER introduced. Actorloss-negativeQ differentiated
through currentQactionandboundedcontroller, candidatenoiseexcluded.

ActorcontextDETACHED. Actorloss NEVERupdatescritic/writer/encoders. Writer
receives onlyMCcriticlossgradients with FULLcausal-prefixBPTT acrossall16frame
windows andprevioustrials. Allsampledcontexts reconstructed undercurrentwriter;
collectioncontexts storedforauditonly and NEVERusedforoptimizerreplay. Fast
TTTwrites stillat16observedframes under fixedcollectionweights. No returns,
rewards,success,friction,cohort orQ values enterTTTfeatures.

ResumeparentAdam/AdamW moments: headAdamWLR1e-4wd1e-4; writerAdamWLR1e-5wd1e-4;
actorAdamLR1e-5wd0beta(.9,.999)eps1e-8; eachparametergroupgradientnormclip1.
Finalcounters76300Q/12630writer/9450actor. Atomic rollingcheckpoint every1000Q
updates or600seconds atsafecompletedgroupboundaries, overwritingrollingonly;
finaljoint_final.ptpreserved. Saveall3optimizers,RNG,replayhashindex,groupstate.

Computepreflight requires causalcacheparity before/afterwriterupdate, exact
actor->writergradientexclusion, writerMCgradientroute, earliesthistorygradient,
controllerparity, eightdistincttaskbalancedqueries andoptimizerresume/cadence.
Independentpost-trainingaudit native/controller/features,groupweights,taskseeds,
causalcontexts andALL6300jointoptimizersteps/finalmomentsexactreplay.

Evaluatefinaldeterministicjointactor versusunchangedBASE on12existingdevelopment
tasks withsameprograms/snapshots asv96. Theseare reusedregressiontasks, notfresh
unseenvalidation; reservedtasksuntouched. 6trials/arm, separatehistories144slots.
Reportnear/broadpairedreturns/success/bootstrapCI, phaseproxyQAUROC/MAE and
bounds/envelope interventions. Threeinputselectedlow/middle/highvideos andQ/Y
plots capturedfromtheseevaluationstarts. No rerunphysicssolelyformedia.
Allscience runs throughcompute.sh on92c316f35f5f. CheckexternallyafterinitialETA
andfirstgroupmeasurement, thennearfinish; nofrequentmonitoringloop.
