# Project Proposal

**Project title:** Salamander Spotter — Open-Set Re-Identification of Fire Salamanders from Natural Spot Patterns

**Team members:** Micha Vardy 

**Project category:** Computer Vision — open-set re-identification / metric learning (wildlife monitoring)

---

**background.** In Kibbutz Sasa (where I lived), there is a species of Salamander that is considered unique in the world.  there is a team of Biologists that wish to classify this new species however require a non invasive population census which is challenging.  the entire kibbutz has been taking photographs of these salamanders whenever they see them and sending them to the lead researcher by whatsapp as a community biology project.  I have tried in the past to take on the challenge of open set re-identification using traditional methods and tools and failed.  this is a second attempt at this process using deep learning.

**Problem.** Fire salamanders (*Salamandra salamandra*) carry a lifelong, individually unique pattern of yellow spots — a natural fingerprint. We will build a system that, given a single field photograph, decides whether the animal is already enrolled in a database (and which individual it is) or has never been seen before. This is **open-set re-identification**, not classification: the roster of individuals is unbounded and grows with every survey. The deliverable is a non-invasive population census: an individual count for a season's photographs, without capture or tagging.

**Challenges.** Only the spot pattern is stable; pose, body curl, camera angle, lighting, skin wetness and background all vary between sightings of one animal. The data is long-tailed and few-shot — most individuals appear in only two or three photographs — so a per-individual classifier is impossible, and the model must learn a transferable notion of *same versus different pattern*. Spot detection is itself unstable: roughly a fifth of an animal's spots appear or vanish between its own photographs, so matching must succeed on partial correspondence. Finally, open-set operation needs a calibrated threshold: a false merge collapses two animals into one profile and irreversibly deflates the count.

**Data.** Roughly 1,900 field photographs of ~750 individuals from three Israeli sites, with ~40,000 annotated spots. Identity is encoded in the filename convention `<code>_<individual>_<instance>`, which supplies the positive and negative pairs for training and evaluation. Each image can extract per-spot contours and masks, a whole-body mask and a head-to-tail axis, stored in DuckDB. Generatively re-rendered views of enrolled animals add training-only positive pairs for single-photo individuals.

**Proposed method.** A spot-centric, two-level matcher rather than a whole-image embedding. (1) Segment the spots and body, then derive a body-intrinsic coordinate frame — arc length along the head-to-tail midline plus lateral offset — so a spot's position is invariant to translation, scale, rotation and body curl. (2) Encode each spot as elliptic Fourier shape descriptors concatenated with that position. (3) Score a photo pair by per-spot correspondence, aggregated through a learned, individual-agnostic voting rule over match statistics (coverage, support, geometric consistency), with spots weighted by learned distinctiveness. (4) Emit **match / new / abstain** from a calibrated threshold.

**Existing work.** HotSpotter and Wildbook for patterned-animal identification; whale-shark ID adapting the Groth star-pattern matching algorithm; deep metric learning (triplet and supervised-contrastive losses); permutation-invariant set encoders (DeepSets, Set Transformer); and RANSAC spatial verification from local-feature matching.

**Planned improvements and innovations.** Body-intrinsic curvilinear spot coordinates that survive body curl; learning the *aggregation rule* over fixed spot descriptors rather than the spot representation, so pair-level supervision transfers to unseen individuals; constellation-level geometric invariants with RANSAC re-ranking; and a calibrated three-way match / new / abstain head, evaluated with precision-weighted census metrics (F0.5, count bias, coverage at fixed precision) under individual-level splits anchored by chance and oracle baselines.
