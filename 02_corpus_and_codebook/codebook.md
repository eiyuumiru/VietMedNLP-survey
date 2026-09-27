# Codebook: Vietnamese Healthcare Text-Processing Survey

## 1. Purpose and Scope

This codebook documents the vocabulary and decision rules used to characterize the Vietnamese healthcare text-processing literature in the survey. It is restricted to the task, methodological-configuration, and data-construction taxonomies stated in the manuscript. The taxonomy is intended to preserve the conditions under which a system produces an output: task objective specifies **what** it produces; methodological configuration specifies **how** it represents inputs, connects components, specializes model behavior, and accesses knowledge; and data construction specifies how supervision and evidence resources are created. These dimensions are complementary and may co-occur; they are not mutually exclusive categories or a ranking of systems.

The corpus covers Vietnamese text-processing research in healthcare-related domains, including clinical text, biomedical literature, health news, medical question answering, COVID-19 information, mental health, traditional medicine, and healthcare regulation. The publication-date window is 2021-01-01 through 2026-05-31. The final corpus contains 33 canonical publications and 39 task-level annotations. The search used ACL Anthology, ACM Digital Library, IEEE Xplore, SpringerLink, and arXiv. A preprint and an archival version describing the same study are treated as one study, with the archival version preferred as the canonical record. Publisher pages are used only to resolve the archival or preprint canonical version after primary-database discovery; they are not an additional search source.

The unit of coding is as follows:

| Unit | Definition in this codebook |
|---|---|
| Study / canonical publication | One retained canonical publication. It is counted once in publication-level summaries. |
| Task-level annotation | A normalized task label assigned to a distinct evaluated output. A multi-task publication may receive more than one task label. The 39 annotations are not a publication count. |
| Benchmark resource | A resource entered as one benchmark row only when it supports evaluation of a defined task or output under an identifiable protocol. A resource used solely for training, continued pre-training, or inference-time grounding does not create a benchmark row. |

Boundary studies are studies with a substantive Vietnamese healthcare text component but a broader or audio-centered primary scope. They are retained with an explicit scope flag and are separated from core-only aggregates where applicable.

## 2. General Coding Principles

### 2.1 Evidence-based coding

Assign labels only from explicitly reported study objectives, evaluated outputs, reported evaluation protocols, and documented construction or validation conditions. Task assignment is based on the target output, not on the resources or models used. Do not infer a task from a model name, dataset name, architecture, or the presence of a resource alone. Likewise, do not infer data quality, clinical realism, generalization, answer reliability, leakage, template repetition, or clinical deployment relevance from a construction pathway or benchmark score.

Record methodological configuration independently from task objective. Evidence access, cross-lingual transfer, model adaptation, and data construction are cross-cutting facets rather than task labels. A component label is structural, not a quality claim.

### 2.2 Status classification

The manuscript explicitly uses **not reported** for unresolved task cases and states that such cases are not inferred. It also uses **not confirmed** for information that could not be verified from reviewed materials; this is not evidence that the underlying practice was absent.

| Status | Manuscript-grounded use |
|---|---|
| Reported | The manuscript records reported provenance, validation, access, documentation, protocol, or other evidence when it is stated in a reviewed source. A separate formal definition of the label `reported` is not specified in manuscript. |
| Not reported | Use for unresolved task cases rather than inferring a label. The manuscript also distinguishes information not reported sufficiently for coding in the prior-survey positioning table. |
| Not confirmed | Use when information could not be verified from reviewed materials. It must not be interpreted as evidence of absence. |
| Not applicable | Not specified in manuscript as a formal coding status. |
| Unclear | Not specified in manuscript as a formal coding status. |

### 2.3 Primary task, secondary tasks, and facets

Each study receives one primary task family according to its principal evaluated output. Assign secondary task labels only where distinct outputs are evaluated. A full system configuration may include a primary task, secondary task labels, and multiple cross-cutting facets. For example, evidence access, cross-lingual transfer, model adaptation, and data construction remain facets and do not constitute task labels.

### 2.4 Prohibition on nomenclature-based inference

Do not assign labels from model or dataset nomenclature. The manuscript explicitly requires task assignment from reported objectives and evaluated outputs, not from resources or models used. A Transformer with a CRF head is coded by its dominant encoder-based representation; a decoder that receives retrieved passages remains decoder-only and receives separate grounding and composition labels.

## 3. Record Schema

The following fields are grounded in the manuscript's described extraction scheme and taxonomy mapping. Where the manuscript does not define a field format, identifier convention, or controlled value beyond the information below, the entry is marked **Not specified in manuscript**.

| Field | Definition or permitted content |
|---|---|
| Study ID | Not specified in manuscript. |
| Citation | Paper title, year, venue, and publication status are extracted for each included study. Citation-key syntax is not specified in manuscript. |
| DOI/URL | DOI and URL may be recorded as bibliographic or supporting evidence. A required field format is not specified in manuscript. |
| Publication status | Archival publication, preprint, or technical report/thesis. When records describe the same study, prefer the archival publication as the canonical record. |
| Canonical-version status | One canonical record per study, selected after DOI and normalized-title duplicate matching and manual inspection. |
| Core/boundary flag | Use the explicit scope flag for a retained study with a substantive Vietnamese healthcare text component but a broader or audio-centered primary scope. |
| Task coverage | Primary and secondary Vietnamese healthcare NLP tasks; multi-label task coding is permitted. |
| Contribution type | Dataset, benchmark, pretrained model, knowledge resource, task-specific method, or deployed/prototype system. |
| Backbone | Feature-based/probabilistic, recurrent/convolutional, encoder-only, encoder--decoder, or decoder-only, when explicitly supported by the manuscript's taxonomy mapping or reported study evidence. |
| Composition | Single-model pipeline, retriever--ranker, retriever--generator, graph-enhanced composition, or multi-component/multi-agent orchestration. |
| Adaptation | Task-specific supervised training, full fine-tuning, continued pre-training/domain adaptation, prompting/in-context learning, instruction fine-tuning, parameter-efficient fine-tuning, or multi-stage adaptation, as reported. These attributes may co-occur. |
| Grounding | Closed-book/parametric-only, provided context/open-book, lexical/dense/hybrid retrieval, document RAG, or knowledge-graph grounding, as reported. |
| Data and evidence provenance | Data source, construction pathway, supervision, evidence grounding, and stated provenance limitations. |
| Annotation and expert validation | Annotation procedure, agreement or quality-control evidence, expert involvement, and clinical validation where reported. |
| Evaluation protocol | Dataset split, leakage controls, output format, baselines, metrics, and other protocol conditions used for comparison. |
| Release and reproducibility artefacts | Dataset size and domain, public availability, versioning, code, model, prompt, or other reusable artefacts, where reported. |
| Evidence anchor | Relevant page, section, table, figure, or resource link supporting the coded value. |
| Main findings and limitations | Reported findings, reported limitations, and implications. Source-reported findings must be distinguished from survey interpretation. |
| Coder | One author conducted study-level extraction and initial taxonomy coding. Individual identity, identifier, and per-record field format are not specified in manuscript. |
| Reviewer | A second reviewer examined taxonomy coding of all included studies against the operational rules. Individual identity, identifier, and per-record field format are not specified in manuscript. |
| Reconciliation status | Differences were resolved by jointly revisiting the full text and reported evaluation protocol until consensus. A formal status vocabulary is not specified in manuscript. |
| Governance fields | The manuscript records reported access and documentation information and states that its review is not an independent licensing, privacy, or governance audit. Exact governance-field names are not specified in manuscript. |

## 4. Rules for Determining Task Taxonomy Groups

### 4.1 Information Extraction and Structuring

**Operational definition.** Information Extraction and Structuring (IE) converts medical text into structured representations for retrieval, knowledge-base construction, or downstream reasoning. The task boundary is defined by the target representation, not by the architecture.

**Sub-labels represented in the final corpus.**

| Sub-label | Manuscript-grounded description |
|---|---|
| Flat NER | A named-entity recognition setting represented in the included corpus. Further operational detail is not specified in manuscript. |
| Nested NER | A named-entity recognition setting represented in the included corpus. Further operational detail is not specified in manuscript. |
| Prescription NER | A NER subtype represented in the final corpus. |
| Relation extraction or classification | A setting that extracts or classifies relations in medical text. The manuscript groups it within IE and structuring. |
| Acronym disambiguation | A setting represented in the included corpus and IE resource inventory. |

**Inclusion rule.** Assign IE when the principal evaluated output is a structured representation of medical text, including the represented sub-labels above.

**Exclusion rule.** Do not assign IE from the model architecture, retrieval module, graph component, or source material. Do not assign concept normalization, entity linking, or temporal information extraction as final-corpus task annotations: the manuscript discusses them as clinically relevant gaps, not represented annotations. Do not label a task solely because it may support retrieval, a knowledge base, or downstream reasoning.

**Representative examples stated in the manuscript.** The manuscript identifies flat and nested NER, prescription NER, relation extraction/classification, and acronym disambiguation as represented IE settings. It identifies CRF-based Vietnamese medical NER as an example of feature-based/probabilistic structure and retains the CRF layer as an auxiliary component rather than a separate backbone.

### 4.2 Classification, Inference, and Verification

**Operational definition.** This family maps text to semantic or decision labels. It includes intent, sentiment, natural language inference (NLI), and regulatory compliance. For NLI, the evaluated relation is whether a hypothesis is entailed, contradicted, or unrelated to a premise.

**Inclusion rule.** Assign this family when the principal evaluated output is a semantic or decision label, including intent, sentiment, NLI relation, or regulatory-compliance output.

**Disambiguation from Question Answering and IE.**

- Assign QA when the system maps a question, optionally with supporting context, to an answer.
- Assign IE when the system converts medical text into a structured representation.
- Assign Classification, Inference, and Verification when the target is a semantic or decision label. Whether the task is closed-book or evidence-informed is a grounding property, not a task-family distinction.

Further exclusion criteria are **Not specified in manuscript**.

### 4.3 Question Answering

**Operational definition.** QA maps a question, optionally with supporting context, to an answer. QA labels are distinguished by answer structure and evidence requirement.

| Sub-type | Manuscript-grounded rule |
|---|---|
| Answer retrieval / Retrieval QA | Candidate acquisition or ranking forms part of the reported inference pathway. Retriever--ranker systems retrieve candidates and apply ranking. |
| Extractive MRC | The task emphasizes span localization. |
| Conversational MRC | The task adds discourse state. |
| Generative or abstractive QA | The principal evaluated output answers a question and permits free-form answers. |
| MCQA | The output space is constrained. |
| KGQA | Knowledge-graph grounding makes entities, relations, and traversal paths available as structured evidence. |
| Multi-hop QA | The task requires evidence composition; graph-grounded multi-hop settings also require evidence composition. |

**Provided-context MRC versus Retrieval QA.** Assign retrieval QA when candidate acquisition or ranking is part of the reported inference pathway. Assign provided-context MRC when the evaluation supplies a fixed context. Retain both labels only when the study evaluates the two conditions separately. Otherwise, code the reported end-to-end pathway.

**Additional boundary.** Provided-context systems receive a passage, conversation, or reference document directly from the benchmark or experiment and test interpretation of supplied evidence rather than acquisition from a larger collection. Retrieval-mediated systems add candidate acquisition before answer selection or generation.

### 4.4 Natural Language Generation and Text Transformation

**Operational definition.** NLG produces or transforms medical text through summarization, translation, and paraphrasing.

| Sub-type | Manuscript-grounded rule |
|---|---|
| Summarization | Produces transformed medical text. Further operational detail is not specified in manuscript. |
| Machine translation | Produces transformed text. Cross-lingual transfer is recorded separately as a data-construction or adaptation facet. |
| Paraphrasing / text rewriting | Produces or transforms text and is represented in the final corpus as sentence paraphrasing. |

**Generative QA versus NLG/text transformation.** Generative and abstractive QA remain under QA because their principal evaluated output answers a question. Assign NLG/text transformation when the principal output is a summary, translation, paraphrase, or other transformed text rather than an answer to a question.

## 5. Rules for Primary Task, Secondary Task, and Multi-label Assignment

1. Select exactly one primary task family according to the principal evaluated output.
2. Assign a secondary task label when a study evaluates a distinct additional output.
3. Preserve evidence access, cross-lingual transfer, adaptation, and data construction as facets rather than task labels.
4. Count each canonical publication once in publication-level trends, even if it has multiple task labels.
5. Do not create a new task annotation for additional downstream benchmark outputs; they remain benchmark evidence. Do not create an additional corpus publication from a benchmark output.
6. Do not add a task label for a capability discussed only as a gap. Concept normalization, entity linking, and temporal information extraction are examples of clinically relevant capabilities discussed as gaps rather than final-corpus task annotations.

### Worked multi-task examples from the manuscript

| Study as represented in the taxonomy mapping | Task assignment | Basis stated in manuscript |
|---|---|---|
| `tran-tien-etal-2023-vipubmeddeberta` | Text Classification; NER; NLI | The taxonomy mapping records three task labels for one retained canonical publication. |
| `Huy_2021` | IC; NER | The taxonomy mapping records intent classification and NER for one retained canonical publication. |
| `nguyen-etal-2022-vihealthbert` | MTS; AD | The taxonomy mapping records multiple task labels for one retained canonical publication. The full expansions of `MTS` and `AD` are not specified in manuscript in the mapping table. |
| `phan-etal-2023-enriching` | MTS; AD; NLI | The taxonomy mapping records three task labels for one retained canonical publication. The full expansions of `MTS` and `AD` are not specified in manuscript in the mapping table. |

## 6. Methodological-Configuration Taxonomy

Methodological labels describe how a system represents inputs, connects computational stages, specializes model behavior, and accesses information at inference. They are recorded independently. Dataset provenance, construction, and size are analyzed separately. The categories are descriptive and do not establish a ranking, quality claim, causal effect, or hard constraint.

### 6.1 Computational Backbone

The backbone is the principal structure that represents input and produces a prediction or sequence. Record the dominant representational regime rather than every operation in a system.

| Backbone | Operational rule |
|---|---|
| Feature-based/probabilistic | Relies on manually specified lexical, orthographic, cluster, or embedding features, often with probabilistic structured prediction. |
| Recurrent/convolutional | Includes RNNs, LSTMs, CNNs, and BiLSTM--CRF combinations that provide sequential or local feature representations. |
| Encoder-only | Produces bidirectional contextual representations and is typically used for token-, sentence-, or span-level prediction. |
| Encoder--decoder | Transforms an input sequence into an output sequence and is used for summarization, translation, and abstractive QA. |
| Decoder-only | Generates autoregressively and supports compact language models and larger instruction-following models in medical QA and generation. |

A CRF head remains an auxiliary prediction component; it does not create a separate backbone. A decoder receiving retrieved passages remains decoder-only and receives separate composition and grounding labels. Model size is an attribute, not a separate backbone category.

### 6.2 System Composition

System composition describes how functional components are connected independently of backbone and evidence source.

| Composition | Operational rule |
|---|---|
| Single-model / monolithic pipeline | One principal predictive or generative model without an external retrieval, graph, ranking, or agent-coordination stage. |
| Retriever--ranker | Retrieves candidates and applies a ranking component. |
| Retriever--generator | Supplies retrieved passages to a generative model. |
| Graph-enhanced composition | Includes explicit graph processing in the computational pipeline. |
| Multi-component / multi-agent orchestration | Divides query interpretation, retrieval, path selection, verification, or generation among specialized modules or agents. |

Record retrieval type under grounding. Record the ordering and connection of retrieval and generation under composition. Graph processing is distinct from graph-based evidence access because a graph may be used for computation, grounding, or both. Modularity does not establish independent validation of each stage.

### 6.3 Adaptation Strategy

Adaptation describes how a model is specialized or conditioned for a target domain and output format. Parameter scope, training signal, and training schedule are distinct and may co-occur.

| Strategy | Operational rule |
|---|---|
| Task-specific supervised training | Optimizes a model with labeled examples for a target task. |
| Full fine-tuning | Updates all model parameters. |
| Continued pre-training / domain adaptation | Adapts a checkpoint using unlabeled domain text before downstream training. |
| Prompting / in-context learning | Instruction prompting conditions a model without changing weights; zero-shot prompting supplies instructions and input, while few-shot in-context learning adds demonstrations. |
| Instruction fine-tuning | Updates parameters using instruction--output examples; it describes the format of the training signal and can coexist with full fine-tuning or PEFT. |
| Parameter-efficient fine-tuning (PEFT) / LoRA | Updates a restricted subset of parameters, such as LoRA adapters. |
| Multi-stage adaptation | Applies successive training settings or objectives. |

Do not treat these strategies as mutually exclusive or as an ordered scale. Context provision is coded under grounding, not prompting.

### 6.4 Knowledge Grounding

Knowledge grounding specifies information accessible when a system produces a prediction or answer, independently of backbone, composition, and adaptation strategy.

| Grounding | Operational rule |
|---|---|
| Parametric-only / closed-book | Relies on information encoded in model parameters and the immediate input. This does not imply that training used no external corpora. |
| Provided-context / open-book | Receives a passage, conversation, or reference document directly from the benchmark or experiment. |
| Lexical, dense, or hybrid retrieval | Adds candidate acquisition before answer selection or generation; the retrieval variants differ in how they match documents, passages, or answers. |
| Document RAG | A document-based retriever--generator system passes selected passages to a generative model. |
| Knowledge-graph grounding | Makes entities, relations, and traversal paths available as structured evidence. |

Hybrid retrieval is not equivalent to RAG: a retriever--ranker pipeline may end in answer selection, whereas RAG conditions generation on selected evidence. Grounding labels describe the source and timing of information access, not the correctness, reliability, provenance, or clinical validity of the output.

## 7. Data-Construction Taxonomy

Data construction is coded through provenance, instance construction and supervision, and quality assurance. Resources may combine native, translated, extracted, synthetic, manually annotated, and expert-validated content. Record construction patterns rather than dataset names, and do not treat any one pathway as intrinsically superior.

### 7.1 Provenance and Source Acquisition

Provenance is characterized by source setting, authority, register, clinical depth, and accessibility of the underlying material.

| Source or pathway | Manuscript-grounded coding description |
|---|---|
| Health news | A source setting represented among surveyed resources. |
| Public medical questions / consultation / dialogue | Source settings represented among surveyed resources. They are recorded separately from annotation and validation. |
| Biomedical literature | A healthcare-related domain included in the review scope; translated biomedical resources are also represented. |
| Clinical and regulatory text | Clinical text and healthcare regulation are included in review scope; regulatory resources follow a distinct structuring pathway. |
| Knowledge graphs | Graph-based resources make entities, relations, and linked evidence central to resource design. |
| Translated sources | Record reported translation-refinement and preservation of task labels or relations where documented. |
| Extracted sources | Record documented source material and transformation procedure. |
| Synthetic or LLM-assisted sources | Record documented generation, filtering, split controls, and human-review coverage. |

Public release supports independent reuse; restricted sources may limit redistribution and re-evaluation. Source type is not a hierarchy of clinical realism or deployment relevance.

### 7.2 Instance Construction and Supervision

The supervision pathway identifies who creates or reviews an instance and at which stage of construction.

| Construction or supervision condition | Manuscript-grounded coding description |
|---|---|
| Manual annotation | Includes schema design and span annotation where reported. |
| Question writing | May be crowd or author question writing where reported. |
| Translation and refinement | Includes translation correction and human refinement where reported. |
| Extraction / task conversion | Creates task-specific instances from existing materials; record whether targets are copied, extractive, rewritten, or independently generated when reported. |
| Synthetic or LLM-assisted generation | May include generation and filtering of questions, answers, distractors, or reasoning traces. |
| Expert review / adjudication | Includes expert refinement, adjudication, and answer review where reported. Independent annotation and post-hoc expert review are separate supervision regimes. |

Record documented source material, transformation or generation procedure, filtering and split controls, and human-review coverage. Do not impose a uniform failure mode on all translated, derived, synthetic, or LLM-assisted resources.

### 7.3 Quality Assurance and Validation

Quality assurance is represented by evidence available for the intended evaluation target, not by a single global quality label.

| Evidence or check | Manuscript-grounded interpretation |
|---|---|
| Agreement statistics | Interpret with the reported annotation unit, adjudication process, reviewer coverage, and independence; not as a stand-alone quality label. |
| Medical-domain expert involvement | Record when clinical or semantic review, expert refinement, or other medical-domain review is reported. It is distinct from annotation consistency. |
| Evidence and path validation | QA and reasoning resources may report answer--evidence alignment; graph and regulation-based benchmarks may report entity, relation, and connected evidence-path validity. |
| Split and leakage documentation | Record source-level partitioning, deduplication, filtering, and split controls when documented. Do not presume leakage or template repetition without reported evidence. |

Quality evidence sets an interpretation boundary for a benchmark score. It does not establish a universal quality, clinical validity, generalization, or deployment-readiness label.

## 8. Contribution Type and Benchmark-Row Rule

### 8.1 Contribution type

Code the reported contribution type as one or more of the following categories: dataset, benchmark, pretrained model, knowledge resource, task-specific method, or deployed/prototype system. The manuscript does not specify whether these categories are mutually exclusive; therefore, do not impose exclusivity.

### 8.2 Benchmark-row rule

A resource qualifies for one benchmark row only when it supports evaluation of a defined task or output under an identifiable protocol. Resources used solely for training, continued pre-training, or inference-time grounding do not create a benchmark row. Distinct evaluated resources reported by the same study retain separate rows. Candidate records without a matched final-corpus decision are not benchmark resources.

The benchmark inventory counts each of 25 resources once according to its primary evaluated output: 7 IE resources, 4 classification/inference/verification resources, 12 QA resources, and 2 generation/text-transformation resources. These resource-level counts are distinct from the 33-publication/39-task corpus. System papers can add publication or task annotations without adding a reusable benchmark row, and one study can contribute two rows when it evaluates two distinct resources.

### 8.3 Multiple resources or benchmark rows

When one study reports multiple distinct evaluated resources, retain separate benchmark rows. When one publication addresses multiple tasks, retain one publication record and multiple task annotations as justified by distinct evaluated outputs. Secondary tasks, evidence conditions, cross-lingual transfer, and construction pathways do not create duplicate benchmark counts.

## 9. Boundary and Edge-Case Handling Rules

| Situation | Coding rule |
|---|---|
| Boundary study | Retain a study with a substantive Vietnamese healthcare text component but a broader or audio-centered primary scope; apply the explicit scope flag and separate it from core-only aggregates where applicable. |
| Preprint versus archival version | Use DOI and normalized-title matching to identify candidate duplicates, manually inspect them, and retain one canonical record. Prefer the peer-reviewed archival version when it and a preprint report the same study. |
| Retrieval QA versus provided-context MRC | Use retrieval QA when candidate acquisition or ranking is part of reported inference. Use provided-context MRC when a fixed context is supplied by evaluation. Retain both only when separately evaluated; otherwise code the reported end-to-end pathway. |
| Graph computation versus knowledge grounding | Record explicit graph processing as graph-enhanced composition. Record graph-based evidence access as knowledge-graph grounding. A graph can be used for computation, grounding, or both. |
| Backbone versus auxiliary head | Code the dominant representational regime. A CRF head is an auxiliary prediction component, not a separate backbone. A decoder with retrieved passages remains decoder-only, with separate composition and grounding labels. |
| Co-occurring adaptation labels | Record parameter scope, training signal, and training schedule independently; continued pre-training, supervised training, instruction fine-tuning, PEFT, and multi-stage adaptation may co-occur. |
| Missing evidence | Use `not reported` for unresolved task cases rather than infer a label. Use `not confirmed` when a fact could not be verified from reviewed materials; it is not evidence of absence. |
| Contradictory evidence | Jointly revisit the full text and reported evaluation protocol until consensus. A more detailed conflict-resolution hierarchy is not specified in manuscript. |

## 10. Quality Control and Reconciliation Protocols

### 10.1 First-pass coding

One author conducts study-level extraction and initial taxonomy coding for the final corpus. Extract the bibliographic data, task coverage, contribution type, approach, data and evidence provenance, annotation and expert validation, evaluation protocol, release/reproducibility artefacts, supporting evidence, and findings/limitations described in Section 3 of this codebook.

### 10.2 Second-reviewer audit

A second reviewer examines taxonomy coding of all included studies against the manuscript's operational rules. The review gives particular attention to primary versus secondary task labels, the rule for creating a benchmark row, retrieval QA versus provided-context MRC, and co-occurring adaptation labels.

### 10.3 Reconciliation

Resolve differences by jointly revisiting the full text and reported evaluation protocol until consensus. The reconciled coding record underlies the taxonomy tables and figures. The manuscript does not specify a disagreement code, a decision form, a timestamp requirement, or a per-record reconciliation-log format.

### 10.4 Inter-annotator agreement

Do not report an inter-rater reliability statistic for the described procedure. The second review verifies primary coding rather than independently duplicate-coding every metadata field or audit variable. The manuscript explicitly states that it therefore does not report IAA.

## 11. Versioning and Provenance

The manuscript states that its finalized internal corpus ledger underlies the inventory and that the reconciled coding record underlies taxonomy tables and figures. It also distinguishes canonical records, task-level annotations, and benchmark resources.

The following versioning details are **Not specified in manuscript**:

- a codebook version-numbering schema;
- a release procedure for taxonomy changes;
- a formal taxonomy-evolution policy across releases;
- file formats, field schemas, or public locations for `taxonomy_mapping.csv`, `included_studies.csv`, `benchmark_audit.csv`, or `reconciliation_log.csv`;
- explicit integration rules between this codebook and those named system logs.

If these named files are created, their content must remain consistent with the manuscript-grounded units and rules in this codebook: one canonical record per study; task-level annotations for distinct evaluated outputs; one benchmark row per qualifying evaluated resource; and evidence anchors for coded values.

## Appendix A. Controlled Vocabulary

The table lists only standardized values explicitly named in the manuscript. Values not shown are **Not specified in manuscript**.

| Field | Valid values or coding rule |
|---|---|
| Primary task family | Information Extraction and Structuring; Classification, Inference, and Verification; Question Answering; Natural Language Generation and Text Transformation. |
| IE sub-label | Flat NER; nested NER; prescription NER; relation extraction; relation classification; acronym disambiguation. |
| Classification/inference/verification sub-label | Intent; sentiment; NLI; regulatory compliance. |
| QA sub-label | Answer retrieval; extractive MRC; conversational MRC; generative QA; abstractive QA; MCQA; KGQA; multi-hop QA; numerical MRC is represented in the taxonomy mapping. |
| NLG/text-transformation sub-label | Summarization; machine translation; paraphrasing; sentence paraphrasing/text rewriting. |
| Backbone | Feature-based/probabilistic; recurrent/convolutional; encoder-only; encoder--decoder; decoder-only. |
| Composition | Single-model pipeline; retriever--ranker; retriever--generator; graph-enhanced composition; multi-component/multi-agent orchestration. |
| Adaptation | Task-specific supervised training; full fine-tuning; continued pre-training/domain adaptation; zero-shot instruction prompting; few-shot in-context learning; instruction fine-tuning; PEFT/LoRA; multi-stage adaptation. |
| Grounding | Closed-book/parametric-only; provided context/open-book; lexical retrieval; dense retrieval; hybrid retrieval; document RAG; knowledge-graph grounding. |
| Data-construction pathway | Native; translated; extracted; synthetic; manual; expert-reviewed; automatic/mixed. |
| Contribution type | Dataset; benchmark; pretrained model; knowledge resource; task-specific method; deployed/prototype system. |
| Publication status | Archival publication; preprint; technical report/thesis. |
| Corpus scope flag | Core; boundary. Boundary is a substantive Vietnamese healthcare text component with broader or audio-centered primary scope. |
| Evidence status | Reported; not reported; not confirmed. Formal values for `not applicable` and `unclear` are not specified in manuscript. |

## Appendix B. Decision Tree

Use the following manuscript-grounded sequence for each retained canonical publication.

```text
1. Does the record satisfy the review scope and full-text eligibility rules?
   ├─ No → do not include in the final corpus.
   └─ Yes → continue.

2. Does a preprint and archival version describe the same study?
   ├─ Yes → retain one canonical record; prefer the archival version.
   └─ No → retain the eligible canonical record.

3. Does the study have a substantive Vietnamese healthcare text component but a
   broader or audio-centered primary scope?
   ├─ Yes → retain with the boundary scope flag.
   └─ No → retain as a core study.

4. What is the principal evaluated output?
   ├─ Structured representation of medical text → Information Extraction and Structuring.
   ├─ Semantic or decision label → Classification, Inference, and Verification.
   ├─ Answer to a question → Question Answering.
   └─ Summary, translation, paraphrase, or transformed text → NLG/Text Transformation.

5. Are distinct additional outputs evaluated?
   ├─ Yes → assign secondary task label(s).
   └─ No → retain only the primary task label.

6. Is a capability only discussed as a gap, or is it an additional downstream
   benchmark output without a distinct evaluated output in the corpus?
   ├─ Yes → do not create a task annotation.
   └─ No → retain the justified task annotation.

7. For QA, is candidate acquisition or ranking part of the reported inference pathway?
   ├─ Yes → code Retrieval QA and the corresponding retrieval grounding.
   └─ No; evaluation supplies a fixed context → code provided-context MRC.

8. Record independently, where reported:
   backbone; composition; adaptation; grounding; data construction; validation;
   contribution type; evaluation protocol; and evidence anchor.

9. Is a resource evaluated on a defined task/output under an identifiable protocol?
   ├─ Yes → create one benchmark row for that resource.
   └─ No; it is used only for training, continued pre-training, or inference-time
      grounding → do not create a benchmark row.

10. Is evidence missing or unresolved?
    ├─ Unresolved task case → record not reported; do not infer.
    └─ Not verifiable from reviewed materials → record not confirmed; do not treat as absent.
```

## Appendix C. Worked Coding Examples

The examples below reproduce only the fields explicitly shown in the manuscript's taxonomy mapping and surrounding taxonomy text. Fields not shown in that mapping remain **Not specified in manuscript**.

### C.1 `tran-tien-etal-2023-vipubmeddeberta`: multi-task canonical publication

| Field | Coded value from manuscript |
|---|---|
| Task labels | Text Classification; NER; NLI. |
| Primary task | Not specified in manuscript. The mapping provides multiple task labels but does not identify which one is primary. |
| Secondary tasks | Not specified in manuscript for the same reason. |
| Backbone | Encoder-only. |
| Composition | Single-model pipeline. |
| Adaptation | Continued pre-training (DAPT); task-specific supervised training. |
| Grounding | Closed-book. |
| Coding implication | The study demonstrates that multiple task labels may be assigned to one canonical publication and that co-occurring adaptation labels are retained independently. |

### C.2 `nguyen2022spbertqa`: retrieval QA

| Field | Coded value from manuscript |
|---|---|
| Task label | Answer Retrieval. |
| Primary task family | Question Answering, because answer retrieval is included in the QA taxonomy. |
| Backbone | Encoder-only. |
| Composition | Retriever--ranker system. |
| Adaptation | Task-specific supervised training (retriever). |
| Grounding | Hybrid retrieval. |
| Coding implication | Candidate acquisition and semantic re-ranking are part of the reported pathway; code retrieval QA rather than provided-context MRC. |

### C.3 `trinh2025vietmedkg`: graph-grounded QA

| Field | Coded value from manuscript |
|---|---|
| Task label | KGQA. |
| Primary task family | Question Answering. |
| Backbone | Decoder-only. |
| Composition | Graph-enhanced composition. |
| Adaptation | Zero-shot instruction prompting. |
| Grounding | Knowledge-graph grounding. |
| Coding implication | Record graph-enhanced computation and knowledge-graph evidence access separately; the manuscript permits both labels because a graph may be used for computation, grounding, or both. |

### C.4 `nguyen2026vihermes`: multi-hop and multi-component configuration

| Field | Coded value from manuscript |
|---|---|
| Task label | MHQA. |
| Primary task family | Question Answering. |
| Backbone | Decoder-only. |
| Composition | Graph-enhanced composition; multi-component; multi-agent orchestration. |
| Adaptation | Zero-shot instruction prompting. |
| Grounding | Knowledge-graph grounding. |
| Coding implication | The mapping demonstrates co-occurring composition labels and separates them from the grounding label. |

### C.5 `le-duc-etal-2025-medical`: boundary study

| Field | Coded value from manuscript |
|---|---|
| Scope flag | Boundary study with a broader or audio-centered primary scope. |
| Task label | NER. |
| Backbone | Encoder-only; encoder--decoder. |
| Composition | Single-model pipeline. |
| Adaptation | Task-specific supervised training. |
| Grounding | Closed-book. |
| Coding implication | Retain the study because it has a substantive Vietnamese healthcare text component, apply the explicit boundary flag, and separate it from core-only aggregates where applicable. |
