# Search Strategy and Study Selection

## Scope and search date

The review targeted primary research in **Vietnamese healthcare text processing**, including clinical text, biomedical literature, health news, medical QA, COVID-19 information, mental health, traditional medicine, and healthcare regulation. The publication-date window was **1 January 2021 through 31 May 2026**, and the searches were run on **31 May 2026**. No publication-language filter was applied.

## Discovery sources and exact query strings

The search used five discovery sources and seven query strings. The three ACL Anthology queries were separate free-text searches. The arXiv query used platform-specific field syntax.

### ACL Anthology

- **Q-ACL-01:** `Vietnamese medical NLP`
- **Q-ACL-02:** `Vietnamese healthcare dataset`
- **Q-ACL-03:** `Vietnamese clinical question answering`

### ACM Digital Library

**Q-ACM-01**

```text
("Vietnamese") AND ("medical" OR "healthcare" OR "clinical" OR "biomedical") AND ("natural language processing" OR NLP OR "information extraction" OR "question answering" OR summarization OR translation OR dataset OR benchmark)
```

### IEEE Xplore

**Q-IEEE-01**

```text
"Vietnamese" AND ("medical" OR "healthcare" OR "clinical" OR "biomedical") AND ("natural language processing" OR NLP OR "information extraction" OR "question answering" OR summarization OR translation OR dataset OR benchmark)
```

### SpringerLink

**Q-SPRINGER-01**

```text
"Vietnamese" AND ("medical" OR "healthcare" OR "clinical" OR "biomedical") AND ("natural language processing" OR NLP OR "information extraction" OR "question answering" OR summarization OR translation OR dataset OR benchmark)
```

### arXiv

**Q-ARXIV-01**

```text
all:"Vietnamese" AND (all:"medical" OR all:"healthcare" OR all:"clinical" OR all:"biomedical") AND (all:"natural language processing" OR all:NLP OR all:"information extraction" OR all:"question answering" OR all:summarization OR all:translation OR all:dataset OR all:benchmark)
```

Publisher pages were consulted after discovery to resolve the canonical archival or preprint version. They were not an additional discovery source. Date and document-type filters were applied on the search platforms before export. Archival publications were retained where a platform supported those filters; arXiv preprints remained candidates under the same eligibility and canonical-version rules.

## Eligibility and canonical record rules

| Decision | Rule |
| --- | --- |
| Include | Vietnamese is a target language, and the study introduces or evaluates a healthcare-related text-processing dataset, benchmark, pretrained model, knowledge resource, method, or end-to-end system. |
| Exclude | Outside healthcare; does not target Vietnamese; background or survey-only; or lacks a clear healthcare-related text-processing contribution. |
| Ambiguous metadata | Retain for full-text assessment when bibliographic information does not settle eligibility. Resolve ambiguous full-text cases by reviewer consensus with a recorded rationale. |
| Boundary case | Retain a substantive Vietnamese healthcare text component in a study with a broader or audio-centered primary scope, but flag it separately from core text-only studies. The two flagged studies are *Medical Spoken Named Entity Recognition* and *Sentiment Reasoning for Healthcare*. |
| Duplicate / version | Normalize bibliographic metadata; identify candidate duplicates using DOI and normalized-title matching; manually inspect candidates; retain one canonical record per study. Prefer the peer-reviewed archival publication over its preprint when both describe the same study. |

## Study-selection flow

| Stage | Count | Interpretation |
| --- | ---: | --- |
| Initial search returns across five sources | 39,155 | Search results before source-side date/document-type filtering. |
| Exported records after source-side filters | 1,753 | Records available for metadata normalization and duplicate detection. |
| Duplicate records identified | 931 | Candidate matches were manually checked; one canonical record was retained per study. |
| Title, snippet, and metadata screening | 822 | Records remaining after deduplication. |
| Excluded at that screening stage | 737 | Records not advanced to full-text assessment. |
| Full-text eligibility assessment | 85 | Potentially eligible records assessed. |
| Excluded at full text | 52 | Records excluded following full-text assessment. |
| Eligible seed publications | 33 | Canonical primary publications used to seed backward and forward citation chasing. |
| Additional unique eligible publications from citation chasing | 0 | No further eligible publication was identified. |
| Final corpus | **33 publications; 39 task annotations** | Includes two flagged boundary studies; 39 is a multi-label task count. |

The stage totals reconcile: 1,753 − 931 = 822; 822 − 737 = 85; 85 − 52 = 33. Counts through full-text assessment refer to **records**. The final corpus counts **canonical publications** and separately **task annotations**.

## Extraction, checking, and synthesis

For each included publication, the review recorded bibliographic data; primary and secondary tasks; contribution type; algorithms; data and evidence provenance; annotation and expert validation; evaluation protocol; release and reproducibility artefacts; supporting page, section, table, or resource links; and findings and limitations. One reviewer performed study-level extraction and initial taxonomy coding. A second reviewer checked taxonomy assignments for all included studies, focusing on primary and secondary task labels, benchmark-row decisions, retrieval-based question answering versus machine reading comprehension with a provided context, and co-occurring adaptation labels. Differences were resolved by jointly revisiting the full text and evaluation protocol. This was a verification check rather than independent duplicate extraction; inter-rater reliability was not reported.

The descriptive synthesis counts publications once and permits multiple task annotations. It compares the full corpus, the core-only corpus excluding the two boundary studies, the peer-reviewed-only corpus retaining archival peer-reviewed publications, and the strict-core corpus applying both restrictions. It does not pool heterogeneous performance scores or infer a cross-paper model ranking.