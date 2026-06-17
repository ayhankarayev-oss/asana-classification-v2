# TOS Asana Task Cluster Summary Report

| | |
|---|---|
| **Report date** | 2026-05-28 |
| **Pipeline** | BERTopic v0.17 - ZeroShot mode - all-MiniLM-L6-v2 embeddings |
| **Corpus** | 2,678 tasks from Client_support.csv |
| **Topics found** | 15 clusters + outlier bin |
| **Coverage** | 84.2% classified / 15.8% outliers |

---

## Executive Summary

The BERTopic model identified 15 operationally distinct clusters from a corpus of 2,678 Asana tasks. The three highest-volume categories -- **New Private Investment Entry** (18.9%), **Addepar Account Updates** (11.7%), and **Private Investment Updates & Valuations** (9.5%) -- together account for 40% of all task volume. The model assigned 84.2% of tasks to a named cluster, leaving only 15.8% as outliers requiring manual review. Cluster profiles below are sorted by volume (largest first) so the highest-priority operational areas appear at the top.

---

## Cluster Profiles

Each profile shows: (1) the mathematical keyword basis for the cluster via c-TF-IDF, (2) the three tasks geometrically closest to the cluster centroid in embedding space, and (3) an operational interpretation of what the data tells us about real work happening inside that bucket.

---

## Topic 2 -- New Private Investment Entry

*Cluster ID: 2 | 505 tasks | 18.9% of total corpus | 22.4% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | private investment | 0.0578 | High -- clearly characteristic |
| 2 | direct owner | 0.0369 | Moderate -- shares vocabulary with adjacent topics |
| 3 | commitment | 0.0358 | Moderate -- shares vocabulary with adjacent topics |
| 4 | asset class | 0.0271 | Moderate -- shares vocabulary with adjacent topics |
| 5 | alternative investments | 0.0268 | Moderate -- shares vocabulary with adjacent topics |
| 6 | contact number | 0.0264 | Moderate -- shares vocabulary with adjacent topics |
| 7 | jsde | 0.0258 | Moderate -- shares vocabulary with adjacent topics |
| 8 | holdings | 0.0255 | Moderate -- shares vocabulary with adjacent topics |
| 9 | inquiries | 0.0215 | Moderate -- shares vocabulary with adjacent topics |
| 10 | private investments | 0.0191 | Low -- diffuse, overlaps many topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. Benjamin Athens, New private investment
2. Please add the new private investment to Addepar as outlined below by Tejal
3. Please add the new private investment for Camp as outlined by Carolina below

### Operational Summary

This cluster is one of the largest operational areas in the corpus covering 505 tasks (18.9% of the full corpus). The leading keyword **private investment** (c-TF-IDF: 0.0578) is clearly characteristic of this cluster, reflecting consistent vocabulary and a recognisable operational theme across the tasks grouped here. Representative tasks include "Benjamin Athens, New private investment" and "Please add the new private investment to Addepar as outlined below by Tejal". At 18.9% of total task volume, this is one of the highest-demand categories in the portfolio and is a strong candidate for workflow automation, task templates, or dedicated SLA tracking to manage recurring throughput at scale.

---

## Topic 1 -- Portfolio Platform Account Updates

*Cluster ID: 1 | 314 tasks | 11.7% of total corpus | 13.9% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | accounts | 0.0383 | Moderate -- shares vocabulary with adjacent topics |
| 2 | support requests | 0.0325 | Moderate -- shares vocabulary with adjacent topics |
| 3 | dfo support | 0.0325 | Moderate -- shares vocabulary with adjacent topics |
| 4 | details missing | 0.0247 | Moderate -- shares vocabulary with adjacent topics |
| 5 | asset class | 0.0242 | Moderate -- shares vocabulary with adjacent topics |
| 6 | issue missing | 0.0211 | Moderate -- shares vocabulary with adjacent topics |
| 7 | inquiries | 0.0202 | Moderate -- shares vocabulary with adjacent topics |
| 8 | jsde | 0.0196 | Low -- diffuse, overlaps many topics |
| 9 | data optional | 0.0172 | Low -- diffuse, overlaps many topics |
| 10 | data missing | 0.0170 | Low -- diffuse, overlaps many topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. Addepar updates
2. [Duplicate] Addepar updates - See updated request "Real estate deals"
3. Rodrigo Besoy - Addepar updates

### Operational Summary

This cluster is a major operational area covering 314 tasks (11.7% of the full corpus). The top keyword **accounts** (c-TF-IDF: 0.0383) provides moderate differentiation. The cluster has a recognisable theme but shares vocabulary with neighbouring categories -- expected behaviour in a narrow, domain-specific operations corpus where most tasks reference the same core systems (Addepar, Arch, Egnyte). Representative tasks include "Addepar updates" and "[Duplicate] Addepar updates - See updated request "Real estate deals"". At 11.7% of total volume, this represents a significant and recurring stream of work that warrants standardised procedures, clear ownership, and defined turnaround expectations.

---

## Topic 3 -- Private Investment Updates & Valuations

*Cluster ID: 3 | 254 tasks | 9.5% of total corpus | 11.3% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | series | 0.0979 | High -- clearly characteristic |
| 2 | unfunded | 0.0462 | Moderate -- shares vocabulary with adjacent topics |
| 3 | lp | 0.0449 | Moderate -- shares vocabulary with adjacent topics |
| 4 | capital | 0.0414 | Moderate -- shares vocabulary with adjacent topics |
| 5 | trousdale sarosphere | 0.0386 | Moderate -- shares vocabulary with adjacent topics |
| 6 | valuations | 0.0288 | Moderate -- shares vocabulary with adjacent topics |
| 7 | equity | 0.0261 | Moderate -- shares vocabulary with adjacent topics |
| 8 | excel | 0.0234 | Moderate -- shares vocabulary with adjacent topics |
| 9 | llc llc | 0.0208 | Moderate -- shares vocabulary with adjacent topics |
| 10 | loan | 0.0200 | Low -- diffuse, overlaps many topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. Direct Deal View Updates
2. Ellipse World, Inc. – Series B-2 Down Round / NAV Update
3. Obtain updated data to pass along to Elevate to update private investments in Addepar

### Operational Summary

This cluster is a major operational area covering 254 tasks (9.5% of the full corpus). The leading keyword **series** (c-TF-IDF: 0.0979) is clearly characteristic of this cluster, reflecting consistent vocabulary and a recognisable operational theme across the tasks grouped here. Representative tasks include "Direct Deal View Updates" and "Ellipse World, Inc. – Series B-2 Down Round / NAV Update". At 9.5% of total volume, this represents a significant and recurring stream of work that warrants standardised procedures, clear ownership, and defined turnaround expectations.

---

## Topic 13 -- Portfolio Platform View & Access Configuration

*Cluster ID: 13 | 193 tasks | 7.2% of total corpus | 8.6% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | access | 0.1136 | Very high -- strongly unique to this cluster |
| 2 | views | 0.0503 | High -- clearly characteristic |
| 3 | columns | 0.0312 | Moderate -- shares vocabulary with adjacent topics |
| 4 | household | 0.0270 | Moderate -- shares vocabulary with adjacent topics |
| 5 | portfolio | 0.0243 | Moderate -- shares vocabulary with adjacent topics |
| 6 | accounts | 0.0227 | Moderate -- shares vocabulary with adjacent topics |
| 7 | ownership structure | 0.0222 | Moderate -- shares vocabulary with adjacent topics |
| 8 | owned | 0.0217 | Moderate -- shares vocabulary with adjacent topics |
| 9 | filter | 0.0205 | Moderate -- shares vocabulary with adjacent topics |
| 10 | owner id | 0.0197 | Low -- diffuse, overlaps many topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. Please create Addepar views for Bellco
2. View Access AWB
3. View Access

### Operational Summary

This cluster is a major operational area covering 193 tasks (7.2% of the full corpus). The dominant keyword **access** carries an exceptionally high c-TF-IDF score of 0.1136, confirming this is a tightly cohesive, well-defined operational category with strongly consistent vocabulary across every grouped task. Representative tasks include "Please create Addepar views for Bellco" and "View Access AWB". At 7.2% of total volume, this represents a significant and recurring stream of work that warrants standardised procedures, clear ownership, and defined turnaround expectations.

---

## Topic 0 -- New Account & Data Feed Setup

*Cluster ID: 0 | 190 tasks | 7.1% of total corpus | 8.4% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | accounts | 0.0969 | High -- clearly characteristic |
| 2 | investment account | 0.0444 | Moderate -- shares vocabulary with adjacent topics |
| 3 | contact information | 0.0391 | Moderate -- shares vocabulary with adjacent topics |
| 4 | data feed | 0.0341 | Moderate -- shares vocabulary with adjacent topics |
| 5 | connection | 0.0297 | Moderate -- shares vocabulary with adjacent topics |
| 6 | new accounts | 0.0286 | Moderate -- shares vocabulary with adjacent topics |
| 7 | online account | 0.0242 | Moderate -- shares vocabulary with adjacent topics |
| 8 | inquiries new | 0.0228 | Moderate -- shares vocabulary with adjacent topics |
| 9 | balance | 0.0215 | Moderate -- shares vocabulary with adjacent topics |
| 10 | backfill | 0.0207 | Moderate -- shares vocabulary with adjacent topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. Please add the new Schwab account for Thompson, as outlined by Carolina below
2. Please create the new Schwab account as outlined by Carolina below for Dunlap
3. Please add the new Schwab account(s) for Carolina, as outlined below in her Asana form submission

### Operational Summary

This cluster is a major operational area covering 190 tasks (7.1% of the full corpus). The leading keyword **accounts** (c-TF-IDF: 0.0969) is clearly characteristic of this cluster, reflecting consistent vocabulary and a recognisable operational theme across the tasks grouped here. Representative tasks include "Please add the new Schwab account for Thompson, as outlined by Carolina below" and "Please create the new Schwab account as outlined by Carolina below for Dunlap". At 7.1% of total volume, this represents a significant and recurring stream of work that warrants standardised procedures, clear ownership, and defined turnaround expectations.

---

## Topic 8 -- Reporting & Performance Analytics

*Cluster ID: 8 | 184 tasks | 6.9% of total corpus | 8.2% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | reports | 0.0569 | High -- clearly characteristic |
| 2 | performance | 0.0511 | High -- clearly characteristic |
| 3 | benchmark | 0.0368 | Moderate -- shares vocabulary with adjacent topics |
| 4 | private capital | 0.0289 | Moderate -- shares vocabulary with adjacent topics |
| 5 | quarterly | 0.0279 | Moderate -- shares vocabulary with adjacent topics |
| 6 | excel | 0.0262 | Moderate -- shares vocabulary with adjacent topics |
| 7 | chart | 0.0253 | Moderate -- shares vocabulary with adjacent topics |
| 8 | views | 0.0228 | Moderate -- shares vocabulary with adjacent topics |
| 9 | analysis | 0.0209 | Moderate -- shares vocabulary with adjacent topics |
| 10 | asset | 0.0194 | Low -- diffuse, overlaps many topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. St. andrews report buys/sells
2. Gary report 04.28.26 followup
3. HMI reports

### Operational Summary

This cluster is a regular operational category covering 184 tasks (6.9% of the full corpus). The leading keyword **reports** (c-TF-IDF: 0.0569) is clearly characteristic of this cluster, reflecting consistent vocabulary and a recognisable operational theme across the tasks grouped here. Representative tasks include "St. andrews report buys/sells" and "Gary report 04.28.26 followup". At 6.9% of total volume, this is a steady category where consistent processes and clear handoff points will improve throughput.

---

## Topic 5 -- Capital Call Audit & Monthly Statement Review

*Cluster ID: 5 | 108 tasks | 4.0% of total corpus | 4.8% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | checking | 0.0543 | High -- clearly characteristic |
| 2 | capital | 0.0533 | High -- clearly characteristic |
| 3 | audit | 0.0516 | High -- clearly characteristic |
| 4 | daily | 0.0421 | Moderate -- shares vocabulary with adjacent topics |
| 5 | pdf | 0.0363 | Moderate -- shares vocabulary with adjacent topics |
| 6 | brokerage | 0.0361 | Moderate -- shares vocabulary with adjacent topics |
| 7 | jsde | 0.0267 | Moderate -- shares vocabulary with adjacent topics |
| 8 | valuation | 0.0238 | Moderate -- shares vocabulary with adjacent topics |
| 9 | asset class | 0.0234 | Moderate -- shares vocabulary with adjacent topics |
| 10 | loan | 0.0228 | Moderate -- shares vocabulary with adjacent topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. TOS Instance Wide Asset class Missing Review
2. PJS Monthly statements for December 2025 - 9002/9039/9040/9089/9090
3. PJS Monthly statements for February 2026 - 9002/9039/9040/9089/9090

### Operational Summary

This cluster is a regular operational category covering 108 tasks (4.0% of the full corpus). The leading keyword **checking** (c-TF-IDF: 0.0543) is clearly characteristic of this cluster, reflecting consistent vocabulary and a recognisable operational theme across the tasks grouped here. Representative tasks include "TOS Instance Wide Asset class Missing Review" and "PJS Monthly statements for December 2025 - 9002/9039/9040/9089/9090". At 4.0% of total volume, this is a steady category where consistent processes and clear handoff points will improve throughput.

---

## Topic 11 -- Loan & Lending Account Setup

*Cluster ID: 11 | 107 tasks | 4.0% of total corpus | 4.7% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | lender | 0.0448 | Moderate -- shares vocabulary with adjacent topics |
| 2 | loan payment | 0.0385 | Moderate -- shares vocabulary with adjacent topics |
| 3 | loans | 0.0374 | Moderate -- shares vocabulary with adjacent topics |
| 4 | attributes | 0.0368 | Moderate -- shares vocabulary with adjacent topics |
| 5 | accounts | 0.0340 | Moderate -- shares vocabulary with adjacent topics |
| 6 | eu | 0.0315 | Moderate -- shares vocabulary with adjacent topics |
| 7 | date loan | 0.0314 | Moderate -- shares vocabulary with adjacent topics |
| 8 | information like | 0.0307 | Moderate -- shares vocabulary with adjacent topics |
| 9 | inquiries | 0.0303 | Moderate -- shares vocabulary with adjacent topics |
| 10 | households | 0.0289 | Moderate -- shares vocabulary with adjacent topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. Ricardo Nazario, New loan
2. Client contact cheat sheet
3. Dorsar Portal view

### Operational Summary

This cluster is a regular operational category covering 107 tasks (4.0% of the full corpus). The top keyword **lender** (c-TF-IDF: 0.0448) provides moderate differentiation. The cluster has a recognisable theme but shares vocabulary with neighbouring categories -- expected behaviour in a narrow, domain-specific operations corpus where most tasks reference the same core systems (Addepar, Arch, Egnyte). Representative tasks include "Ricardo Nazario, New loan" and "Client contact cheat sheet". At 4.0% of total volume, this is a steady category where consistent processes and clear handoff points will improve throughput.

---

## Topic 9 -- Ownership Structure & Legal Entity Changes

*Cluster ID: 9 | 93 tasks | 3.5% of total corpus | 4.1% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | ownership | 0.1093 | Very high -- strongly unique to this cluster |
| 2 | legal entity | 0.1044 | Very high -- strongly unique to this cluster |
| 3 | vijay | 0.0402 | Moderate -- shares vocabulary with adjacent topics |
| 4 | direct owner | 0.0317 | Moderate -- shares vocabulary with adjacent topics |
| 5 | respective | 0.0306 | Moderate -- shares vocabulary with adjacent topics |
| 6 | owns | 0.0263 | Moderate -- shares vocabulary with adjacent topics |
| 7 | merge | 0.0253 | Moderate -- shares vocabulary with adjacent topics |
| 8 | holding account | 0.0245 | Moderate -- shares vocabulary with adjacent topics |
| 9 | directly owned | 0.0213 | Moderate -- shares vocabulary with adjacent topics |
| 10 | transfer | 0.0209 | Moderate -- shares vocabulary with adjacent topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. New legal entity
2. Process two Dissolved Oklahoma Entities for Ken and Dean: B Three Farming, LLC a
3. Please set up the ownership structure and add the missing items for Letsos. **No

### Operational Summary

This cluster is a regular operational category covering 93 tasks (3.5% of the full corpus). The dominant keyword **ownership** carries an exceptionally high c-TF-IDF score of 0.1093, confirming this is a tightly cohesive, well-defined operational category with strongly consistent vocabulary across every grouped task. Representative tasks include "New legal entity" and "Process two Dissolved Oklahoma Entities for Ken and Dean: B Three Farming, LLC a". At 3.5% of total volume, this is a steady category where consistent processes and clear handoff points will improve throughput.

---

## Topic 10 -- Real Asset Transaction Audit

*Cluster ID: 10 | 69 tasks | 2.6% of total corpus | 3.1% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | audit | 0.0950 | High -- clearly characteristic |
| 2 | transaction | 0.0564 | High -- clearly characteristic |
| 3 | excel | 0.0402 | Moderate -- shares vocabulary with adjacent topics |
| 4 | accounts | 0.0384 | Moderate -- shares vocabulary with adjacent topics |
| 5 | monthly | 0.0359 | Moderate -- shares vocabulary with adjacent topics |
| 6 | oil | 0.0353 | Moderate -- shares vocabulary with adjacent topics |
| 7 | prior month | 0.0347 | Moderate -- shares vocabulary with adjacent topics |
| 8 | cash flows | 0.0301 | Moderate -- shares vocabulary with adjacent topics |
| 9 | clients | 0.0275 | Moderate -- shares vocabulary with adjacent topics |
| 10 | real assets | 0.0260 | Moderate -- shares vocabulary with adjacent topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. Missing transaction audit
2. Please reach out to Jenn Musto from Armanino requesting the latest Real Asset entity cash flows in Excel
3. Reach out to Jenn Musto from Armanino requesting the Quarter's Real Asset entity cash flows in Excel

### Operational Summary

This cluster is a focused specialist category covering 69 tasks (2.6% of the full corpus). The leading keyword **audit** (c-TF-IDF: 0.0950) is clearly characteristic of this cluster, reflecting consistent vocabulary and a recognisable operational theme across the tasks grouped here. Representative tasks include "Missing transaction audit" and "Please reach out to Jenn Musto from Armanino requesting the latest Real Asset entity cash". Although smaller in volume (2.6%), this cluster represents specialist or high-touch work that may require dedicated expertise, third-party coordination, or specific regulatory awareness.

---

## Topic 12 -- Direct Deal Updates & Unlinked Accounts

*Cluster ID: 12 | 60 tasks | 2.2% of total corpus | 2.7% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | sale | 0.0723 | High -- clearly characteristic |
| 2 | inquiries | 0.0623 | High -- clearly characteristic |
| 3 | direct deals | 0.0536 | High -- clearly characteristic |
| 4 | green grasshopper | 0.0476 | Moderate -- shares vocabulary with adjacent topics |
| 5 | statements | 0.0449 | Moderate -- shares vocabulary with adjacent topics |
| 6 | unlinked | 0.0419 | Moderate -- shares vocabulary with adjacent topics |
| 7 | high inquiries | 0.0417 | Moderate -- shares vocabulary with adjacent topics |
| 8 | loan | 0.0417 | Moderate -- shares vocabulary with adjacent topics |
| 9 | received | 0.0382 | Moderate -- shares vocabulary with adjacent topics |
| 10 | urgent | 0.0347 | Moderate -- shares vocabulary with adjacent topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. Reach out to Emily for updated statements to update unlinked direct deals that require manual updates
2. Ricardo Nazario, New loan
3. Ricardo Nazario, General

### Operational Summary

This cluster is a focused specialist category covering 60 tasks (2.2% of the full corpus). The leading keyword **sale** (c-TF-IDF: 0.0723) is clearly characteristic of this cluster, reflecting consistent vocabulary and a recognisable operational theme across the tasks grouped here. Representative tasks include "Reach out to Emily for updated statements to update unlinked direct deals that require man" and "Ricardo Nazario, New loan". Although smaller in volume (2.2%), this cluster represents specialist or high-touch work that may require dedicated expertise, third-party coordination, or specific regulatory awareness.

---

## Topic 6 -- Cost Basis & Data Quality Fixes

*Cluster ID: 6 | 59 tasks | 2.2% of total corpus | 2.6% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | cost basis | 0.1995 | Very high -- strongly unique to this cluster |
| 2 | export | 0.0759 | High -- clearly characteristic |
| 3 | reporting | 0.0733 | High -- clearly characteristic |
| 4 | missing cost | 0.0455 | Moderate -- shares vocabulary with adjacent topics |
| 5 | days | 0.0341 | Moderate -- shares vocabulary with adjacent topics |
| 6 | grouping | 0.0320 | Moderate -- shares vocabulary with adjacent topics |
| 7 | missing data | 0.0285 | Moderate -- shares vocabulary with adjacent topics |
| 8 | asset class | 0.0281 | Moderate -- shares vocabulary with adjacent topics |
| 9 | liquidity | 0.0259 | Moderate -- shares vocabulary with adjacent topics |
| 10 | dfi | 0.0243 | Moderate -- shares vocabulary with adjacent topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. Download Goldman cost basis data and ensure asset classes are properly documented
2. Download Goldman cost basis data
3. Please update Goldman cost basis for Dorsar

### Operational Summary

This cluster is a focused specialist category covering 59 tasks (2.2% of the full corpus). The dominant keyword **cost basis** carries an exceptionally high c-TF-IDF score of 0.1995, confirming this is a tightly cohesive, well-defined operational category with strongly consistent vocabulary across every grouped task. Representative tasks include "Download Goldman cost basis data and ensure asset classes are properly documented" and "Download Goldman cost basis data". Although smaller in volume (2.2%), this cluster represents specialist or high-touch work that may require dedicated expertise, third-party coordination, or specific regulatory awareness.

---

## Topic 7 -- Document Upload & Client Billing

*Cluster ID: 7 | 46 tasks | 1.7% of total corpus | 2.0% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | upload | 0.0985 | High -- clearly characteristic |
| 2 | llc llc | 0.0981 | High -- clearly characteristic |
| 3 | prepare | 0.0935 | High -- clearly characteristic |
| 4 | lp | 0.0836 | High -- clearly characteristic |
| 5 | document | 0.0756 | High -- clearly characteristic |
| 6 | llog | 0.0613 | High -- clearly characteristic |
| 7 | geib family | 0.0428 | Moderate -- shares vocabulary with adjacent topics |
| 8 | geib | 0.0396 | Moderate -- shares vocabulary with adjacent topics |
| 9 | family trust | 0.0384 | Moderate -- shares vocabulary with adjacent topics |
| 10 | password | 0.0332 | Moderate -- shares vocabulary with adjacent topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. Prepare and send client invoices
2. Billing discussion with Ian
3. Evaluate SAOS email on data issue for Woodland account at Morgan Stanley

### Operational Summary

This cluster is a focused specialist category covering 46 tasks (1.7% of the full corpus). The leading keyword **upload** (c-TF-IDF: 0.0985) is clearly characteristic of this cluster, reflecting consistent vocabulary and a recognisable operational theme across the tasks grouped here. Representative tasks include "Prepare and send client invoices" and "Billing discussion with Ian". Although smaller in volume (1.7%), this cluster represents specialist or high-touch work that may require dedicated expertise, third-party coordination, or specific regulatory awareness.

---

## Topic 4 -- Trust & Estate Cash Flow Verification

*Cluster ID: 4 | 45 tasks | 1.7% of total corpus | 2.0% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | billing | 0.0887 | High -- clearly characteristic |
| 2 | non-exempt marital | 0.0616 | High -- clearly characteristic |
| 3 | transfers | 0.0604 | High -- clearly characteristic |
| 4 | gst non-exempt | 0.0594 | High -- clearly characteristic |
| 5 | marital trust | 0.0594 | High -- clearly characteristic |
| 6 | schmulen gst | 0.0575 | High -- clearly characteristic |
| 7 | verification | 0.0552 | High -- clearly characteristic |
| 8 | cash flows | 0.0427 | Moderate -- shares vocabulary with adjacent topics |
| 9 | noble mortgage | 0.0420 | Moderate -- shares vocabulary with adjacent topics |
| 10 | dcs | 0.0378 | Moderate -- shares vocabulary with adjacent topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. Please gather a list of cash flows for all private deals for the Schmulens
2. Noble Mortgage - Verify latest transactions have been captured by Arch
3. st. andrews vaughan nelson transactions backfill

### Operational Summary

This cluster is a focused specialist category covering 45 tasks (1.7% of the full corpus). The leading keyword **billing** (c-TF-IDF: 0.0887) is clearly characteristic of this cluster, reflecting consistent vocabulary and a recognisable operational theme across the tasks grouped here. Representative tasks include "Please gather a list of cash flows for all private deals for the Schmulens" and "Noble Mortgage - Verify latest transactions have been captured by Arch". Although smaller in volume (1.7%), this cluster represents specialist or high-touch work that may require dedicated expertise, third-party coordination, or specific regulatory awareness.

---

## Topic 14 -- General & Ad Hoc Requests

*Cluster ID: 14 | 28 tasks | 1.0% of total corpus | 1.2% of classified tasks*

### Defining Keywords (c-TF-IDF)

| Rank | Keyword | Score | Distinctiveness |
|:----:|---------|------:|-----------------|
| 1 | flow | 0.2375 | Very high -- strongly unique to this cluster |
| 2 | inquiries | 0.0771 | High -- clearly characteristic |
| 3 | information like | 0.0586 | High -- clearly characteristic |
| 4 | low inquiries | 0.0565 | High -- clearly characteristic |
| 5 | included description | 0.0483 | Moderate -- shares vocabulary with adjacent topics |
| 6 | medium inquiries | 0.0459 | Moderate -- shares vocabulary with adjacent topics |
| 7 | requests | 0.0453 | Moderate -- shares vocabulary with adjacent topics |
| 8 | manually | 0.0408 | Moderate -- shares vocabulary with adjacent topics |
| 9 | dfo support | 0.0367 | Moderate -- shares vocabulary with adjacent topics |
| 10 | support requests | 0.0367 | Moderate -- shares vocabulary with adjacent topics |

### Representative Tasks

The three tasks geometrically closest to this cluster's centroid in sentence-embedding space (i.e., most archetypal examples of this category):

1. Robert McGill, General updates
2. Ricardo L. Nazario, General updates
3. Please create a workflow to override custodial price per share for Contra Akero Therapeut (Entity ID - 138869742) and Contra Avade

### Operational Summary

This cluster is a focused specialist category covering 28 tasks (1.0% of the full corpus). The dominant keyword **flow** carries an exceptionally high c-TF-IDF score of 0.2375, confirming this is a tightly cohesive, well-defined operational category with strongly consistent vocabulary across every grouped task. Representative tasks include "Robert McGill, General updates" and "Ricardo L. Nazario, General updates". Although smaller in volume (1.0%), this cluster represents specialist or high-touch work that may require dedicated expertise, third-party coordination, or specific regulatory awareness.

---

## Outlier Bin -- Outliers / Unclassified

*Cluster ID: -1 | 423 tasks | 15.8% of total corpus | 0.0% of classified tasks*

### What Are Outliers?

These 423 tasks (15.8% of the corpus) could not be confidently assigned to any cluster. Common causes: task names that are too short or vague to produce a reliable embedding, one-off tasks with highly unique vocabulary, or borderline tasks that scored below the 0.30 cosine-similarity threshold against all 15 zero-shot seeds. These tasks are preserved in **complaints_classified.csv** with `topic_id = -1` and should be reviewed manually or assigned to the nearest business category by a domain expert.

---

*Generated by `model_pipeline.py` using BERTopic v0.17 + ZeroShot topic modelling.*  
*c-TF-IDF (class-based TF-IDF): scores reflect how uniquely a term identifies one cluster*  
*relative to the full corpus. Higher = more diagnostic.*