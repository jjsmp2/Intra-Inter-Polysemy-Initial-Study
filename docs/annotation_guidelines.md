# Annotation Guidelines: Inter- and Intra-Domain Ambiguity in Software Requirements

**Version**: 2.0  
**Target Reader Community**: Requirements Engineers, Software Architects, and Cross-Functional Domain Stakeholders (Clinicians, Financial Analysts, DevOps Engineers).

---

## 1. Purpose & Motivation
Software requirement specifications (SRS) and user stories frequently employ natural language terminology that possesses multiple distinct senses across different application domains or within the same technical sub-discipline. When multiple stakeholders interpret a requirement with differing mental models of a term, costly specification defects and implementation mismatches occur.

These guidelines establish unambiguous operational definitions and a decision protocol for classifying target words in requirement sentences into one of three categories:
1. **Unambiguous (`None`)**
2. **Inter-Domain Ambiguity (`Inter-Domain`)**
3. **Intra-Domain Ambiguity (`Intra-Domain`)**

---

## 2. Core Definitions

### 2.1 Ambiguity in Context
A word $w$ occurring in a requirement sentence $s$ is **ambiguous in context** if:
- Given the requirement text and surrounding operational domain, $w$ admits more than one plausible semantic interpretation by members of the reader community; **OR**
- The context fails to supply sufficient qualification or disambiguating cues to rule out a competing, plausible interpretation.

If the sentence provides sufficient restrictive modifiers or domain context such that only one single sense is plausible to any reasonable reader, the word is labeled **`None`** (Unambiguous).

---

### 2.2 Inter-Domain Ambiguity (`Inter-Domain`)
A word exhibits **Inter-Domain Ambiguity** when its competing plausible interpretations originate from **distinct, high-level application domains** (e.g., Software/IT vs. Healthcare, or Software/IT vs. Finance, or Networking vs. General maritime/physical).

#### Canonical Criteria:
1. One plausible sense belongs to the core software/computational domain (e.g., data structures, protocols, operating system mechanisms).
2. Another competing plausible sense belongs to the client application domain (e.g., clinical practice, fiscal governance, medical diagnostics, physical shipping).
3. Misunderstanding leads to **cross-discipline misunderstanding** (e.g., developers building a database schema while clinicians expect a physical clinical binder).

#### Examples:
- **"record"** in *"The system shall store the patient record."*
  - Sense A (*Software/IT*): A discrete database row or tuple in a relational database.
  - Sense B (*Healthcare*): The comprehensive legal medical chart spanning encounters, allergies, and history.
  - **Verdict**: `Inter-Domain` (Software/IT vs. Healthcare).
- **"chart"** in *"The clinician shall review the patient chart."*
  - Sense A (*General/Visual*): A statistical visualization or graphical plot.
  - Sense B (*Healthcare*): A patient's legal electronic health record document.
  - **Verdict**: `Inter-Domain` (General vs. Healthcare).
- **"port"** in *"The system shall connect to the external port."*
  - Sense A (*Software/Networking*): A logical TCP/UDP socket or serial communication interface.
  - Sense B (*General/Physical*): A marine harbor or physical transit facility.
  - **Verdict**: `Inter-Domain` (Networking vs. General).

---

### 2.3 Intra-Domain Ambiguity (`Intra-Domain`)
A word exhibits **Intra-Domain Ambiguity** when its competing plausible interpretations both belong to the **same application domain**, yet denote distinct concepts, entities, or behavioral scopes within that domain.

#### Canonical Criteria:
1. Both competing senses reside within the same domain (e.g., both within Software/IT, or both within Finance).
2. The senses differ in granularity, lifecycle state, architectural layer, or operational mechanism.
3. Misunderstanding leads to **subtle implementation defects** (e.g., using a primary key instead of an encryption key, or treating a user login session as an HTTP socket session).

#### Examples:
- **"key"** in *"The system shall validate each entity using a key."*
  - Sense A (*Software/Database*): A unique relational primary identifier.
  - Sense B (*Software/Security*): A cryptographic decryption token.
  - **Verdict**: `Intra-Domain` (Both Software/IT).
- **"session"** in *"The system shall terminate the session after 15 minutes of inactivity."*
  - Sense A (*Software/Auth*): The authenticated user credential login state.
  - Sense B (*Software/Network*): The underlying TCP/HTTP transport session connection.
  - **Verdict**: `Intra-Domain` (Both Software/IT).
- **"service"** in *"The system shall invoke the patient lookup service."*
  - Sense A (*Software/Architecture*): A remote microservice / REST API endpoint.
  - Sense B (*Software/OS*): A long-running background daemon process on the host operating system.
  - **Verdict**: `Intra-Domain` (Both Software/IT).

---

### 2.4 Unambiguous / Control (`None`)
A word is classified as **`None`** if:
- It is monosemous or functionally monosemous within the discourse (e.g., *patient*, *physician*, *password*, *encryption*).
- Even if polysemous in a general dictionary, the sentence context unambiguously constrains the interpretation to a single sense (e.g., *"The user shall enter a numeric password."*).

---

## 3. Step-by-Step Annotation Decision Protocol

For each (target word $w$, requirement sentence $s$):

1. **Candidate Identification**:
   - Is $w$ an open-class content word (Noun/Verb) carrying potential polysemy?
   - If $w$ is purely functional or unambiguous in this context $\rightarrow$ **`None`**.

2. **Competing Sense Assessment**:
   - In sentence $s$, could a competent reader reasonably invoke more than one distinct meaning of $w$?
   - If only one meaning is contextually plausible $\rightarrow$ **`None`**.

3. **Domain Boundary Test**:
   - Identify Domain $D_1$ of the primary contextually supported sense $S_1$.
   - Identify Domain $D_2$ of the competing plausible sense $S_2$.
   - Does $D_1 \neq D_2$?
     - **YES** $\rightarrow$ **`Inter-Domain`**.
     - **NO** ($D_1 = D_2$) $\rightarrow$ **`Intra-Domain`**.

---

## 4. Multi-Annotator Calibration & Adjudication Protocol
- Each item is independently evaluated by at least two annotators (Annotator 1: Requirements Analyst, Annotator 2: Software Engineer; clinical items reviewed with Domain Specialist Annotator 3).
- Inter-annotator agreement is quantified via **Cohen's Kappa ($\kappa$)** for pairwise agreement and **Fleiss' Kappa ($\kappa_{\text{Fleiss}}$)** for multi-rater consistency.
- Disagreements are adjudicated via consensus review; if consensus is tied, the conservative non-ambiguous label or higher-granularity label is adopted.
