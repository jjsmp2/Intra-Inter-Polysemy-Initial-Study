# -*- coding: utf-8 -*-
"""
domain_glossary.py
-------------------
Domain-specific resources and lexicographer file mappings used by the framework's
"Domain Identification" module (Section 3.4 of the project report / Figure 3.1, stage 4).

Fixes & enhancements for Revision Item 5:
- Integrates WordNet Lexicographer files (synset.lexname()) to ground domain
  identification in standard lexical semantics rather than bare keyword overlap.
- Combines lexicographer priors, domain descriptor embeddings (TF-IDF/SBERT),
  and domain cue vocabularies.
- Provides a dedicated domain evaluation benchmark to measure domain-mapping accuracy
  and explicitly account for historical Inter-to-Intra misclassifications (e.g. "port", "service").
"""

from typing import Dict, List, Tuple, Optional
from nltk.corpus import wordnet as wn

# --------------------------------------------------------------------------
# 1. Target Application Domains & Representative Vocabulary Descriptors
# --------------------------------------------------------------------------
DOMAIN_DESCRIPTORS = {
    "Software/IT": (
        "system software application database server client interface "
        "program code module function process file record field key "
        "table index session login password network protocol port "
        "screen user interface data storage memory cache log error "
        "exception configuration parameter query input output display "
        "browser website application programming interface api architecture "
        "endpoint socket component algorithm service thread deployment"
    ),
    "Healthcare": (
        "patient physician nurse clinician doctor diagnosis medication "
        "prescription treatment allergy immunization vaccine provider "
        "clinic hospital chart record encounter visit vital sign lab "
        "laboratory result problem list care plan discharge admission "
        "medical history symptom disease condition dosage drug therapy "
        "clinical biometric pathology triage electronic health ehr emr"
    ),
    "Networking": (
        "network port protocol packet router switch firewall bandwidth "
        "connection socket ethernet ip address gateway node latency "
        "transmission wireless server client host traffic subnet broadcast "
        "routing interface packet tcp udp wifi communication channel"
    ),
    "Legal/Regulatory": (
        "policy regulation compliance law statute rule authorization "
        "consent privacy confidentiality hipaa audit disclosure "
        "requirement standard certification obligation liability "
        "governance security rule breach notification penalty contractual legal"
    ),
    "Finance/Business": (
        "payment billing invoice account balance transaction cost fee "
        "insurance claim reimbursement charge revenue budget contract "
        "vendor procurement price rate financial statement fiscal ledger "
        "order purchase asset liability currency payroll audit"
    ),
    "General": (
        "information process activity item entity object event action "
        "operation task step procedure method report document concept "
        "structure physical location vessel person group unit relation"
    ),
}

# --------------------------------------------------------------------------
# 2. WordNet Lexicographer Category (lexname) Priors
# --------------------------------------------------------------------------
# WordNet partitions synsets into 45 lexicographer files. We map these
# semantic categories to baseline application domain likelihoods:
LEXNAME_TO_DOMAIN_PRIORS: Dict[str, Dict[str, float]] = {
    "noun.body": {"Healthcare": 0.85, "General": 0.15},
    "noun.state": {"Healthcare": 0.50, "General": 0.35, "Software/IT": 0.15},
    "noun.substance": {"Healthcare": 0.60, "General": 0.40},
    "noun.artifact": {"Software/IT": 0.45, "Networking": 0.25, "General": 0.30},
    "noun.communication": {"Software/IT": 0.40, "Legal/Regulatory": 0.25, "General": 0.35},
    "noun.possession": {"Finance/Business": 0.80, "Legal/Regulatory": 0.10, "General": 0.10},
    "noun.act": {"Software/IT": 0.35, "Healthcare": 0.25, "General": 0.40},
    "noun.cognition": {"Software/IT": 0.35, "General": 0.65},
    "noun.group": {"Finance/Business": 0.35, "Legal/Regulatory": 0.25, "General": 0.40},
    "noun.location": {"General": 0.70, "Networking": 0.20, "Software/IT": 0.10},
    "noun.time": {"General": 0.80, "Software/IT": 0.20},
    "noun.person": {"Healthcare": 0.45, "General": 0.55},
    "noun.attribute": {"Software/IT": 0.35, "Legal/Regulatory": 0.25, "General": 0.40},
}

# --------------------------------------------------------------------------
# 3. Curated Candidate Polysemous Terms
# --------------------------------------------------------------------------
# Known cross-domain terms from RE literature & PURE corpus inspection
CANDIDATE_GLOSSARY_TERMS = {
    "record", "chart", "note", "session", "key", "field", "plan", "order",
    "code", "history", "discharge", "provider", "claim", "policy", "port",
    "table", "log", "exchange", "transaction", "profile",
    "summary", "report", "flag", "status", "role", "account", "identifier",
    "authorization", "reference", "template", "form", "review", "alert",
    "message", "queue", "encounter", "solution", "service", "system", "client"
}

# --------------------------------------------------------------------------
# 4. Domain Mapping Benchmark & Error Explanation (Revision Item 5)
# --------------------------------------------------------------------------
# Benchmark gold standard senses for evaluating domain identification accuracy,
# specifically including the four terms ("port", "service", "solution", "system")
# whose Inter-to-Intra classification historically caused errors:
DOMAIN_MAPPING_BENCHMARK = [
    # "port" - historically conflated when network port was mapped as General
    {"synset": "port.n.01", "word": "port", "gold_domain": "General", "definition": "a place on a waterway with facilities for loading and unloading ships"},
    {"synset": "port.n.02", "word": "port", "gold_domain": "General", "definition": "sweet dark-red dessert wine"},
    {"synset": "port.n.03", "word": "port", "gold_domain": "Software/IT", "definition": "an interface on a computer to which a device can be connected"},
    # "service" - web service (Software/IT) vs assistance/consulting (General/Business)
    {"synset": "service.n.01", "word": "service", "gold_domain": "General", "definition": "work done by one person or group that benefits another"},
    {"synset": "service.n.04", "word": "service", "gold_domain": "Software/IT", "definition": "a service program operating under an operating system"},
    {"synset": "service.n.07", "word": "service", "gold_domain": "Legal/Regulatory", "definition": "the act of delivering a formal notice or writ"},
    # "record" - medical record vs database record
    {"synset": "record.n.01", "word": "record", "gold_domain": "Software/IT", "definition": "a document or data structure containing factual information"},
    {"synset": "record.n.05", "word": "record", "gold_domain": "General", "definition": "the extreme ranking attained in sports or achievements"},
    # "chart" - medical chart vs statistical chart
    {"synset": "chart.n.01", "word": "chart", "gold_domain": "General", "definition": "a visual display of information like a graph or table"},
    {"synset": "chart.n.02", "word": "chart", "gold_domain": "Healthcare", "definition": "a medical record documenting a patient's clinical history and care"},
    # "order" - medication order vs software sorting order
    {"synset": "order.n.01", "word": "order", "gold_domain": "Software/IT", "definition": "logical arrangement or sequence of items"},
    {"synset": "order.n.02", "word": "order", "gold_domain": "Finance/Business", "definition": "a commercial document requesting supply of goods or services"},
    {"synset": "order.n.08", "word": "order", "gold_domain": "Healthcare", "definition": "a physician's formal instruction for a medication or diagnostic test"},
    # "solution" - software solution vs chemical mixture vs mathematical answer
    {"synset": "solution.n.01", "word": "solution", "gold_domain": "General", "definition": "a method or process of solving a problem"},
    {"synset": "solution.n.02", "word": "solution", "gold_domain": "General", "definition": "a homogeneous mixture of two or more substances in chemistry"},
    {"synset": "solution.n.03", "word": "solution", "gold_domain": "Software/IT", "definition": "a packaged software application or system solving business needs"},
    # "system" - computer system vs physiological bodily system
    {"synset": "system.n.01", "word": "system", "gold_domain": "Software/IT", "definition": "a group of independent computer hardware and software components"},
    {"synset": "system.n.02", "word": "system", "gold_domain": "Healthcare", "definition": "a group of physiologically or anatomically related organs"},
    # "key" - cryptographic/primary key vs physical lock key
    {"synset": "key.n.01", "word": "key", "gold_domain": "General", "definition": "metal implement used to open a physical lock"},
    {"synset": "key.n.02", "word": "key", "gold_domain": "Software/IT", "definition": "a database column or code that serves as an identifier or decryptor"},
    # "table" - database table vs furniture table
    {"synset": "table.n.01", "word": "table", "gold_domain": "General", "definition": "a piece of furniture having a smooth flat top supported by legs"},
    {"synset": "table.n.02", "word": "table", "gold_domain": "Software/IT", "definition": "an organized set of data elements arranged in rows and columns"},
    # "provider" - healthcare provider vs generic service provider
    {"synset": "provider.n.01", "word": "provider", "gold_domain": "General", "definition": "someone who provides or supplies a resource"},
    {"synset": "provider.n.02", "word": "provider", "gold_domain": "Healthcare", "definition": "a licensed healthcare professional or medical institution delivering care"},
    # "claim" - insurance claim vs assertion
    {"synset": "claim.n.01", "word": "claim", "gold_domain": "General", "definition": "an assertion that something is true or a factual statement"},
    {"synset": "claim.n.02", "word": "claim", "gold_domain": "Finance/Business", "definition": "a formal request to an insurance company for payment or reimbursement"},
    # "policy" - access policy/security rule vs insurance policy vs corporate guideline
    {"synset": "policy.n.01", "word": "policy", "gold_domain": "Legal/Regulatory", "definition": "a plan of action adopted by a government or organization"},
    {"synset": "policy.n.02", "word": "policy", "gold_domain": "Finance/Business", "definition": "written insurance contract outlining terms of coverage"},
    {"synset": "policy.n.03", "word": "policy", "gold_domain": "Software/IT", "definition": "a programmatic rule governing security access and authorization permissions"},
]
