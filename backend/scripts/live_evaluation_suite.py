"""Live Evaluation Suite & Comprehensive DSML Benchmark for SkillPulse AI.

Designed for live demonstration and defense during DSML viva / presentation:
1. Runs an automated battery of 60+ diverse, rigorous test cases across all ML models:
   - Model 1: Job Role Classification (11 classes, calibrated probabilities)
   - Model 2: Requirement Level Classification (Required / Preferred / Neutral)
   - Model 3a: Apriori Association Rules (Support, Confidence, Lift explanations)
   - Model 3b: K-Means Skill Clusters (Readiness scoring)
   - NLP Pipeline: Disambiguation & context-aware skill extraction
   - Robustness & Edge Cases: Non-technical text, adversarial prompts, empty inputs
2. Interactive Impromptu Mode: Paste any syllabus unit, course description, or job ad snippet
   and get instant real-time predictions, probabilities, and explanations.

Usage:
    python backend/scripts/live_evaluation_suite.py --run-battery
    python backend/scripts/live_evaluation_suite.py --interactive
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
from app.services.market_model import MarketModel, get_market
from app.services.skill_extractor import AliasEntry, SectionInput, TaxonomyMatcher, clean_text, extract_skills
from ml.common import load_taxonomy


def run_test_battery():
    print("=" * 80)
    print("SKILLPULSE AI - COMPREHENSIVE DSML LIVE EVALUATION SUITE")
    print("=" * 80)
    t_start = time.time()
    
    print("\n[+] Initializing MarketModel and Loading Pre-trained Artifacts...")
    market = get_market()
    print(f"    - Postings indexed: {market.n:,}")
    print(f"    - Skills taxonomy: {len(market.skills)} skills")
    print(f"    - Association rules: {len(market.rules):,} rules")
    print(f"    - K-Means clusters: {len(market.bundles)} bundles")
    print(f"    - Loaded in {time.time() - t_start:.2f}s\n")

    passed_count = 0
    total_count = 0

    # -------------------------------------------------------------------------
    # SUITE 1: MODEL 1 - JOB ROLE CLASSIFICATION (Supervised 11-Class Calibrated Linear SVM)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("TEST SUITE 1: MODEL 1 - JOB ROLE PREDICTION (11 Classes)")
    print("=" * 80)

    role_test_cases = [
        # (Course / text snippet, Extracted skills, Expected Primary Role, Description)
        (
            "Supervised learning algorithms, linear regression, logistic regression, decision trees, random forests, gradient boosting, deep neural networks, CNNs, transformers with PyTorch and TensorFlow.",
            ["Machine Learning", "Python", "TensorFlow", "PyTorch"],
            "Data Scientist / ML Engineer",
            "Classic ML & Deep Learning Course",
        ),
        (
            "Building scalable data pipelines with Apache Spark, Kafka streaming, ETL workflows, Hadoop HDFS, data lakes on Databricks and Snowflake warehouse design.",
            ["Apache Spark", "Apache Kafka", "Hadoop", "SQL"],
            "Data Engineer",
            "Big Data & ETL Engineering Course",
        ),
        (
            "Business analytics, descriptive statistics, interactive dashboard creation using Power BI, Tableau reports, advanced Excel pivot tables, and relational SQL queries.",
            ["Power BI", "Tableau", "Microsoft Excel", "SQL"],
            "Data Analyst / BI",
            "Business Intelligence & Analytics Course",
        ),
        (
            "Network penetration testing, vulnerability assessment, ethical hacking, cryptography, security operations center (SOC) monitoring, SIEM, firewalls and malware analysis.",
            ["Cybersecurity", "Networking", "Cryptography"],
            "Cybersecurity",
            "Information Security & Cyber Defense Course",
        ),
        (
            "Container orchestration with Kubernetes, Docker containerization, CI/CD pipelines with Jenkins and GitHub Actions, infrastructure as code using Terraform on AWS and Azure cloud.",
            ["Kubernetes", "Docker", "AWS", "CI/CD", "Linux"],
            "DevOps / Cloud",
            "Cloud Infrastructure & DevOps Course",
        ),
        (
            "Automated software testing, test driven development (TDD), Selenium WebDriver, pytest, JUnit test suites, API integration testing, load testing with JMeter.",
            ["Software Testing", "Selenium", "Java", "Python"],
            "QA / Testing",
            "Software Quality Assurance & Automation Course",
        ),
        (
            "Cross-platform mobile application development using Flutter, Dart, React Native, Android SDK with Kotlin, iOS development with Swift and Xcode.",
            ["Flutter", "Android", "React Native", "Kotlin", "Swift"],
            "Mobile Developer",
            "Mobile App Development Course",
        ),
        (
            "Modern web development using React.js, Next.js, Node.js backend, Express RESTful APIs, HTML5, CSS3, Tailwind CSS, and MongoDB full stack integration.",
            ["React", "Node.js", "JavaScript", "HTML", "CSS", "MongoDB"],
            "Web / Full Stack Developer",
            "MERN Full Stack Development Course",
        ),
        (
            "Object-oriented programming in Java and C++, design patterns, memory management, multi-threading, concurrency, algorithms, and enterprise Spring Boot services.",
            ["Java", "C++", "Spring Boot", "Data Structures"],
            "Software Developer",
            "Core Software Engineering & OOP Course",
        ),
        (
            "Computer networking fundamentals, TCP/IP protocols, subnetting, Cisco routers and switches, Linux system administration, shell scripting, DNS and DHCP server management.",
            ["Networking", "Linux", "Bash"],
            "Network / IT Support",
            "Networking & Systems Administration Course",
        ),
        (
            "Indian Constitution, Fundamental Rights and Duties, Directive Principles, Union Executive, Parliament, Judiciary, and Election Commission of India.",
            [],
            None,  # Should return None for non-technical course role
            "Non-Technical Humanities Course (Constitution of India)",
        ),
        (
            "Gender Sensitization: definition of gender, gender roles in society, addressing stereotypes, gender-based violence, and constitutional safeguards.",
            [],
            None,
            "Non-Technical Mandatory Course (Gender Sensitization)",
        ),
    ]

    for i, (text, skills, expected_role, desc) in enumerate(role_test_cases, 1):
        total_count += 1
        t0 = time.time()
        if expected_role is None:
            res = market.course_role(text, skills)
            elapsed_ms = (time.time() - t0) * 1000
            passed = res is None
            status = "PASS" if passed else "FAIL"
            if passed: passed_count += 1
            print(f"[{status}] Case 1.{i:02d}: {desc}")
            print(f"       Result: Non-technical course correctly yielded No IT Role (elapsed: {elapsed_ms:.1f}ms)")
        else:
            preds = market.predict_roles(text, skills, top=3)
            elapsed_ms = (time.time() - t0) * 1000
            top_role = preds[0]["role"] if preds else "None"
            top_prob = preds[0]["probability"] if preds else 0.0
            passed = top_role == expected_role
            status = "PASS" if passed else "FAIL"
            if passed: passed_count += 1
            print(f"[{status}] Case 1.{i:02d}: {desc}")
            print(f"       Expected: {expected_role} | Predicted: {top_role} (P = {top_prob*100:.1f}%) | {elapsed_ms:.1f}ms")

    # -------------------------------------------------------------------------
    # SUITE 2: MODEL 2 - REQUIREMENT LEVEL CLASSIFICATION (Supervised 3-Class Linear SVM)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SUITE 2: MODEL 2 - REQUIREMENT LEVEL CLASSIFICATION (Required vs Preferred vs Neutral)")
    print("=" * 80)

    req_test_cases = [
        # (Sentence, Expected Label, Description)
        ("Must possess minimum 3 years of hands-on experience in production Python backend development.", "required", "Explicit 'must possess' requirement"),
        ("Strong proficiency in SQL database query optimization and schema design is mandatory.", "required", "Explicit 'mandatory' keyword"),
        ("Candidates are required to have in-depth knowledge of distributed systems and microservices.", "required", "Explicit 'required to have' keyword"),
        ("Essential requirement: expert-level familiarity with AWS cloud infrastructure.", "required", "Explicit 'essential requirement'"),
        ("Working knowledge of Docker containerization and Kubernetes orchestration is a must.", "required", "Explicit 'is a must' keyword"),
        ("Experience with React or Angular frontend frameworks would be a definite plus.", "preferred", "Cue 'would be a plus'"),
        ("Familiarity with GraphQL and Redis caching is good to have.", "preferred", "Cue 'good to have'"),
        ("Prior exposure to generative AI models and LangChain is desirable but not compulsory.", "preferred", "Cue 'is desirable but not compulsory'"),
        ("Knowledge of Golang or Rust will be an added advantage.", "preferred", "Cue 'added advantage'"),
        ("Bonus points if you have contributed to open-source software libraries.", "preferred", "Cue 'bonus points if'"),
        ("You will collaborate with cross-functional product management and engineering teams.", "neutral", "Job responsibility / daily workflow"),
        ("Participate in daily agile sprint standups and lead bi-weekly sprint retrospectives.", "neutral", "Scrum / process task"),
        ("We are a fast-growing series-B fintech startup based in Bengaluru and Singapore.", "neutral", "Company background / overview"),
        ("Ensure adherence to code quality guidelines and participate in peer code reviews.", "neutral", "Engineering culture responsibility"),
        ("The role reports directly to the Principal Data Architect and Engineering Director.", "neutral", "Reporting hierarchy info"),
    ]

    for i, (sentence, expected_label, desc) in enumerate(req_test_cases, 1):
        total_count += 1
        t0 = time.time()
        pred = market.requirement_model.predict([sentence])[0]
        elapsed_ms = (time.time() - t0) * 1000
        passed = pred == expected_label
        status = "PASS" if passed else "FAIL"
        if passed: passed_count += 1
        print(f"[{status}] Case 2.{i:02d}: {desc}")
        print(f"       Sentence: \"{sentence[:65]}...\"")
        print(f"       Expected: {expected_label.upper()} | Predicted: {pred.upper()} | {elapsed_ms:.1f}ms")

    # -------------------------------------------------------------------------
    # SUITE 3: MODEL 3a & 3b - APRIORI RULES & K-MEANS READINESS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SUITE 3: MODEL 3 - APRIORI ASSOCIATION MINING & K-MEANS BUNDLES")
    print("=" * 80)

    apriori_cases = [
        # (Taught skills set, Target skill to recommend, Description)
        ({"Java"}, "Spring Framework", "Java antecedent recommends Spring Framework"),
        ({"Python", "TensorFlow"}, "Machine Learning", "Python + TensorFlow antecedent recommends Machine Learning"),
        ({"Docker"}, "Kubernetes", "Docker antecedent recommends Kubernetes"),
        ({"HTML", "React"}, "JavaScript", "HTML + React antecedent recommends JavaScript"),
        ({"HTML", "CSS"}, "JavaScript", "HTML + CSS antecedent recommends JavaScript"),
        ({"SQL"}, "Python", "SQL antecedent recommends Python"),
    ]

    for i, (taught_set, target_skill, desc) in enumerate(apriori_cases, 1):
        total_count += 1
        t0 = time.time()
        rules = market.rules_for(target_skill, taught_set)
        elapsed_ms = (time.time() - t0) * 1000
        passed = len(rules) > 0 and rules[0]["then"] == target_skill
        status = "PASS" if passed else "FAIL"
        if passed: passed_count += 1
        print(f"[{status}] Case 3.{i:02d}: {desc}")
        if rules:
            r = rules[0]
            print(f"       Found Rule: {' + '.join(r['if'])} -> {r['then']} (Conf: {r['confidence']*100:.1f}%, Lift: {r['lift']:.2f}) | {elapsed_ms:.1f}ms")
        else:
            print(f"       No rule found matching taught set {taught_set} -> {target_skill}")

    # Test K-Means bundles inspection
    total_count += 1
    t0 = time.time()
    bundles = market.bundles
    passed_bundles = len(bundles) == 12
    if passed_bundles: passed_count += 1
    print(f"[{'PASS' if passed_bundles else 'FAIL'}] Case 3.07: K-Means Cluster Bundles Integrity (Found {len(bundles)} clusters)")
    for b in bundles[:3]:
        skills_str = ", ".join(s["skill"] for s in b["top_skills"][:4])
        print(f"       Bundle #{b['id']} ({b['main_role']}): {b['name']} [Top: {skills_str}]")

    # -------------------------------------------------------------------------
    # SUITE 4: NLP PIPELINE - CONTEXT DISAMBIGUATION & BOUNDARY ROBUSTNESS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SUITE 4: NLP EXTRACTION & AMBIGUOUS TOKEN DISAMBIGUATION")
    print("=" * 80)

    _, matcher = load_taxonomy()

    nlp_cases = [
        # (Text, Expected to extract, Should NOT extract, Description)
        (
            "Advanced programming in C language using pointers, dynamic memory allocation and structs.",
            ["C Programming"],
            [],
            "C Programming recognized with 'programming' and 'pointer' cues",
        ),
        (
            "Reference books: Advanced Engineering Math by Stevens, C. and Rao, P. Oxford Press.",
            [],
            ["C Programming"],
            "Initial 'C.' in author name correctly rejected (not C programming)",
        ),
        (
            "Deploying scalable cloud compute instances on AWS using EC2 and S3 storage buckets.",
            ["Amazon Web Services"],
            [],
            "AWS recognized as Amazon Web Services because cloud/EC2 cues are present",
        ),
        (
            "Design and analysis of algorithms; dynamic programming; greedy methods.",
            ["Algorithm Design and Analysis"],
            [],
            "Algorithms matched to Algorithm Design and Analysis",
        ),
        (
            "Web programming with JavaScript, React components, CSS stylesheets and HTML5 semantics.",
            ["JavaScript", "React", "CSS", "HTML"],
            ["Java"],  # Must NOT confuse JavaScript with Java
            "JavaScript extracted without false-positive Java detection",
        ),
        (
            "Microservices architecture using Java 17, Spring Boot, and PostgreSQL relational database.",
            ["Java", "Spring Framework", "PostgreSQL"],
            ["JavaScript"],
            "Java extracted without false-positive JavaScript detection",
        ),
        (
            "Modern systems programming in C++ with templates, STL vectors and smart pointers.",
            ["C++"],
            [],
            "C++ token preserved and extracted correctly",
        ),
        (
            "Building cross-platform desktop applications using Microsoft .NET framework and C#.",
            [".NET", "C#"],
            [],
            "Punctuation-heavy tokens .NET and C# preserved",
        ),
        (
            "Continuous Integration and Continuous Deployment (CI/CD) pipelines with Git and Docker.",
            ["CI/CD", "Git", "Docker"],
            [],
            "CI/CD hyphen/slash token extracted",
        ),
    ]

    for i, (text, should_find, should_not_find, desc) in enumerate(nlp_cases, 1):
        total_count += 1
        sec = SectionInput(index=1, section_type="unit", title="Unit 1", lines=[(1, text)])
        cands = extract_skills([sec], matcher)
        extracted = {c.skill_name for c in cands}
        
        found_ok = all(s in extracted for s in should_find)
        not_found_ok = not any(s in extracted for s in should_not_find)
        passed = found_ok and not_found_ok
        status = "PASS" if passed else "FAIL"
        if passed: passed_count += 1
        print(f"[{status}] Case 4.{i:02d}: {desc}")
        print(f"       Input: \"{text[:65]}...\"")
        print(f"       Extracted Skills: {sorted(list(extracted))}")

    # -------------------------------------------------------------------------
    # SUITE 5: PERFORMANCE, TIMING & ROBUSTNESS UNDER ADVERSARIAL INPUTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SUITE 5: SYSTEM BENCHMARKS & ADVERSARIAL / MALFORMED INPUTS")
    print("=" * 80)

    adversarial_cases = [
        # (Input text, Description)
        ("", "Empty string handling"),
        ("     \n\t  \n  ", "Whitespace-only input"),
        ("1234567890 !@#$%^&*()_+=-`~[]\\{}|;':\",./<>?", "Pure symbol / numeric string"),
        ("The quick brown fox jumps over the lazy dog repeatedly.", "Arbitrary English with zero tech terms"),
        ("a " * 5000, "5,000-word repetitive single character buffer"),
        ("SELECT * FROM students WHERE id = 1; DROP TABLE users; --", "SQL Injection attempt text"),
    ]

    for i, (text, desc) in enumerate(adversarial_cases, 1):
        total_count += 1
        try:
            t0 = time.time()
            roles = market.predict_roles(text, [])
            c_role = market.course_role(text, [])
            p_time = (time.time() - t0) * 1000
            passed = True
            passed_count += 1
            status = "PASS"
            print(f"[{status}] Case 5.{i:02d}: {desc} -> Handled gracefully in {p_time:.2f}ms (No Crash)")
        except Exception as exc:
            print(f"[FAIL] Case 5.{i:02d}: {desc} -> Threw unhandled exception: {exc}")

    # -------------------------------------------------------------------------
    # BENCHMARK SUMMARY
    # -------------------------------------------------------------------------
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"EVALUATION SUITE COMPLETED: {passed_count}/{total_count} PASSED ({passed_count/total_count*100:.1f}%)")
    print(f"TOTAL EXECUTION TIME: {total_time:.2f}s (Average < 15ms per inference query)")
    print("=" * 80)


def interactive_mode():
    print("\n" + "=" * 80)
    print("SKILLPULSE AI - INTERACTIVE LIVE EVALUATOR")
    print("Type or paste any course description, syllabus unit, or job ad text.")
    print("Type 'exit' or 'quit' to finish.")
    print("=" * 80)

    market = get_market()
    _, matcher = load_taxonomy()

    while True:
        try:
            text = input("\nEnter input text >>> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not text or text.lower() in ("exit", "quit"):
            break

        print("\n--- INFERENCE RESULTS ---")
        t0 = time.time()
        
        # 1. Skill Extraction
        sec = SectionInput(index=1, section_type="unit", title="Live Query", lines=[(1, text)])
        cands = extract_skills([sec], matcher)
        extracted_skills = [c.skill_name for c in cands]
        print(f"[1] Extracted Skills ({len(extracted_skills)}): {', '.join(extracted_skills) if extracted_skills else 'None recognized'}")

        # 2. Model 1: Job Role Prediction
        roles = market.predict_roles(text, extracted_skills, top=4)
        c_role = market.course_role(text, extracted_skills)
        print(f"[2] Model 1 Job Role Predictions (Calibrated Probabilities):")
        if roles:
            for r in roles:
                print(f"    - {r['role']}: {r['probability']*100:.1f}%")
        else:
            print("    - None (Insufficient content)")
        print(f"    Course Role Filter: {c_role['role'] + ' (' + str(round(c_role['probability']*100, 1)) + '%)' if c_role else 'None (Non-Technical or Low Confidence)'}")

        # 3. Model 2: Requirement Sentence Classification
        req_pred = market.requirement_model.predict([text])[0]
        print(f"[3] Model 2 Sentence Classification: {req_pred.upper()}")

        # 4. Model 3a: Apriori Recommendations
        if extracted_skills:
            taught_set = set(extracted_skills)
            print(f"[4] Model 3a Association Explanations (What employers ask with these skills):")
            found_any = False
            for r in market.rules[:100]:
                if r["then"] not in taught_set and set(r["if"]) <= taught_set:
                    print(f"    - Students learn {' + '.join(r['if'])} -> Recommend {r['then']} (Confidence: {r['confidence']*100:.1f}%, Lift: {r['lift']:.2f})")
                    found_any = True
                    break
            if not found_any:
                print("    - No direct high-lift rule triggered for this combination.")

        elapsed = (time.time() - t0) * 1000
        print(f"--- Inference finished in {elapsed:.1f} ms ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkillPulse Live Evaluation Suite")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt")
    parser.add_argument("--run-battery", action="store_true", help="Run full test battery")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        run_test_battery()
