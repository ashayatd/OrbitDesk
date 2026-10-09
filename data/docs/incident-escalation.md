# IT Incident Escalation Runbook

> Fictional document created for the OrbitDesk demo. Orbit Systems is not a real company.

## 1. Priority Levels

| Priority | Definition | Response target | Resolution target |
|---|---|---|---|
| P1 Critical | Company-wide outage or security breach | 15 minutes | 4 hours |
| P2 High | A whole team or key system affected | 1 hour | 8 hours |
| P3 Medium | One user blocked, workaround exists | 4 hours | 3 working days |
| P4 Low | Question or minor issue | 1 working day | 5 working days |

## 2. How to Report an Incident

Raise an IT ticket in the Service Portal or through OrbitDesk. For **P1 incidents call the Service Desk on extension 4357** in addition to raising a ticket.

## 3. Escalation Path

### 3.1 Level 1: Service Desk

The Service Desk triages every ticket and resolves common issues (passwords, VPN, software installs).

### 3.2 Level 2: Platform Team

If L1 cannot resolve the issue within the response target, it escalates to the **Platform Team**.

### 3.3 Level 3: Engineering On-Call

P1 and P2 incidents that L2 cannot resolve within **1 hour** are escalated to the **Engineering On-Call** engineer via the paging system.

## 4. Security Incidents

Suspected phishing, malware or data leaks are always treated as **P1**. Do not investigate yourself: disconnect from the network and report to security@orbit.example immediately.

## 5. Communication

During a P1, the Incident Manager posts updates in **#status-incidents** every **30 minutes** until resolved. A post-incident review is written within **5 working days**.
