# Privileged Access Management Runbook (IT Admins Only)

> Fictional document created for the OrbitDesk demo. Orbit Systems is not a real company.
> **Restricted to IT administrators.**

## 1. Admin Accounts

IT administrators use a separate **adm-** prefixed account (for example adm-jdoe) for all privileged work. Admin accounts must never be used for email or web browsing.

## 2. Just-in-Time Elevation

Domain admin rights are granted just-in-time through **OrbitVault** for a maximum of **4 hours** per request. Each request needs a linked ticket number and is approved by a second admin.

## 3. Break-Glass Accounts

Two break-glass accounts exist for emergencies when OrbitVault is unavailable. Their credentials are stored in a sealed envelope in the **data-centre safe**. Using a break-glass account triggers an automatic P1 security alert and requires a written justification within **24 hours**.

## 4. Access Reviews

All privileged access is reviewed **quarterly** by the IT Security Lead. Unused admin accounts are disabled after **45 days** of inactivity.

## 5. Unlocking a User Account (Admin Procedure)

Admins can unlock a locked user account from OrbitVault > Directory > Unlock. Always verify the user's identity through their manager first, and record the verification in the ticket.
