# VPN Setup Guide

> Fictional document created for the OrbitDesk demo. Orbit Systems is not a real company.

## 1. Overview

Orbit Systems uses **OrbitConnect VPN** for remote access to internal systems such as OrbitHR, the wiki and code repositories. The VPN is mandatory outside the office network.

## 2. Installing the Client

### 2.1 Windows

1. Open **Company Portal** and search for "OrbitConnect".
2. Click **Install**. The installer requires a restart.
3. After restarting, OrbitConnect appears in the system tray.

### 2.2 macOS

1. Open **Self Service** and install "OrbitConnect".
2. When prompted, allow the system extension in **System Settings > Privacy & Security**.

## 3. Connecting

1. Open OrbitConnect and enter the gateway address **vpn.orbit.example**.
2. Sign in with your Orbit email and password.
3. Approve the push notification in the **OrbitAuth** authenticator app (MFA).

A successful connection shows a green shield icon. Sessions disconnect automatically after **12 hours**.

## 4. Troubleshooting

### 4.1 Error VPN-809: Gateway Unreachable

Error **VPN-809** means the client cannot reach the gateway. Check your internet connection, switch off any third-party VPN, and try again. Hotel and airport Wi-Fi often block the VPN port; switch to the **"TCP 443 fallback"** profile in OrbitConnect settings.

### 4.2 Error VPN-691: Authentication Failed

Error **VPN-691** means your credentials were rejected. Check that your password has not expired (see the Password Reset Guide). After **5 failed attempts** your account is locked for 30 minutes.

### 4.3 MFA Push Not Arriving

Open OrbitAuth manually and pull to refresh. If your phone was replaced, raise an IT ticket with category **"MFA re-enrolment"**; this needs identity verification by the Service Desk.

## 5. Getting Help

If none of the above works, raise an IT ticket with category **"VPN"** and include the error code and a screenshot.
