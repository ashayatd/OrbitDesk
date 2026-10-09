# Password Reset Guide

> Fictional document created for the OrbitDesk demo. Orbit Systems is not a real company.

## 1. Password Rules

Passwords must be at least **14 characters** long and must not reuse any of your **last 10 passwords**. Passwords expire every **180 days**; OrbitAuth reminds you 14 days before expiry.

## 2. Self-Service Reset

### 2.1 When You Know Your Current Password

Press Ctrl+Alt+Del on a company Windows laptop and choose **Change a password**, or use **account.orbit.example > Security**.

### 2.2 When You Forgot Your Password

1. Go to **reset.orbit.example**.
2. Enter your Orbit email address.
3. Verify with the OrbitAuth push notification **or** the one-time code sent to your registered personal phone.
4. Choose a new password that meets the rules above.

The new password works everywhere (laptop, email, VPN) within **15 minutes**.

## 3. Account Lockout

Your account is locked for **30 minutes** after **5 failed sign-in attempts**. You can wait, or unlock it immediately at reset.orbit.example using MFA.

## 4. When Self-Service Is Not Possible

If you lost access to both OrbitAuth and your registered phone, call the **Service Desk on extension 4357 (HELP)**. The Service Desk will verify your identity with your manager before resetting your password. Service Desk staff will **never ask you for your current password**.

## 5. Laptop Password Out of Sync

After a reset done while off the VPN, your laptop may still accept only the old password. Sign in with the old password, connect to the VPN, then lock and unlock the screen to sync.
