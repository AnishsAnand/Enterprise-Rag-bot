import { Component, type OnInit } from "@angular/core"
import { CommonModule } from "@angular/common"
import { ReactiveFormsModule, FormBuilder, type FormGroup, Validators } from "@angular/forms"
import { MatCardModule } from "@angular/material/card"
import { MatFormFieldModule } from "@angular/material/form-field"
import { MatInputModule } from "@angular/material/input"
import { MatButtonModule } from "@angular/material/button"
import { MatIconModule } from "@angular/material/icon"
import { MatTabsModule } from "@angular/material/tabs"
import { MatSelectModule } from "@angular/material/select"
import { MatSlideToggleModule } from "@angular/material/slide-toggle"
import { MatDividerModule } from "@angular/material/divider"
import { MatChipsModule } from "@angular/material/chips"
import {  MatSnackBar, MatSnackBarModule } from "@angular/material/snack-bar"
import { AuthService, User, UserPreferences, UserProfileUpdate } from "../../services/auth.service"
import { ThemeService } from "../../services/theme.service"

@Component({
  selector: "app-profile",
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatTabsModule,
    MatSelectModule,
    MatSlideToggleModule,
    MatDividerModule,
    MatChipsModule,
    MatSnackBarModule,
  ],
  template: `
    <div class="profile-container">
      <mat-card class="profile-card">
        <mat-card-header>
          <div class="avatar-container">
            <img [src]="user?.avatar || 'assets/default-avatar.png'" alt="User Avatar" class="avatar">
            <button mat-mini-fab color="primary" class="avatar-edit-btn">
              <mat-icon>edit</mat-icon>
            </button>
          </div>
          <div class="user-info">
            <mat-card-title>{{ user?.name }}</mat-card-title>
            <mat-card-subtitle>{{ user?.email }}</mat-card-subtitle>
            <div class="user-badges">
              <mat-chip-set>
                <mat-chip [color]="user?.role === 'admin' ? 'primary' : 'accent'">
                  {{ user?.role }}
                </mat-chip>
                <mat-chip *ngIf="user?.is_verified" color="primary">
                  <mat-icon matChipAvatar>verified</mat-icon>
                  Verified
                </mat-chip>
                <mat-chip *ngIf="user?.provider && user?.provider !== 'email'">
                  {{ user?.provider }}
                </mat-chip>
              </mat-chip-set>
            </div>
          </div>
        </mat-card-header>
        
        <mat-card-content>
          <mat-tab-group>
            
            <mat-tab label="Profile">
              <div class="tab-content">
                <form [formGroup]="profileForm" (ngSubmit)="updateProfile()">
                  <div class="form-row">
                    <mat-form-field appearance="outline" class="full-width">
                      <mat-label>Full Name</mat-label>
                      <input matInput formControlName="name">
                      <mat-icon matSuffix>person</mat-icon>
                    </mat-form-field>
                  </div>
                  
                  <div class="form-row">
                    <mat-form-field appearance="outline" class="full-width">
                      <mat-label>Bio</mat-label>
                      <textarea matInput formControlName="bio" rows="3" placeholder="Tell us about yourself..."></textarea>
                      <mat-icon matSuffix>description</mat-icon>
                    </mat-form-field>
                  </div>
                  
                  <div class="form-row">
                    <mat-form-field appearance="outline">
                      <mat-label>Location</mat-label>
                      <input matInput formControlName="location" placeholder="City, Country">
                      <mat-icon matSuffix>location_on</mat-icon>
                    </mat-form-field>
                    
                    <mat-form-field appearance="outline">
                      <mat-label>Website</mat-label>
                      <input matInput formControlName="website" placeholder="https://example.com">
                      <mat-icon matSuffix>link</mat-icon>
                    </mat-form-field>
                  </div>
                  
                  <div class="form-row">
                    <mat-form-field appearance="outline">
                      <mat-label>Company</mat-label>
                      <input matInput formControlName="company">
                      <mat-icon matSuffix>business</mat-icon>
                    </mat-form-field>
                    
                    <mat-form-field appearance="outline">
                      <mat-label>Job Title</mat-label>
                      <input matInput formControlName="job_title">
                      <mat-icon matSuffix>work</mat-icon>
                    </mat-form-field>
                  </div>
                  
                  <div class="form-actions">
                    <button mat-raised-button color="primary" type="submit" [disabled]="profileForm.invalid">
                      <mat-icon>save</mat-icon>
                      Update Profile
                    </button>
                  </div>
                </form>
              </div>
            </mat-tab>
    
            <mat-tab label="Preferences">
              <div class="tab-content">
                <form [formGroup]="preferencesForm" (ngSubmit)="updatePreferences()">
                  <h3>Appearance</h3>
                  <div class="preference-section">
                    <mat-form-field appearance="outline">
                      <mat-label>Theme</mat-label>
                      <mat-select formControlName="theme">
                        <mat-option value="light">Light</mat-option>
                        <mat-option value="dark">Dark</mat-option>
                        <mat-option value="system">System</mat-option>
                      </mat-select>
                      <mat-icon matSuffix>palette</mat-icon>
                    </mat-form-field>
                    
                    <mat-form-field appearance="outline">
                      <mat-label>Language</mat-label>
                      <mat-select formControlName="language">
                        <mat-option value="en">English</mat-option>
                        <mat-option value="es">Spanish</mat-option>
                        <mat-option value="fr">French</mat-option>
                        <mat-option value="de">German</mat-option>
                        <mat-option value="zh">Chinese</mat-option>
                      </mat-select>
                      <mat-icon matSuffix>language</mat-icon>
                    </mat-form-field>
                    
                    <mat-form-field appearance="outline">
                      <mat-label>Timezone</mat-label>
                      <mat-select formControlName="timezone">
                        <mat-option value="UTC">UTC</mat-option>
                        <mat-option value="America/New_York">Eastern Time</mat-option>
                        <mat-option value="America/Chicago">Central Time</mat-option>
                        <mat-option value="America/Denver">Mountain Time</mat-option>
                        <mat-option value="America/Los_Angeles">Pacific Time</mat-option>
                        <mat-option value="Europe/London">London</mat-option>
                        <mat-option value="Europe/Paris">Paris</mat-option>
                        <mat-option value="Asia/Tokyo">Tokyo</mat-option>
                      </mat-select>
                      <mat-icon matSuffix>schedule</mat-icon>
                    </mat-form-field>
                  </div>
                  
                  <mat-divider></mat-divider>
                  
                  <h3>Notifications</h3>
                  <div class="preference-section">
                    <div class="toggle-option">
                      <div class="toggle-info">
                        <h4>Push Notifications</h4>
                        <p>Receive notifications about system updates and important events</p>
                      </div>
                      <mat-slide-toggle formControlName="notifications_enabled"></mat-slide-toggle>
                    </div>
                    
                    <div class="toggle-option">
                      <div class="toggle-info">
                        <h4>Email Notifications</h4>
                        <p>Receive email notifications about account activity and updates</p>
                      </div>
                      <mat-slide-toggle formControlName="email_notifications"></mat-slide-toggle>
                    </div>
                  </div>
                  
                  <div class="form-actions">
                    <button mat-raised-button color="primary" type="submit">
                      <mat-icon>save</mat-icon>
                      Save Preferences
                    </button>
                  </div>
                </form>
              </div>
            </mat-tab>
         
            <mat-tab label="Security" *ngIf="user?.provider === 'email'">
              <div class="tab-content">
                <form [formGroup]="passwordForm" (ngSubmit)="changePassword()">
                  <h3>Change Password</h3>
                  <div class="security-section">
                    <mat-form-field appearance="outline" class="full-width">
                      <mat-label>Current Password</mat-label>
                      <input matInput formControlName="currentPassword" type="password">
                      <mat-icon matSuffix>lock</mat-icon>
                    </mat-form-field>
                    
                    <mat-form-field appearance="outline" class="full-width">
                      <mat-label>New Password</mat-label>
                      <input matInput formControlName="newPassword" type="password">
                      <mat-icon matSuffix>lock_open</mat-icon>
                    </mat-form-field>
                    
                    <mat-form-field appearance="outline" class="full-width">
                      <mat-label>Confirm New Password</mat-label>
                      <input matInput formControlName="confirmPassword" type="password">
                      <mat-icon matSuffix>lock_open</mat-icon>
                    </mat-form-field>
                  </div>
                  
                  <div class="form-actions">
                    <button mat-raised-button color="primary" type="submit" [disabled]="passwordForm.invalid">
                      <mat-icon>security</mat-icon>
                      Change Password
                    </button>
                  </div>
                </form>
                
                <mat-divider class="section-divider"></mat-divider>
                
                <div class="two-factor-section">
                  <h3>Two-Factor Authentication</h3>
                  <div class="toggle-option">
                    <div class="toggle-info">
                      <h4>Enable 2FA</h4>
                      <p>Add an extra layer of security to your account</p>
                    </div>
                    <mat-slide-toggle [checked]="user?.two_factor_enabled" (change)="toggle2FA($event)"></mat-slide-toggle>
                  </div>
                </div>
              </div>
            </mat-tab>
          
            <mat-tab label="Account">
              <div class="tab-content">
                <div class="account-stats">
                  <h3>Account Statistics</h3>
                  <div class="stats-grid">
                    <div class="stat-item">
                      <mat-icon>login</mat-icon>
                      <div class="stat-info">
                        <span class="stat-value">{{ user?.login_count || 0 }}</span>
                        <span class="stat-label">Total Logins</span>
                      </div>
                    </div>
                    
                    <div class="stat-item">
                      <mat-icon>event</mat-icon>
                      <div class="stat-info">
                        <span class="stat-value">{{ user?.created_at | date:'shortDate' }}</span>
                        <span class="stat-label">Member Since</span>
                      </div>
                    </div>
                    
                    <div class="stat-item">
                      <mat-icon>access_time</mat-icon>
                      <div class="stat-info">
                        <span class="stat-value">{{ user?.last_login | date:'short' }}</span>
                        <span class="stat-label">Last Login</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <mat-divider class="section-divider"></mat-divider>
                
                <div class="danger-zone">
                  <h3>Danger Zone</h3>
                  <div class="danger-actions">
                    <button mat-stroked-button color="warn" (click)="exportData()">
                      <mat-icon>download</mat-icon>
                      Export My Data
                    </button>
                    
                    <button mat-raised-button color="warn" (click)="deleteAccount()">
                      <mat-icon>delete_forever</mat-icon>
                      Delete Account
                    </button>
                  </div>
                </div>
              </div>
            </mat-tab>
          </mat-tab-group>
        </mat-card-content>
      </mat-card>
    </div>
  `,
  styles: [
    `
    .profile-container {
      padding: 20px;
      max-width: 900px;
      margin: 0 auto;
    }
    
    .profile-card {
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    mat-card-header {
      display: flex;
      align-items: center;
      padding: 24px;
      gap: 24px;
    }
    
    .avatar-container {
      position: relative;
    }
    
    .avatar {
      width: 120px;
      height: 120px;
      border-radius: 50%;
      object-fit: cover;
      border: 4px solid var(--primary);
    }
    
    .avatar-edit-btn {
      position: absolute;
      bottom: 0;
      right: 0;
    }
    
    .user-info {
      flex: 1;
    }
    
    .user-badges {
      margin-top: 12px;
    }
    
    .tab-content {
      padding: 24px 16px;
    }
    
    .form-row {
      display: flex;
      gap: 20px;
      margin-bottom: 20px;
    }
    
    .full-width {
      width: 100%;
    }
    
    .form-actions {
      margin-top: 32px;
      display: flex;
      gap: 16px;
    }
    
    .preference-section {
      margin: 20px 0;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    
    .toggle-option {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px;
      border: 1px solid var(--divider);
      border-radius: 8px;
    }
    
    .toggle-info h4 {
      margin: 0 0 4px 0;
      font-weight: 500;
    }
    
    .toggle-info p {
      margin: 0;
      font-size: 14px;
      color: var(--text-secondary);
    }
    
    .security-section {
      margin: 20px 0;
    }
    
    .two-factor-section {
      margin-top: 32px;
    }
    
    .section-divider {
      margin: 32px 0;
    }
    
    .account-stats {
      margin-bottom: 32px;
    }
    
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-top: 20px;
    }
    
    .stat-item {
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 20px;
      background-color: var(--card-background);
      border: 1px solid var(--divider);
      border-radius: 8px;
    }
    
    .stat-item mat-icon {
      font-size: 32px;
      width: 32px;
      height: 32px;
      color: var(--primary);
    }
    
    .stat-info {
      display: flex;
      flex-direction: column;
    }
    
    .stat-value {
      font-size: 18px;
      font-weight: 500;
      color: var(--text-primary);
    }
    
    .stat-label {
      font-size: 14px;
      color: var(--text-secondary);
    }
    
    .danger-zone {
      padding: 24px;
      border: 2px solid var(--warn);
      border-radius: 8px;
      background-color: rgba(244, 67, 54, 0.05);
    }
    
    .danger-zone h3 {
      color: var(--warn);
      margin-top: 0;
      margin-bottom: 16px;
    }
    
    .danger-actions {
      display: flex;
      gap: 16px;
    }
    
    @media (max-width: 768px) {
      mat-card-header {
        flex-direction: column;
        text-align: center;
      }
      
      .form-row {
        flex-direction: column;
      }
      
      .stats-grid {
        grid-template-columns: 1fr;
      }
      
      .danger-actions {
        flex-direction: column;
      }
    }
  `,
  ],
})
export class ProfileComponent implements OnInit {
  user: User | null = null
  profileForm: FormGroup
  preferencesForm: FormGroup
  passwordForm: FormGroup

  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private themeService: ThemeService,
    private snackBar: MatSnackBar,
  ) {
    this.profileForm = this.fb.group({
      name: ["", Validators.required],
      bio: [""],
      location: [""],
      website: [""],
      company: [""],
      job_title: [""],
    })

    this.preferencesForm = this.fb.group({
      theme: ["system"],
      language: ["en"],
      timezone: ["UTC"],
      notifications_enabled: [true],
      email_notifications: [true],
    })

    this.passwordForm = this.fb.group(
      {
        currentPassword: ["", Validators.required],
        newPassword: ["", [Validators.required, Validators.minLength(8)]],
        confirmPassword: ["", Validators.required],
      },
      { validators: this.passwordMatchValidator },
    )
  }

  ngOnInit(): void {
    this.authService.currentUser$.subscribe((user) => {
      this.user = user
      if (user) {
        this.profileForm.patchValue({
          name: user.name,
          bio: user.bio,
          location: user.location,
          website: user.website,
          company: user.company,
          job_title: user.job_title,
        })

        this.preferencesForm.patchValue({
          theme: user.theme,
          language: user.language,
          timezone: user.timezone,
          notifications_enabled: user.notifications_enabled,
          email_notifications: user.email_notifications,
        })
      }
    })
  }

  passwordMatchValidator(form: FormGroup) {
    const newPassword = form.get("newPassword")?.value
    const confirmPassword = form.get("confirmPassword")?.value

    if (newPassword !== confirmPassword) {
      form.get("confirmPassword")?.setErrors({ passwordMismatch: true })
      return { passwordMismatch: true }
    }

    return null
  }

  updateProfile(): void {
    if (this.profileForm.invalid) {
      return
    }

    const profileData: UserProfileUpdate = this.profileForm.value

    this.authService.updateProfile(profileData).subscribe({
      next: () => {
        this.snackBar.open("Profile updated successfully!", "Close", {
          duration: 3000,
        })
      },
      error: (error) => {
        this.snackBar.open(error.error?.detail || "Failed to update profile", "Close", {
          duration: 5000,
        })
      },
    })
  }

  updatePreferences(): void {
    const preferencesData: UserPreferences = this.preferencesForm.value

    this.authService.updatePreferences(preferencesData).subscribe({
      next: () => {
        // Update theme service if theme changed
        if (preferencesData.theme) {
          this.themeService.setTheme(preferencesData.theme as any)
        }

        this.snackBar.open("Preferences updated successfully!", "Close", {
          duration: 3000,
        })
      },
      error: (error) => {
        this.snackBar.open(error.error?.detail || "Failed to update preferences", "Close", {
          duration: 5000,
        })
      },
    })
  }

  changePassword(): void {
    if (this.passwordForm.invalid) {
      return
    }

    // In a real app, you would call an API to change the password
    this.snackBar.open("Password changed successfully!", "Close", {
      duration: 3000,
    })

    this.passwordForm.reset()
  }

  toggle2FA(event: any): void {
    const enabled = event.checked
    // In a real app, you would call an API to enable/disable 2FA
    this.snackBar.open(`Two-factor authentication ${enabled ? "enabled" : "disabled"}`, "Close", {
      duration: 3000,
    })
  }

  exportData(): void {
    // In a real app, you would call an API to export user data
    this.snackBar.open("Data export request submitted. You'll receive an email when ready.", "Close", {
      duration: 5000,
    })
  }

  deleteAccount(): void {
    const confirm = window.confirm(
      "Are you sure you want to delete your account? This action cannot be undone and all your data will be permanently deleted.",
    )

    if (confirm) {
      const doubleConfirm = window.prompt('Type "DELETE" to confirm account deletion:')

      if (doubleConfirm === "DELETE") {
        // In a real app, you would call an API to delete the account
        this.snackBar.open("Account deletion request submitted.", "Close", {
          duration: 3000,
        })
      }
    }
  }
}
