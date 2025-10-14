import { Component, type OnInit } from "@angular/core"
import { CommonModule } from "@angular/common"
import { ReactiveFormsModule, FormBuilder, type FormGroup, Validators } from "@angular/forms"
import { ActivatedRoute, Router, RouterModule } from "@angular/router"
import { MatCardModule } from "@angular/material/card"
import { MatFormFieldModule } from "@angular/material/form-field"
import { MatInputModule } from "@angular/material/input"
import { MatButtonModule } from "@angular/material/button"
import { MatIconModule } from "@angular/material/icon"
import { MatProgressSpinnerModule } from "@angular/material/progress-spinner"
import { MatSnackBar, MatSnackBarModule } from "@angular/material/snack-bar"
import { AuthService } from "../../services/auth.service"

@Component({
  selector: "app-reset-password",
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    RouterModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatSnackBarModule,
  ],
  template: `
    <div class="reset-password-container">
      <mat-card class="reset-password-card">
        <mat-card-header>
          <div class="logo-container">
            <img src="assets/logo.png" alt="Enterprise RAG Bot" class="logo">
          </div>
          <mat-card-title>Set New Password</mat-card-title>
          <mat-card-subtitle>Enter your new password below</mat-card-subtitle>
        </mat-card-header>
        
        <mat-card-content>
          <form [formGroup]="resetPasswordForm" (ngSubmit)="onSubmit()" *ngIf="!isSuccess && hasValidToken">
            <mat-form-field appearance="outline" class="full-width">
              <mat-label>New Password</mat-label>
              <input matInput formControlName="password" [type]="hidePassword ? 'password' : 'text'" autocomplete="new-password">
              <button type="button" mat-icon-button matSuffix (click)="hidePassword = !hidePassword">
                <mat-icon>{{hidePassword ? 'visibility_off' : 'visibility'}}</mat-icon>
              </button>
              <mat-error *ngIf="resetPasswordForm.get('password')?.hasError('required')">Password is required</mat-error>
              <mat-error *ngIf="resetPasswordForm.get('password')?.hasError('minlength')">Password must be at least 8 characters</mat-error>
            </mat-form-field>
            
            <mat-form-field appearance="outline" class="full-width">
              <mat-label>Confirm New Password</mat-label>
              <input matInput formControlName="confirmPassword" [type]="hideConfirmPassword ? 'password' : 'text'" autocomplete="new-password">
              <button type="button" mat-icon-button matSuffix (click)="hideConfirmPassword = !hideConfirmPassword">
                <mat-icon>{{hideConfirmPassword ? 'visibility_off' : 'visibility'}}</mat-icon>
              </button>
              <mat-error *ngIf="resetPasswordForm.get('confirmPassword')?.hasError('required')">Please confirm your password</mat-error>
              <mat-error *ngIf="resetPasswordForm.get('confirmPassword')?.hasError('passwordMismatch')">Passwords do not match</mat-error>
            </mat-form-field>
            
            <div class="password-requirements">
              <h4>Password Requirements:</h4>
              <ul>
                <li [class.met]="hasMinLength">At least 8 characters</li>
                <li [class.met]="hasUppercase">One uppercase letter</li>
                <li [class.met]="hasLowercase">One lowercase letter</li>
                <li [class.met]="hasNumber">One number</li>
              </ul>
            </div>
            
            <div class="form-actions">
              <button mat-raised-button color="primary" type="submit" [disabled]="resetPasswordForm.invalid || isLoading" class="full-width">
                <mat-spinner diameter="20" *ngIf="isLoading"></mat-spinner>
                <span *ngIf="!isLoading">Reset Password</span>
              </button>
            </div>
          </form>
          
          <div *ngIf="isSuccess" class="success-message">
            <mat-icon color="primary" class="large-icon">check_circle</mat-icon>
            <h3>Password Reset Successfully!</h3>
            <p>Your password has been updated. You can now log in with your new password.</p>
            <button mat-raised-button color="primary" (click)="goToLogin()">
              Go to Login
            </button>
          </div>
          
          <div *ngIf="!hasValidToken" class="error-message">
            <mat-icon color="warn" class="large-icon">error</mat-icon>
            <h3>Invalid Reset Link</h3>
            <p>This password reset link is invalid or has expired. Please request a new one.</p>
            <button mat-raised-button color="primary" routerLink="/forgot-password">
              Request New Link
            </button>
          </div>
        </mat-card-content>
        
        <mat-card-actions>
          <div class="back-to-login">
            <a [routerLink]="['/login']" mat-button>
              <mat-icon>arrow_back</mat-icon>
              Back to Login
            </a>
          </div>
        </mat-card-actions>
      </mat-card>
    </div>
  `,
  styles: [
    `
    .reset-password-container {
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      padding: 20px;
      background-color: var(--background);
    }
    
    .reset-password-card {
      width: 100%;
      max-width: 500px;
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    
    .logo-container {
      display: flex;
      justify-content: center;
      width: 100%;
      margin-bottom: 20px;
    }
    
    .logo {
      width: 80px;
      height: 80px;
      object-fit: contain;
    }
    
    mat-card-header {
      display: flex;
      flex-direction: column;
      align-items: center;
      margin-bottom: 20px;
      text-align: center;
    }
    
    mat-card-title {
      margin-top: 10px;
      font-size: 24px;
      font-weight: 500;
    }
    
    mat-card-subtitle {
      margin-bottom: 20px;
      color: var(--text-secondary);
    }
    
    .full-width {
      width: 100%;
      margin-bottom: 16px;
    }
    
    .password-requirements {
      margin: 20px 0;
      padding: 16px;
      background-color: var(--card-background);
      border: 1px solid var(--divider);
      border-radius: 8px;
    }
    
    .password-requirements h4 {
      margin: 0 0 12px 0;
      font-size: 14px;
      font-weight: 500;
    }
    
    .password-requirements ul {
      margin: 0;
      padding-left: 20px;
      list-style: none;
    }
    
    .password-requirements li {
      margin: 8px 0;
      font-size: 14px;
      color: var(--text-secondary);
      position: relative;
    }
    
    .password-requirements li::before {
      content: '✗';
      position: absolute;
      left: -20px;
      color: var(--warn);
      font-weight: bold;
    }
    
    .password-requirements li.met {
      color: var(--primary);
    }
    
    .password-requirements li.met::before {
      content: '✓';
      color: var(--primary);
    }
    
    .form-actions {
      margin-top: 24px;
      margin-bottom: 24px;
    }
    
    .success-message, .error-message {
      text-align: center;
      padding: 20px;
    }
    
    .large-icon {
      font-size: 64px;
      width: 64px;
      height: 64px;
      margin-bottom: 16px;
    }
    
    .success-message h3, .error-message h3 {
      margin: 16px 0;
      color: var(--text-primary);
    }
    
    .success-message p, .error-message p {
      margin: 12px 0;
      color: var(--text-secondary);
    }
    
    .back-to-login {
      display: flex;
      justify-content: center;
      width: 100%;
      padding: 16px;
    }
  `,
  ],
})
export class ResetPasswordComponent implements OnInit {
  resetPasswordForm: FormGroup
  isLoading = false
  isSuccess = false
  hasValidToken = true
  hidePassword = true
  hideConfirmPassword = true
  resetToken = ""

  // Password strength indicators
  hasMinLength = false
  hasUppercase = false
  hasLowercase = false
  hasNumber = false

  constructor(
    private fb: FormBuilder,
    private route: ActivatedRoute,
    private router: Router,
    private authService: AuthService,
    private snackBar: MatSnackBar,
  ) {
    this.resetPasswordForm = this.fb.group(
      {
        password: ["", [Validators.required, Validators.minLength(8)]],
        confirmPassword: ["", Validators.required],
      },
      { validators: this.passwordMatchValidator },
    )

    // Watch password changes for strength indicators
    this.resetPasswordForm.get("password")?.valueChanges.subscribe((password) => {
      this.updatePasswordStrength(password)
    })
  }

  ngOnInit(): void {
    this.resetToken = this.route.snapshot.queryParams["token"]
    if (!this.resetToken) {
      this.hasValidToken = false
    }
  }

  passwordMatchValidator(form: FormGroup) {
    const password = form.get("password")?.value
    const confirmPassword = form.get("confirmPassword")?.value

    if (password !== confirmPassword) {
      form.get("confirmPassword")?.setErrors({ passwordMismatch: true })
      return { passwordMismatch: true }
    }

    return null
  }

  updatePasswordStrength(password: string): void {
    this.hasMinLength = password.length >= 8
    this.hasUppercase = /[A-Z]/.test(password)
    this.hasLowercase = /[a-z]/.test(password)
    this.hasNumber = /\d/.test(password)
  }

  onSubmit(): void {
    if (this.resetPasswordForm.invalid) {
      return
    }

    this.isLoading = true
    const newPassword = this.resetPasswordForm.get("password")?.value

    this.authService.resetPassword(this.resetToken, newPassword).subscribe({
      next: () => {
        this.isSuccess = true
        this.isLoading = false
      },
      error: (error) => {
        this.isLoading = false
        if (error.status === 400) {
          this.hasValidToken = false
        } else {
          this.snackBar.open(error.error?.detail || "Failed to reset password. Please try again.", "Close", {
            duration: 5000,
          })
        }
      },
    })
  }

  goToLogin(): void {
    this.router.navigate(["/login"])
  }
}
