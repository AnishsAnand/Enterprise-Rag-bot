import { Component } from "@angular/core"
import { CommonModule } from "@angular/common"
import { ReactiveFormsModule, FormBuilder, type FormGroup, Validators } from "@angular/forms"
import { RouterModule } from "@angular/router"
import { MatCardModule } from "@angular/material/card"
import { MatFormFieldModule } from "@angular/material/form-field"
import { MatInputModule } from "@angular/material/input"
import { MatButtonModule } from "@angular/material/button"
import { MatIconModule } from "@angular/material/icon"
import { MatProgressSpinnerModule } from "@angular/material/progress-spinner"
import { MatSnackBar, MatSnackBarModule } from "@angular/material/snack-bar"
import { AuthService } from "../../services/auth.service"

@Component({
  selector: "app-forgot-password",
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
    <div class="forgot-password-container">
      <mat-card class="forgot-password-card">
        <mat-card-header>
          <div class="logo-container">
            <img src="assets/logo.png" alt="Enterprise RAG Bot" class="logo">
          </div>
          <mat-card-title>Reset Your Password</mat-card-title>
          <mat-card-subtitle>Enter your email address and we'll send you a reset link</mat-card-subtitle>
        </mat-card-header>
        
        <mat-card-content>
          <form [formGroup]="forgotPasswordForm" (ngSubmit)="onSubmit()" *ngIf="!emailSent">
            <mat-form-field appearance="outline" class="full-width">
              <mat-label>Email Address</mat-label>
              <input matInput formControlName="email" placeholder="Enter your email" type="email" autocomplete="email">
              <mat-icon matSuffix>email</mat-icon>
              <mat-error *ngIf="forgotPasswordForm.get('email')?.hasError('required')">Email is required</mat-error>
              <mat-error *ngIf="forgotPasswordForm.get('email')?.hasError('email')">Please enter a valid email</mat-error>
            </mat-form-field>
            
            <div class="form-actions">
              <button mat-raised-button color="primary" type="submit" [disabled]="forgotPasswordForm.invalid || isLoading" class="full-width">
                <mat-spinner diameter="20" *ngIf="isLoading"></mat-spinner>
                <span *ngIf="!isLoading">Send Reset Link</span>
              </button>
            </div>
          </form>
          
          <div *ngIf="emailSent" class="success-message">
            <mat-icon color="primary" class="large-icon">mark_email_read</mat-icon>
            <h3>Check Your Email</h3>
            <p>We've sent a password reset link to <strong>{{ emailAddress }}</strong></p>
            <p>Click the link in the email to reset your password. The link will expire in 1 hour.</p>
            
            <div class="resend-section">
              <p>Didn't receive the email?</p>
              <button mat-button color="primary" (click)="resendEmail()" [disabled]="resendCooldown > 0">
                {{ resendCooldown > 0 ? 'Resend in ' + resendCooldown + 's' : 'Resend Email' }}
              </button>
            </div>
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
    .forgot-password-container {
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      padding: 20px;
      background-color: var(--background);
    }
    
    .forgot-password-card {
      width: 100%;
      max-width: 450px;
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
    
    .form-actions {
      margin-top: 24px;
      margin-bottom: 24px;
    }
    
    .success-message {
      text-align: center;
      padding: 20px;
    }
    
    .large-icon {
      font-size: 64px;
      width: 64px;
      height: 64px;
      margin-bottom: 16px;
    }
    
    .success-message h3 {
      margin: 16px 0;
      color: var(--text-primary);
    }
    
    .success-message p {
      margin: 12px 0;
      color: var(--text-secondary);
    }
    
    .resend-section {
      margin-top: 32px;
      padding-top: 20px;
      border-top: 1px solid var(--divider);
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
export class ForgotPasswordComponent {
  forgotPasswordForm: FormGroup
  isLoading = false
  emailSent = false
  emailAddress = ""
  resendCooldown = 0
  private resendTimer: any

  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private snackBar: MatSnackBar,
  ) {
    this.forgotPasswordForm = this.fb.group({
      email: ["", [Validators.required, Validators.email]],
    })
  }

  onSubmit(): void {
    if (this.forgotPasswordForm.invalid) {
      return
    }

    this.isLoading = true
    const email = this.forgotPasswordForm.get("email")?.value

    this.authService.forgotPassword(email).subscribe({
      next: () => {
        this.emailAddress = email
        this.emailSent = true
        this.isLoading = false
        this.startResendCooldown()
      },
      error: (error) => {
        this.isLoading = false
        this.snackBar.open(error.error?.detail || "Failed to send reset email. Please try again.", "Close", {
          duration: 5000,
        })
      },
    })
  }

  resendEmail(): void {
    if (this.resendCooldown > 0) {
      return
    }

    this.authService.forgotPassword(this.emailAddress).subscribe({
      next: () => {
        this.snackBar.open("Reset email sent again!", "Close", {
          duration: 3000,
        })
        this.startResendCooldown()
      },
      error: (error) => {
        this.snackBar.open("Failed to resend email. Please try again.", "Close", {
          duration: 5000,
        })
      },
    })
  }

  private startResendCooldown(): void {
    this.resendCooldown = 60 
    this.resendTimer = setInterval(() => {
      this.resendCooldown--
      if (this.resendCooldown <= 0) {
        clearInterval(this.resendTimer)
      }
    }, 1000)
  }

  ngOnDestroy(): void {
    if (this.resendTimer) {
      clearInterval(this.resendTimer)
    }
  }
}
