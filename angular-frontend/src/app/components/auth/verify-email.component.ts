import { Component, type OnInit } from "@angular/core"
import { CommonModule } from "@angular/common"
import { ActivatedRoute, Router } from "@angular/router"
import { MatCardModule } from "@angular/material/card"
import { MatButtonModule } from "@angular/material/button"
import { MatIconModule } from "@angular/material/icon"
import { MatProgressSpinnerModule } from "@angular/material/progress-spinner"
import { AuthService } from "../../services/auth.service"

@Component({
  selector: "app-verify-email",
  standalone: true,
  imports: [CommonModule, MatCardModule, MatButtonModule, MatIconModule, MatProgressSpinnerModule],
  template: `
    <div class="verify-container">
      <mat-card class="verify-card">
        <mat-card-header>
          <div class="logo-container">
            <img src="assets/logo.png" alt="Enterprise RAG Bot" class="logo">
          </div>
          <mat-card-title>Email Verification</mat-card-title>
        </mat-card-header>
        
        <mat-card-content>
          <div *ngIf="isLoading" class="loading-state">
            <mat-spinner></mat-spinner>
            <p>Verifying your email...</p>
          </div>
          
          <div *ngIf="!isLoading && isSuccess" class="success-state">
            <mat-icon color="primary" class="large-icon">check_circle</mat-icon>
            <h2>Email Verified Successfully!</h2>
            <p>Your email has been verified. You can now log in to your account.</p>
            <button mat-raised-button color="primary" (click)="goToLogin()">
              Go to Login
            </button>
          </div>
          
          <div *ngIf="!isLoading && !isSuccess" class="error-state">
            <mat-icon color="warn" class="large-icon">error</mat-icon>
            <h2>Verification Failed</h2>
            <p>{{ errorMessage }}</p>
            <div class="actions">
              <button mat-raised-button color="primary" (click)="goToLogin()">
                Go to Login
              </button>
              <button mat-button (click)="resendVerification()">
                Resend Verification Email
              </button>
            </div>
          </div>
        </mat-card-content>
      </mat-card>
    </div>
  `,
  styles: [
    `
    .verify-container {
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      padding: 20px;
      background-color: var(--background);
    }
    
    .verify-card {
      width: 100%;
      max-width: 500px;
      text-align: center;
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
    
    .loading-state, .success-state, .error-state {
      padding: 40px 20px;
    }
    
    .large-icon {
      font-size: 64px;
      width: 64px;
      height: 64px;
      margin-bottom: 20px;
    }
    
    .actions {
      display: flex;
      gap: 16px;
      justify-content: center;
      margin-top: 24px;
    }
    
    @media (max-width: 480px) {
      .actions {
        flex-direction: column;
      }
    }
  `,
  ],
})
export class VerifyEmailComponent implements OnInit {
  isLoading = true
  isSuccess = false
  errorMessage = ""

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private authService: AuthService,
  ) {}

  ngOnInit(): void {
    const token = this.route.snapshot.queryParams["token"]

    if (token) {
      this.verifyEmail(token)
    } else {
      this.isLoading = false
      this.errorMessage = "No verification token provided."
    }
  }

  verifyEmail(token: string): void {
    this.authService.verifyEmail(token).subscribe({
      next: () => {
        this.isLoading = false
        this.isSuccess = true
      },
      error: (error) => {
        this.isLoading = false
        this.isSuccess = false
        this.errorMessage = error.error?.detail || "Verification failed. The token may be invalid or expired."
      },
    })
  }

  goToLogin(): void {
    this.router.navigate(["/login"])
  }

  resendVerification(): void {
    // In a real app, you might want to collect the email and resend verification
    this.router.navigate(["/login"], {
      queryParams: { message: "Please contact support to resend verification email." },
    })
  }
}
