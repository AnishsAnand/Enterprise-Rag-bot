import { Component, type OnInit } from "@angular/core"
import { CommonModule } from "@angular/common"
import { ActivatedRoute, Router } from "@angular/router"
import { MatProgressSpinnerModule } from "@angular/material/progress-spinner"
import { AuthService } from "../../services/auth.service"

@Component({
  selector: "app-auth-callback",
  standalone: true,
  imports: [CommonModule, MatProgressSpinnerModule],
  template: `
    <div class="callback-container">
      <mat-spinner></mat-spinner>
      <p>{{ message }}</p>
    </div>
  `,
  styles: [
    `
    .callback-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      gap: 20px;
    }
    
    p {
      font-size: 18px;
      color: var(--text-primary);
    }
  `,
  ],
})
export class AuthCallbackComponent implements OnInit {
  message = "Authenticating, please wait..."

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private authService: AuthService,
  ) {}

  ngOnInit(): void {
    this.route.queryParams.subscribe((params) => {
      const token = params["token"]
      const refreshToken = params["refresh_token"]
      const error = params["error"]

      if (error) {
        this.message = "Authentication failed. Redirecting to login..."
        setTimeout(() => {
          this.router.navigate(["/login"], {
            queryParams: { error: "Social authentication failed" },
          })
        }, 2000)
        return
      }

      if (token) {
        this.message = "Authentication successful! Redirecting..."
        this.authService.handleSocialCallback(token, refreshToken)

        setTimeout(() => {
          this.router.navigate(["/dashboard"])
        }, 1000)
      } else {
        this.message = "Invalid authentication response. Redirecting to login..."
        setTimeout(() => {
          this.router.navigate(["/login"])
        }, 2000)
      }
    })
  }
}
