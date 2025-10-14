import { Injectable } from "@angular/core"
import { HttpClient } from "@angular/common/http"
import { BehaviorSubject, type Observable, of } from "rxjs"
import { catchError, tap } from "rxjs/operators"
import { Router } from "@angular/router"

export interface User {
  id: string
  email: string
  name: string
  avatar?: string
  provider?: string
  role: "user" | "admin"
  is_verified: boolean
  theme: string
  language: string
  timezone: string
  notifications_enabled: boolean
  email_notifications: boolean
  bio?: string
  location?: string
  website?: string
  company?: string
  job_title?: string
  login_count: number
  created_at: Date
  last_login: Date
  two_factor_enabled?: boolean;
}

export interface LoginRequest {
  email: string
  password: string
}

export interface RegisterRequest {
  name: string
  email: string
  password: string
}

export interface PasswordResetRequest {
  email: string
}

export interface PasswordResetConfirm {
  token: string
  new_password: string
}

export interface UserPreferences {
  theme?: string
  language?: string
  timezone?: string
  notifications_enabled?: boolean
  email_notifications?: boolean
}

export interface UserProfileUpdate {
  name?: string
  bio?: string
  location?: string
  website?: string
  company?: string
  job_title?: string
}

export interface AuthResponse {
  user: User
  token: string
  refresh_token?: string
}

@Injectable({
  providedIn: "root",
})
export class AuthService {
  private baseUrl = "http://localhost:8000/api/auth"
  private currentUserSubject = new BehaviorSubject<User | null>(null)
  public currentUser$ = this.currentUserSubject.asObservable()
  private tokenKey = "auth_token"
  private refreshTokenKey = "refresh_token"

  constructor(
    private http: HttpClient,
    private router: Router,
  ) {
    this.loadUserFromStorage()
  }

  private loadUserFromStorage(): void {
    const token = localStorage.getItem(this.tokenKey)
    if (token) {
      try {
        const base64Url = token.split(".")[1]
        const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/")
        const payload = JSON.parse(window.atob(base64))

        if (payload && payload.user) {
          this.currentUserSubject.next(payload.user)
        }
      } catch (e) {
        console.error("Error parsing token", e)
        this.logout()
      }
    }
  }

  register(userData: RegisterRequest): Observable<any> {
    return this.http.post(`${this.baseUrl}/register`, userData).pipe(
      catchError((error) => {
        console.error("Registration error", error)
        throw error
      }),
    )
  }

  login(credentials: LoginRequest): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.baseUrl}/login`, credentials).pipe(
      tap((response) => this.handleAuthentication(response)),
      catchError((error) => {
        console.error("Login error", error)
        throw error
      }),
    )
  }

  verifyEmail(token: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/verify-email`, null, {
      params: { token },
    })
  }

  forgotPassword(email: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/forgot-password`, { email })
  }

  resetPassword(token: string, new_password: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/reset-password`, {
      token,
      new_password,
    })
  }

  updatePreferences(preferences: UserPreferences): Observable<any> {
    return this.http.put(`${this.baseUrl}/preferences`, preferences)
  }

  updateProfile(profile: UserProfileUpdate): Observable<any> {
    return this.http.put(`${this.baseUrl}/profile`, profile)
  }

  loginWithGoogle(): void {
    window.location.href = `${this.baseUrl}/google`
  }

  loginWithGithub(): void {
    window.location.href = `${this.baseUrl}/github`
  }

  handleSocialCallback(token: string, refreshToken?: string): void {
    if (token) {
      localStorage.setItem(this.tokenKey, token)
      if (refreshToken) {
        localStorage.setItem(this.refreshTokenKey, refreshToken)
      }
      try {
        const base64Url = token.split(".")[1]
        const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/")
        const payload = JSON.parse(window.atob(base64))

        if (payload && payload.user) {
          this.currentUserSubject.next(payload.user)
        }
      } catch (e) {
        console.error("Error parsing social auth token", e)
      }
    }
  }

  logout(): void {
    localStorage.removeItem(this.tokenKey)
    localStorage.removeItem(this.refreshTokenKey)
    this.currentUserSubject.next(null)
    this.router.navigate(["/login"])
  }

  refreshToken(): Observable<AuthResponse> {
    const refreshToken = localStorage.getItem(this.refreshTokenKey)
    if (!refreshToken) {
      return of(null as any)
    }

    return this.http.post<AuthResponse>(`${this.baseUrl}/refresh-token`, { refreshToken }).pipe(
      tap((response) => this.handleAuthentication(response)),
      catchError((error) => {
        console.error("Token refresh error", error)
        this.logout()
        throw error
      }),
    )
  }

  getCurrentUser(): Observable<User> {
    return this.http.get<User>(`${this.baseUrl}/me`)
  }

  isAuthenticated(): boolean {
    return !!this.currentUserSubject.value
  }

  isAdmin(): boolean {
    const user = this.currentUserSubject.value
    return user?.role === "admin"
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey)
  }

  private handleAuthentication(response: AuthResponse): void {
    if (response && response.token) {
      localStorage.setItem(this.tokenKey, response.token)
      if (response.refresh_token) {
        localStorage.setItem(this.refreshTokenKey, response.refresh_token)
      }
      this.currentUserSubject.next(response.user)
    }
  }
}
